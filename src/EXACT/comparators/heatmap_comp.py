# comparator/compare.py

"""
HeatmapComparator
=================
Compare any two or more XAI heatmap explainers across four axes:

  • Faithfulness  – Deletion AUC + Insertion AUC
  • Sharpness     – Sparsity entropy + Top-k concentration
  • Stability     – Mean heatmap deviation under noisy inputs
  • Localization  – IoU + Pointing Game (only when a GT mask is provided)

Usage
-----
    from EXACT.comparator import HeatmapComparator

    # 1. Instantiate your explainers as usual
    gradcam_exp = GradCAM(model=model)
    vit_exp     = ViTGradCAM(model=vit_model, arch="vit")

    # 2. Run them
    gradcam_result   = gradcam_exp.explain(input_tensor, method="gradcam")
    gradcampp_result = gradcam_exp.explain(input_tensor, method="gradcam++")
    vit_result       = vit_exp.explain(input_tensor)

    # 3. Pass results AND explainer objects together — comparator handles the rest
    cmp = HeatmapComparator(model, device="cuda")

    results = cmp.compare(
        entries={
            "GradCAM":   (gradcam_result,   gradcam_exp, {"method": "gradcam"}),
            "GradCAM++": (gradcampp_result,  gradcam_exp, {"method": "gradcam++"}),
            "ViT-CAM":   (vit_result,        vit_exp,     {}),
        },
        input_tensor=input_tensor,
        input_image=input_image,   # optional
        gt_mask=gt_mask,           # optional
        stability_runs=10,
        noise_std=0.05,
    )

    cmp.report(results)                      # console table + ranked summary
    cmp.plot(results, save_png=True)         # 4-panel visual report

Entry format
------------
Each value in `entries` is a 3-tuple:

    (result_dict, explainer_object, extra_kwargs)

  result_dict      – dict returned by explainer.explain(). Must contain 'heatmap'.
  explainer_object – the explainer instance; must expose
                     .explain(input_tensor, **extra_kwargs).
                     Pass None to skip stability for that method.
  extra_kwargs     – forwarded to .explain() on every stability re-run.
                     GradCAM  → {"method": "gradcam"}
                     ViTGradCAM / DFF → {}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_hw(cam: Any) -> np.ndarray:
    """Accept (H,W) ndarray or (1,H,W)/(H,W,1) tensor/array → (H,W) float32."""
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
    cam = np.array(cam, dtype=np.float32)
    if cam.ndim == 3:
        cam = cam.squeeze(0) if cam.shape[0] == 1 else cam.squeeze(-1)
    if cam.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {cam.shape}")
    return cam


def _normalize(cam: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    mn, mx = cam.min(), cam.max()
    if mx - mn < 1e-8:
        return np.zeros_like(cam)
    return (cam - mn) / (mx - mn)


def _resize_to(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    if cam.shape == (h, w):
        return cam
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def _tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) or (C,H,W) tensor → (H,W,3) float32 in [0,1]."""
    img = t[0] if t.ndim == 4 else t
    img = img.cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return _to_display_image(img)


def _to_display_image(img: np.ndarray) -> np.ndarray:
    """
    Robustly convert any image array to float32 in [0, 1] for display.

    Handles:
      - uint8   [0, 255]  → divide by 255
      - float32 [0, 255]  → divide by 255
      - float32 [0, 1]    → pass through
      - float32 with negative values (e.g. ImageNet-normalised tensors)
                          → shift+scale to [0, 1] via min-max

    This prevents the solid-white overlay bug caused by blending a
    [0,1] heatmap with an unnormalised [0,255] image.
    """
    img = np.array(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] == 3:        # (C,H,W) → (H,W,C)
        img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img / 255.0
    if img.min() < 0.0:                             # e.g. ImageNet-normalised
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics  (all return float in [0,1], higher = better)
# ---------------------------------------------------------------------------

class _Metrics:

    # ── Faithfulness ─────────────────────────────────────────────────────

    @staticmethod
    def deletion_auc(model, input_tensor, cam, steps, device) -> float:
        """
        Zero out the most-activated pixels progressively; track confidence drop.
        Raw AUC (lower = better) is inverted so higher = better, like all metrics.
        """
        model.eval()
        # Resize CAM to the tensor's own spatial dims (may differ from display image)
        _, _, th, tw = input_tensor.shape
        cam          = _resize_to(cam, th, tw)
        n_px         = th * tw
        sorted_idx   = np.argsort(cam.flatten())[::-1].copy()  # .copy() removes negative stride — required for torch indexing
        x          = input_tensor.clone().to(device)
        baseline   = torch.zeros_like(x)
        fracs      = np.linspace(0, 1, steps + 1)
        confs      = []

        with torch.no_grad():
            for frac in fracs:
                n = int(frac * n_px)
                m = x.clone().reshape(1, -1, n_px)
                if n:
                    m[:, :, sorted_idx[:n]] = baseline.reshape(1, -1, n_px)[:, :, sorted_idx[:n]]
                logits = model(m.reshape_as(x))
                prob   = F.softmax(logits, dim=1)
                confs.append(prob[0, logits.argmax(1)].item())

        return round(1.0 - float(np.trapz(confs, fracs)), 6)   # inverted

    @staticmethod
    def insertion_auc(model, input_tensor, cam, steps, device) -> float:
        """
        Reveal most-activated pixels from a blurred baseline; track confidence rise.
        Higher AUC = better faithfulness.
        """
        model.eval()
        # Resize CAM to the tensor's own spatial dims (may differ from display image)
        _, _, th, tw = input_tensor.shape
        cam          = _resize_to(cam, th, tw)
        n_px         = th * tw
        sorted_idx   = np.argsort(cam.flatten())[::-1].copy()  # .copy() removes negative stride — required for torch indexing
        x          = input_tensor.clone().to(device)
        blurred    = cv2.GaussianBlur(x[0].cpu().permute(1,2,0).numpy(), (51,51), 0)
        baseline   = torch.from_numpy(blurred).permute(2,0,1).unsqueeze(0).to(device)
        fracs      = np.linspace(0, 1, steps + 1)
        confs      = []

        with torch.no_grad():
            for frac in fracs:
                n = int(frac * n_px)
                r = baseline.clone().reshape(1, -1, n_px)
                if n:
                    r[:, :, sorted_idx[:n]] = x.reshape(1, -1, n_px)[:, :, sorted_idx[:n]]
                logits = model(r.reshape_as(x))
                prob   = F.softmax(logits, dim=1)
                confs.append(prob[0, logits.argmax(1)].item())

        return round(float(np.trapz(confs, fracs)), 6)

    # ── Sharpness ────────────────────────────────────────────────────────

    @staticmethod
    def sparsity(cam) -> float:
        """Normalised entropy: low entropy = focused = high sparsity score."""
        p = cam.flatten() + 1e-9
        p /= p.sum()
        return round(float(1.0 - (-np.sum(p * np.log(p))) / np.log(len(p))), 6)

    @staticmethod
    def concentration(cam, topk_frac=0.2) -> float:
        """Fraction of total activation in the top-k% pixels."""
        flat = cam.flatten()
        k    = max(1, int(topk_frac * len(flat)))
        return round(float(np.sort(flat)[::-1][:k].sum() / (flat.sum() + 1e-9)), 6)

    # ── Stability ────────────────────────────────────────────────────────

    @staticmethod
    def stability(explainer_obj, extra_kwargs, input_tensor, cam_ref, runs, noise_std) -> float:
        """
        Re-run the explainer on `runs` noisy inputs; measure mean pixel deviation
        from the reference heatmap. Returns (1 - mean_deviation), higher = more stable.

        Memory note
        -----------
        Gradient-heavy explainers (e.g. Integrated Gradients) can accumulate
        significant GPU/CPU memory across runs. We therefore:
          - cap IG steps at 20 for stability reruns via the 'steps' extra_kwarg
            override (full accuracy is not needed here, just directional signal)
          - explicitly free the result dict and flush the CUDA cache after each run
        """
        h, w = cam_ref.shape
        devs = []

        # For IG explainers: override steps to a lightweight value for reruns.
        # This prevents the 10 x N_steps forward+backward passes from consuming
        # all available memory. Users can override by setting 'steps' in extra_kwargs.
        is_ig = hasattr(explainer_obj, "_compute_attributions")  # IG-specific method
        stability_kwargs = dict(extra_kwargs)
        if is_ig and "steps" not in stability_kwargs:
            stability_kwargs["steps"] = 20   # fast enough for deviation measurement

        for _ in range(runs):
            noisy = (input_tensor + torch.randn_like(input_tensor) * noise_std).clamp(0, 1)
            result = None
            try:
                result = explainer_obj.explain(noisy, **stability_kwargs)
                # Accept 'heatmap' (canonical) or legacy 'cam' key
                raw = result.get("heatmap") if "heatmap" in result else result.get("cam")
                if raw is None:
                    raise KeyError(
                        f"explain() result has no 'heatmap' or 'cam' key. "
                        f"Found: {list(result.keys())}"
                    )
                cam_noisy = _normalize(_resize_to(_to_numpy_hw(raw), h, w))
                devs.append(float(np.mean(np.abs(cam_noisy - cam_ref))))
            except Exception as exc:
                warnings.warn(f"Stability run failed: {exc}")
            finally:
                # Explicitly release result dict and flush CUDA cache every run
                # to prevent gradient-heavy explainers from accumulating memory.
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return round(float(1.0 - np.mean(devs)) if devs else 0.0, 6)

    # ── Localization ──────────────────────────────────────────────────────

    @staticmethod
    def iou(cam, gt_mask, threshold=0.5) -> float:
        pred = (cam >= threshold).astype(np.uint8)
        gt   = (gt_mask > 0).astype(np.uint8)
        return round(float((pred & gt).sum() / ((pred | gt).sum() + 1e-9)), 6)

    @staticmethod
    def pointing_game(cam, gt_mask) -> float:
        """1.0 if the peak activation pixel is inside the GT mask."""
        return float(gt_mask[np.unravel_index(cam.argmax(), cam.shape)] > 0)


# ---------------------------------------------------------------------------
# HeatmapComparator
# ---------------------------------------------------------------------------

class HeatmapComparator:
    """
    Compare XAI heatmap methods on Faithfulness, Sharpness, Stability,
    and optionally Localization.

    Parameters
    ----------
    model : torch.nn.Module
        The model being explained (used for faithfulness metrics).
    device : str
        'cpu' or 'cuda'. Default 'cpu'.
    deletion_steps : int
        AUC curve resolution. Default 10. Raise to 20 for smoother curves.
    faithfulness_enabled : bool
        Set False to skip Deletion/Insertion AUC (faster for large models).
    stability_enabled : bool
        Set False to skip stability checks entirely for all methods.
        Recommended when using explainers that are themselves expensive
        (e.g. LIME with num_samples=1000), since stability reruns multiply
        that cost by stability_runs. You can also skip stability per-method
        by passing None as the explainer object in the entry tuple.
        Default True.
    save_dir : str
        Directory for saved plots. Default 'user_saves/comparator_saves'.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        deletion_steps: int = 10,
        faithfulness_enabled: bool = True,
        stability_enabled: bool = True,
        save_dir: str = "user_saves/comparator_saves",
    ):
        self.model                = model
        self.model.eval()
        self.device               = device
        self.deletion_steps       = deletion_steps
        self.faithfulness_enabled = faithfulness_enabled
        self.stability_enabled    = stability_enabled
        self.save_dir             = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._m                   = _Metrics()

    # ──────────────────────────────────────────────────────────────────────
    # compare()
    # ──────────────────────────────────────────────────────────────────────

    def compare(
        self,
        entries: dict[str, tuple],
        input_tensor: torch.Tensor,
        input_image: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        stability_runs: int = 10,
        noise_std: float = 0.05,
        iou_threshold: float = 0.5,
    ) -> dict:
        """
        Run all metrics and return a structured results dict.

        Parameters
        ----------
        entries : dict[str, tuple]
            {display_name: (result_dict, explainer_object, extra_kwargs)}

            result_dict      – output of explainer.explain(); must have 'heatmap'.
            explainer_object – the explainer instance used to get that result;
                               must have .explain(input_tensor, **extra_kwargs).
                               Pass None to skip stability for this method.
            extra_kwargs     – kwargs forwarded to .explain() during stability runs.
                               GradCAM  : {"method": "gradcam"}
                               ViTGradCAM / DFF : {}

        input_tensor : torch.Tensor   shape (1, C, H, W)
        input_image  : np.ndarray, optional   (H, W, 3) float32 in [0,1]
        gt_mask      : np.ndarray, optional   (H, W) binary — enables IoU / Pointing Game
        stability_runs : int    noisy re-runs per method. Default 10.
        noise_std      : float  Gaussian noise std. Default 0.05.
        iou_threshold  : float  CAM binarisation threshold. Default 0.5.

        Returns
        -------
        dict
            scores       : dict[name -> dict[metric -> float]]
            summary      : dict[name -> float]   weighted composite
            ranked       : list[(name, score)]   best → worst
            winner       : str
            cams         : dict[name -> np.ndarray]  normalised (H,W)
            input_image  : np.ndarray
            gt_mask      : np.ndarray or None
            metrics_used : list[str]
            weights      : dict[metric -> float]
        """
        if not entries:
            raise ValueError("entries is empty.")

        # ── unpack ───────────────────────────────────────────────────────
        results_map:   dict[str, dict] = {}
        explainer_map: dict[str, Any]  = {}
        kwargs_map:    dict[str, dict] = {}

        for name, entry in entries.items():
            if len(entry) != 3:
                raise ValueError(
                    f"Entry '{name}' must be a 3-tuple "
                    f"(result_dict, explainer_object, extra_kwargs). Got {len(entry)} elements."
                )
            result, explainer, extra = entry

            # 'heatmap' is the canonical EXACT key for all explainers.
            # 'cam' is accepted as a backward-compatible alias (older GradCAM usage).
            # Everything is normalised to 'heatmap' internally from here on.
            if "heatmap" not in result and "cam" in result:
                result = {**result, "heatmap": result["cam"]}
            elif "heatmap" not in result:
                raise KeyError(
                    f"result_dict for '{name}' has no 'heatmap' key. "
                    f"All explainer result dicts must return a 'heatmap' key "
                    f"containing a 2-D (H, W) float32 numpy array. "
                    f"The legacy 'cam' key is also accepted for backward compatibility. "
                    f"Found keys: {list(result.keys())}"
                )
            results_map[name]   = result
            explainer_map[name] = explainer     # may be None
            kwargs_map[name]    = extra or {}

        # ── input image ──────────────────────────────────────────────────
        # Always pass through _to_display_image so that user-supplied arrays
        # (uint8 [0,255], unnormalised float, normalised tensors) are all
        # safely converted to float32 [0,1] before any blending or display.
        if input_image is None:
            input_image = _tensor_to_image(input_tensor)
        else:
            input_image = _to_display_image(input_image)
        h, w = input_image.shape[:2]

        # ── normalise CAMs to display-image space ─────────────────────
        # All CAMs are resized to (H_img, W_img) and min-max normalised here.
        # This single normalised representation is used by:
        #   - visual overlays  (heatmap grid)
        #   - sharpness        (sparsity, concentration)
        #   - stability        (pixel-wise deviation between heatmaps)
        #   - localization     (IoU, pointing game -- gt_mask is also in image space)
        #
        # IMPORTANT: faithfulness metrics (deletion_auc, insertion_auc) do NOT
        # use this dict directly. They operate on input_tensor and resize the CAM
        # internally to (H_tensor, W_tensor) -- the tensor's own spatial dims.
        # These two spaces can differ (e.g. 550x550 display image vs 224x224 tensor),
        # which is the root cause of the shape mismatch bug this design prevents.
        cams: dict[str, np.ndarray] = {
            name: _normalize(_resize_to(_to_numpy_hw(r["heatmap"]), h, w))
            for name, r in results_map.items()
        }

        # ── gt_mask ──────────────────────────────────────────────────────
        if gt_mask is not None:
            gt_mask = _resize_to((gt_mask > 0).astype(np.float32), h, w)

        scores: dict[str, dict[str, float]] = {name: {} for name in cams}
        metrics_used: list[str] = []

        # ════════════════════════════════════════════════════════════════
        # 1. FAITHFULNESS
        # ════════════════════════════════════════════════════════════════
        if self.faithfulness_enabled:
            print("  [1/4] Computing faithfulness (Deletion & Insertion AUC)...")
            x = input_tensor.to(self.device)
            for name, cam in cams.items():
                scores[name]["deletion_auc"]  = self._m.deletion_auc(
                    self.model, x, cam, self.deletion_steps, self.device)
                scores[name]["insertion_auc"] = self._m.insertion_auc(
                    self.model, x, cam, self.deletion_steps, self.device)
            metrics_used += ["deletion_auc", "insertion_auc"]
        else:
            print("  [1/4] Faithfulness skipped (faithfulness_enabled=False).")

        # ════════════════════════════════════════════════════════════════
        # 2. SHARPNESS
        # ════════════════════════════════════════════════════════════════
        print("  [2/4] Computing sharpness (Sparsity + Concentration)...")
        for name, cam in cams.items():
            scores[name]["sparsity"]      = self._m.sparsity(cam)
            scores[name]["concentration"] = self._m.concentration(cam)
        metrics_used += ["sparsity", "concentration"]

        # ════════════════════════════════════════════════════════════════
        # 3. STABILITY
        # ════════════════════════════════════════════════════════════════
        if not self.stability_enabled:
            print(
                "  [3/4] Stability skipped (stability_enabled=False)."
                "         Tip: disable per-method by passing None as the explainer"
                "         object, e.g. (result, None, {}) to skip just that method."
            )
        else:
            has_explainers = any(v is not None for v in explainer_map.values())
            if has_explainers:
                print(f"  [3/4] Computing stability ({stability_runs} noisy runs per method)...")
                for name, cam in cams.items():
                    exp = explainer_map[name]
                    if exp is None:
                        print(f"         Stability skipped for '{name}' (explainer=None).")
                        continue
                    scores[name]["stability"] = self._m.stability(
                        explainer_obj=exp,
                        extra_kwargs=kwargs_map[name],
                        input_tensor=input_tensor,
                        cam_ref=cam,
                        runs=stability_runs,
                        noise_std=noise_std,
                    )
                if any("stability" in scores[n] for n in cams):
                    metrics_used.append("stability")
            else:
                print(
                    "  [3/4] Stability skipped."
                    "         To enable: pass the explainer instance as the 2nd element"
                    "         of each entry tuple, e.g. (result, gradcam_exp, {'method':'gradcam'})."
                )

        # ════════════════════════════════════════════════════════════════
        # 4. LOCALIZATION
        # ════════════════════════════════════════════════════════════════
        if gt_mask is not None:
            print("  [4/4] Computing localization (IoU + Pointing Game)...")
            for name, cam in cams.items():
                scores[name]["iou"]           = self._m.iou(cam, gt_mask, iou_threshold)
                scores[name]["pointing_game"] = self._m.pointing_game(cam, gt_mask)
            metrics_used += ["iou", "pointing_game"]
        else:
            print("  [4/4] Localization skipped (no gt_mask provided).")

        # ════════════════════════════════════════════════════════════════
        # COMPOSITE SCORE
        # ════════════════════════════════════════════════════════════════
        weights = self._default_weights(metrics_used)
        summary = {
            name: round(sum(
                weights.get(m, 0.0) * scores[name].get(m, 0.0)
                for m in metrics_used
            ), 4)
            for name in cams
        }
        ranked = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        winner = ranked[0][0] if ranked else None

        return {
            "scores":       scores,
            "summary":      summary,
            "ranked":       ranked,
            "winner":       winner,
            "cams":         cams,
            "input_image":  input_image,
            "gt_mask":      gt_mask,
            "metrics_used": metrics_used,
            "weights":      weights,
        }

    # ──────────────────────────────────────────────────────────────────────
    # report()
    # ──────────────────────────────────────────────────────────────────────

    def report(self, results: dict, decimals: int = 4) -> None:
        """
        Print a formatted score table and ranked summary to the console.

        Parameters
        ----------
        results  : dict – output of compare().
        decimals : int  – decimal places. Default 4.
        """
        scores  = results["scores"]
        ranked  = results["ranked"]
        winner  = results["winner"]
        metrics = results["metrics_used"]
        weights = results["weights"]
        methods = list(scores.keys())

        col_w = max(14, max(len(m) for m in methods) + 2)
        met_w = max(20, max(len(m) for m in metrics) + 2)
        width = met_w + col_w * len(methods) + 4

        print("\n" + "═" * width)
        print("  EXACT — Heatmap Comparison Report")
        print("═" * width)
        print(f"{'Metric':<{met_w}}" + "".join(f"{m:>{col_w}}" for m in methods))
        print("─" * width)

        for met in metrics:
            row_vals = [scores[m].get(met, float("nan")) for m in methods]
            valid    = [v for v in row_vals if not np.isnan(v)]
            best     = max(valid) if valid else None
            row      = f"{'  ' + met:<{met_w}}"
            for v in row_vals:
                if np.isnan(v):
                    row += f"{'N/A':>{col_w}}"
                else:
                    cell = f"{v:.{decimals}f}"
                    star = " ★" if (best is not None and abs(v - best) < 1e-9) else "  "
                    row += f"{(cell + star):>{col_w}}"
            print(row)

        print("─" * width)
        comp_row = f"{'  COMPOSITE':<{met_w}}"
        for m in methods:
            comp_row += f"{results['summary'][m]:>{col_w}.{decimals}f}"
        print(comp_row)
        print("═" * width)

        print("\n  RANKED SUMMARY  (all metrics, higher = better)")
        print("  " + "─" * 44)
        medals = ["[1]", "[2]", "[3]"] + ["   "] * 20
        for i, (name, score) in enumerate(ranked):
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"  {medals[i]}  #{i+1}  {name:<24} {score:.4f}  {bar}")

        print(f"\n  [WINNER] : {winner}")
        print(f"  Weights   : { {k: round(v, 2) for k, v in weights.items()} }\n")

    # ──────────────────────────────────────────────────────────────────────
    # plot()
    # ──────────────────────────────────────────────────────────────────────

    def plot(
        self,
        results: dict,
        save_png: bool = False,
        filename: str = "comparison.png",
        figsize_scale: float = 1.0,
    ) -> None:
        """
        Render a 4-panel visual comparison report and optionally save it.

          [1] Side-by-side heatmap overlays
          [2] Colour-coded numeric score table
          [3] Horizontal bar chart of composite scores
          [4] Radar chart of per-metric profiles

        Parameters
        ----------
        results       : dict   – output of compare().
        save_png      : bool   – save to save_dir/filename. Default False.
        filename      : str    – saved filename. Default 'comparison.png'.
        figsize_scale : float  – figure scale factor. Default 1.0.
        """
        try:
            import matplotlib
            if save_png:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("pip install matplotlib")

        methods   = list(results["cams"].keys())
        n_m       = len(methods)
        cams      = results["cams"]
        img       = results["input_image"]
        metrics   = results["metrics_used"]
        scores    = results["scores"]
        ranked    = results["ranked"]
        winner    = results["winner"]
        s         = figsize_scale

        PALETTE       = ["#4E9AF1","#F4845F","#54C27D","#A78BFA","#F7C948","#E879A0"]
        method_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
        BG, CARD      = "#0D1117", "#161B22"
        TEXT, SUB     = "#E6EDF3", "#8B949E"
        GOLD          = "#F7C948"

        fig = plt.figure(figsize=(max(18, 5 + 3 * n_m) * s, 22 * s), facecolor=BG)
        fig.suptitle("EXACT -- Heatmap Comparison Report",
                     color=TEXT, fontsize=18*s, fontweight="bold",
                     fontfamily="monospace", y=0.98)

        outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                                  top=0.95, bottom=0.04, left=0.05, right=0.97,
                                  height_ratios=[2.5, 2.0, 1.5, 1.5])

        # [1] heatmap grid ──────────────────────────────────────────────────
        gs1 = gridspec.GridSpecFromSubplotSpec(1, n_m+1, subplot_spec=outer[0], wspace=0.06)
        ax  = fig.add_subplot(gs1[0])
        ax.imshow(img); ax.set_title("Original", color=TEXT, fontsize=9*s, fontweight="bold", pad=6); ax.axis("off")
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs1[i+1])
            ax.imshow(_overlay(img, cams[method]))
            ax.set_title(method + ("  [W]" if method == winner else ""),
                         color=method_colors[method], fontsize=9*s, fontweight="bold", pad=6)
            ax.axis("off")
        _panel_label(fig, outer[0], "[1] Heatmap Overlay", SUB, s)

        # [2] score table ───────────────────────────────────────────────────
        ax_tbl = fig.add_subplot(outer[1])
        ax_tbl.set_facecolor(CARD); ax_tbl.axis("off")
        cell_texts, cell_colors = [], []
        for met in metrics + ["COMPOSITE"]:
            is_c    = met == "COMPOSITE"
            vals    = ([results["summary"][m] for m in methods] if is_c
                       else [scores[m].get(met, float("nan")) for m in methods])
            valid   = [v for v in vals if not np.isnan(v)]
            best_v  = max(valid) if valid else None
            worst_v = min(valid) if valid else None
            base    = "#1C1F26" if is_c else CARD
            row_t   = [f"  {met}"]
            row_c   = [base]
            for v in vals:
                if np.isnan(v):
                    row_t.append("N/A"); row_c.append(base)
                else:
                    row_t.append(f"{v:.4f}")
                    if best_v is not None and abs(v - best_v) < 1e-9:
                        row_c.append("#1A3A1A")
                    elif worst_v is not None and abs(v - worst_v) < 1e-9 and best_v != worst_v:
                        row_c.append("#3A1A1A")
                    else:
                        row_c.append(base)
            cell_texts.append(row_t); cell_colors.append(row_c)
        tbl = ax_tbl.table(cellText=cell_texts, colLabels=["Metric"]+methods,
                           cellColours=cell_colors, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9*s); tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#30363D")
            cell.set_text_props(color=TEXT if r > 0 else GOLD,
                                fontweight="bold" if r == 0 else "normal",
                                fontfamily="monospace")
        _panel_label(fig, outer[1], "[2] Per-metric Score Table  (green = best, red = worst per row)", SUB, s)

        # [3] bar chart ─────────────────────────────────────────────────────
        ax_bar = fig.add_subplot(outer[2])
        ax_bar.set_facecolor(CARD)
        ax_bar.spines[["top","right","left","bottom"]].set_color("#30363D")
        sm = [r[0] for r in ranked]; sv = [r[1] for r in ranked]
        bc = [GOLD if m == winner else method_colors[m] for m in sm]
        bars = ax_bar.barh(sm[::-1], sv[::-1], color=bc[::-1], height=0.55, edgecolor="#30363D")
        for bar, val in zip(bars, sv[::-1]):
            ax_bar.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                        f"{val:.4f}", va="center", ha="left",
                        color=TEXT, fontsize=8*s, fontfamily="monospace")
        ax_bar.set_xlim(0, max(sv)*1.22+0.01)
        ax_bar.set_xlabel("Composite Score  (higher = better)", color=SUB, fontsize=8*s)
        ax_bar.tick_params(axis="y", labelcolor=TEXT, labelsize=9*s)
        ax_bar.tick_params(axis="x", labelcolor=SUB, labelsize=8*s)
        ax_bar.set_title(f"[WINNER]  {winner}", color=GOLD, fontsize=10*s, fontweight="bold", pad=8)
        _panel_label(fig, outer[2], "[3] Composite Score Ranking", SUB, s)

        # [4] radar chart ───────────────────────────────────────────────────
        ax_host = fig.add_subplot(outer[3]); ax_host.axis("off")
        if len(metrics) >= 3:
            ax_r = fig.add_subplot(
                gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer[3])[0], polar=True)
            _draw_radar(ax_r, methods, metrics, scores, method_colors, TEXT, SUB, CARD, s)
            _panel_label(fig, outer[3], "[4] Radar Chart  (per-metric profile)", SUB, s)
        else:
            ax_host.text(0.5, 0.5, "Radar chart requires ≥ 3 metrics",
                         ha="center", va="center", color=SUB,
                         fontsize=10*s, transform=ax_host.transAxes)

        plt.tight_layout()
        if save_png:
            out = self.save_dir / filename
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"✓ Saved: {out}")
        else:
            plt.show()
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_weights(metrics: list[str]) -> dict[str, float]:
        """
        Pre-set weights renormalised over active metrics.
        Faithfulness > Stability > Sharpness > Localization.
        """
        base = {
            "deletion_auc":  0.20,
            "insertion_auc": 0.20,
            "stability":     0.15,
            "concentration": 0.13,
            "sparsity":      0.12,
            "iou":           0.12,
            "pointing_game": 0.08,
        }
        active = {m: base[m] for m in metrics if m in base}
        total  = sum(active.values())
        if not total:
            return {m: 1.0 / len(metrics) for m in metrics}
        return {m: v / total for m, v in active.items()}


# ---------------------------------------------------------------------------
# Plot helpers (module-level)
# ---------------------------------------------------------------------------

def _overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a jet-coloured heatmap over a display image.
    Defensively normalises img to [0,1] in case it arrives unnormalised.
    """
    import matplotlib.cm as cm
    img = _to_display_image(img)   # guard: ensures [0,1] float32 regardless of input
    heatmap = cm.jet(cam)[..., :3].astype(np.float32)
    return np.clip(0.55 * heatmap + 0.45 * img, 0, 1)


def _panel_label(fig, subplot_spec, text, color, scale):
    bbox = subplot_spec.get_position(fig)
    fig.text(bbox.x0, bbox.y1 + 0.005 * scale, text,
             color=color, fontsize=8*scale, fontfamily="monospace",
             transform=fig.transFigure)


def _draw_radar(ax, methods, metrics, scores, colors, text_c, sub_c, bg, s):
    N      = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_facecolor(bg)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color=text_c, fontsize=8*s, fontfamily="monospace")
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.grid(color="#30363D", linewidth=0.8)
    ax.spines["polar"].set_color("#30363D")
    for method in methods:
        vals = [scores[method].get(m, 0.0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[method], linewidth=1.8, label=method)
        ax.fill(angles, vals, color=colors[method], alpha=0.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              labelcolor=text_c, facecolor=bg, edgecolor="#30363D", fontsize=8*s)