# comparators/heatmap_comp.py

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

    cmp = HeatmapComparator(model, device="cuda")

    results = cmp.compare(
        explainer_results={                 # dict[name -> explain() output]
            "GradCAM":    gradcam_result,
            "GradCAM++":  gradcampp_result,
            "EigenCAM":   eigencam_result,
        },
        input_tensor=input_tensor,          # (1, C, H, W)
        input_image=input_image,            # (H, W, 3) float32 in [0,1]
        gt_mask=gt_mask,                    # (H, W) binary, optional
        stability_runs=10,                  # how many noisy passes for stability
        noise_std=0.05,
    )

    cmp.report(results)                     # prints table + ranked summary
    cmp.plot(results, file_name)          # saves complete report to disk
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_hw(cam) -> np.ndarray:
    """Accept (H,W) ndarray or (1,H,W) / (H,W,1) and return (H,W) float32."""
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


# ---------------------------------------------------------------------------
# Metric functions (all operate on normalised float32 (H,W) arrays)
# ---------------------------------------------------------------------------

class _Metrics:
    """Static metric implementations."""

    # -- Faithfulness --------------------------------------------------------

    @staticmethod
    def deletion_auc(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        cam: np.ndarray,
        steps: int = 10,
        device: str = "cpu",
    ) -> float:
        """
        Deletion AUC: progressively mask the *most important* pixels (highest cam)
        and measure how fast the model's confidence drops.
        Lower AUC → heatmap highlights truly important regions → BETTER faithfulness.

        Returns AUC in [0, 1] (lower is better, inverted to higher-is-better below).
        """
        model.eval()
        h, w = cam.shape
        n_pixels = h * w
        flat_cam = cam.flatten()
        sorted_idx = np.argsort(flat_cam)[::-1]  # most important first

        x = input_tensor.clone().to(device)       # (1, C, H, W)
        baseline = torch.zeros_like(x)

        scores = []
        step_fracs = np.linspace(0, 1, steps + 1)

        with torch.no_grad():
            for frac in step_fracs:
                n_mask = int(frac * n_pixels)
                masked = x.clone().reshape(1, -1, n_pixels)
                if n_mask > 0:
                    masked[:, :, sorted_idx[:n_mask]] = baseline.reshape(1, -1, n_pixels)[:, :, sorted_idx[:n_mask]]
                masked = masked.reshape_as(x)
                logits = model(masked)
                prob = F.softmax(logits, dim=1)
                # score for originally predicted class
                pred_class = logits.argmax(dim=1)
                scores.append(prob[0, pred_class].item())

        return float(np.trapz(scores, step_fracs))   # AUC; lower = better faithfulness

    @staticmethod
    def insertion_auc(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        cam: np.ndarray,
        steps: int = 10,
        device: str = "cpu",
    ) -> float:
        """
        Insertion AUC: start from a blurred baseline and progressively *reveal*
        the most important pixels.
        Higher AUC → heatmap correctly identifies what the model needs → BETTER.
        """
        model.eval()
        h, w = cam.shape
        n_pixels = h * w
        flat_cam = cam.flatten()
        sorted_idx = np.argsort(flat_cam)[::-1]

        x = input_tensor.clone().to(device)
        # Blurred baseline (blurs the image so model can't see it)
        x_np = x[0].cpu().permute(1, 2, 0).numpy()
        blurred_np = cv2.GaussianBlur(x_np, (51, 51), 0)
        baseline = torch.from_numpy(blurred_np).permute(2, 0, 1).unsqueeze(0).to(device)

        scores = []
        step_fracs = np.linspace(0, 1, steps + 1)

        with torch.no_grad():
            for frac in step_fracs:
                n_reveal = int(frac * n_pixels)
                revealed = baseline.clone().reshape(1, -1, n_pixels)
                if n_reveal > 0:
                    revealed[:, :, sorted_idx[:n_reveal]] = x.reshape(1, -1, n_pixels)[:, :, sorted_idx[:n_reveal]]
                revealed = revealed.reshape_as(x)
                logits = model(revealed)
                prob = F.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1)
                scores.append(prob[0, pred_class].item())

        return float(np.trapz(scores, step_fracs))   # higher = better

    # -- Sharpness -----------------------------------------------------------

    @staticmethod
    def sparsity(cam: np.ndarray) -> float:
        """
        Sparsity via normalised entropy.  
        A focused heatmap has low entropy → high sparsity score.
        Returns value in [0, 1]; HIGHER = sharper / more focused.
        """
        p = cam.flatten() + 1e-9
        p = p / p.sum()
        n = len(p)
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(n)
        return float(1.0 - entropy / max_entropy)

    @staticmethod
    def concentration(cam: np.ndarray, topk_frac: float = 0.2) -> float:
        """
        Fraction of total activation energy concentrated in the top-k% pixels.
        HIGHER = more concentrated / sharper.
        """
        flat = cam.flatten()
        k = max(1, int(topk_frac * len(flat)))
        top_k_sum = np.sort(flat)[::-1][:k].sum()
        total = flat.sum() + 1e-9
        return float(top_k_sum / total)

    # -- Stability -----------------------------------------------------------

    @staticmethod
    def stability(
        explainer_fn,
        input_tensor: torch.Tensor,
        cam_ref: np.ndarray,
        runs: int = 10,
        noise_std: float = 0.05,
    ) -> float:
        """
        Run the explainer `runs` times on slightly noisy copies of the input.
        Compute mean pixel-wise absolute deviation from the reference heatmap.
        LOWER deviation = more stable. We return (1 - mean_deviation) so that
        HIGHER = more stable (consistent with other metrics).
        """
        deviations = []
        h, w = cam_ref.shape

        for _ in range(runs):
            noise = torch.randn_like(input_tensor) * noise_std
            noisy = (input_tensor + noise).clamp(0, 1)
            try:
                result = explainer_fn(noisy)
                cam_noisy = _normalize(_resize_to(_to_numpy_hw(result), h, w))
                deviations.append(np.mean(np.abs(cam_noisy - cam_ref)))
            except Exception:
                pass

        if not deviations:
            return 0.0
        return float(1.0 - np.mean(deviations))

    # -- Localization (needs GT mask) ----------------------------------------

    @staticmethod
    def iou(cam: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """
        Binarise the CAM at `threshold` and compute IoU against the GT mask.
        HIGHER = better localization.
        """
        pred = (cam >= threshold).astype(np.uint8)
        gt   = (gt_mask > 0).astype(np.uint8)
        intersection = (pred & gt).sum()
        union        = (pred | gt).sum()
        return float(intersection / (union + 1e-9))

    @staticmethod
    def pointing_game(cam: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        1.0 if the pixel with the highest activation falls inside the GT mask,
        0.0 otherwise. A single-sample metric; averaged across a dataset it 
        becomes the standard "Pointing Game" accuracy.
        """
        peak_idx = np.unravel_index(cam.argmax(), cam.shape)
        return float(gt_mask[peak_idx] > 0)


# ---------------------------------------------------------------------------
# Main comparator class
# ---------------------------------------------------------------------------

class HeatmapComparator:
    """
    Compare XAI heatmap methods on Faithfulness, Sharpness, Stability,
    and optionally Localization (when a GT mask is supplied).

    Parameters
    ----------
    model : torch.nn.Module
        The model being explained. Used for faithfulness and stability metrics.
    device : str, optional
        'cpu' or 'cuda'. Default is 'cpu'.
    deletion_steps : int, optional
        Number of masking steps for Deletion/Insertion AUC. Default 10.
        Increase (e.g. 20) for smoother curves, at higher compute cost.
    faithfulness_enabled : bool, optional
        Set to False to skip Deletion/Insertion (saves time for large models).
    save_dir : str, optional
        Where to save output plots. Default 'user_saves/comparator_saves'.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        deletion_steps: int = 10,
        faithfulness_enabled: bool = True,
        save_dir: str = "user_saves/comparator_saves",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.deletion_steps = deletion_steps
        self.faithfulness_enabled = faithfulness_enabled
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._m = _Metrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        explainer_results: dict,
        input_tensor: torch.Tensor,
        input_image: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        stability_fns: Optional[dict] = None,
        stability_runs: int = 10,
        noise_std: float = 0.05,
        iou_threshold: float = 0.5,
    ) -> dict:
        """
        Run all metrics for each method and return a structured results dict.

        Parameters
        ----------
        explainer_results : dict[str, dict]
            Output of any explainer's explain() method, keyed by a name you choose.
            Each value must have a 'cam' key containing an (H, W) numpy array.
            Example::

                {
                    "GradCAM":  gradcam_explainer.explain(input_tensor, ...),
                    "EigenCAM": eigencam_explainer.explain(input_tensor, ...),
                }

        input_tensor : torch.Tensor
            Shape (1, C, H, W). The input used to generate the heatmaps.

        input_image : np.ndarray, optional
            Shape (H, W, 3), float32 in [0, 1]. Used for visual grid.
            If None, derived from input_tensor.

        gt_mask : np.ndarray, optional
            Shape (H, W), binary (0/1 or bool). Ground-truth foreground mask
            or bounding-box mask. If None, localization metrics are skipped.

        stability_fns : dict[str, callable], optional
            Dict mapping method names to callables that accept a noisy
            input_tensor and return a result dict with a 'cam' key.
            If None, stability is skipped.
            Example::

                stability_fns={
                    "GradCAM":  lambda t: gradcam_exp.explain(t),
                    "EigenCAM": lambda t: eigencam_exp.explain(t),
                }

        stability_runs : int
            Number of noisy runs per method. Default 10.

        noise_std : float
            Std of Gaussian noise added for stability. Default 0.05.

        iou_threshold : float
            Binarisation threshold for IoU. Default 0.5.

        Returns
        -------
        dict
            {
              'scores':  dict[method_name -> dict[metric_name -> float]],
              'summary': dict[method_name -> float],   # weighted composite score
              'ranked':  list[tuple[method_name, composite_score]],
              'winner':  str,
              'cams':    dict[method_name -> np.ndarray],  # normalised (H,W)
              'input_image': np.ndarray,
              'gt_mask': np.ndarray or None,
              'metrics_used': list[str],
            }
        """
        if not explainer_results:
            raise ValueError("explainer_results is empty.")

        # --- Infer image ---
        if input_image is None:
            img = input_tensor[0].cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.max() > 1.0:
                img = img / 255.0
            input_image = np.float32(img)

        h, w = input_image.shape[:2]

        # --- Extract and normalise CAMs ---
        cams: dict[str, np.ndarray] = {}
        for name, result in explainer_results.items():
            if "cam" not in result:
                raise KeyError(f"Result for '{name}' has no 'cam' key. "
                               f"Keys found: {list(result.keys())}")
            cam_raw = _to_numpy_hw(result["cam"])
            cam_raw = _resize_to(cam_raw, h, w)
            cams[name] = _normalize(cam_raw)

        # --- Prepare gt_mask ---
        if gt_mask is not None:
            gt_mask = _resize_to(
                (gt_mask > 0).astype(np.float32), h, w
            )

        scores: dict[str, dict[str, float]] = {name: {} for name in cams}
        metrics_used: list[str] = []

        # === FAITHFULNESS ===
        if self.faithfulness_enabled:
            print("  [1/4] Computing faithfulness (Deletion & Insertion AUC)...")
            x = input_tensor.to(self.device)
            for name, cam in cams.items():
                del_auc = self._m.deletion_auc(
                    self.model, x, cam, self.deletion_steps, self.device
                )
                ins_auc = self._m.insertion_auc(
                    self.model, x, cam, self.deletion_steps, self.device
                )
                # Invert deletion AUC: lower deletion = better → flip to [0,1]
                scores[name]["deletion_auc"]  = round(1.0 - del_auc, 4)
                scores[name]["insertion_auc"] = round(ins_auc, 4)
            metrics_used += ["deletion_auc", "insertion_auc"]
        else:
            print("  [1/4] Faithfulness skipped (faithfulness_enabled=False).")

        # === SHARPNESS ===
        print("  [2/4] Computing sharpness (Sparsity + Concentration)...")
        for name, cam in cams.items():
            scores[name]["sparsity"]      = round(self._m.sparsity(cam), 4)
            scores[name]["concentration"] = round(self._m.concentration(cam), 4)
        metrics_used += ["sparsity", "concentration"]

        # === STABILITY ===
        if stability_fns is not None:
            print(f"  [3/4] Computing stability ({stability_runs} noisy runs each)...")
            for name, cam in cams.items():
                if name in stability_fns:
                    stab = self._m.stability(
                        stability_fns[name], input_tensor, cam,
                        stability_runs, noise_std
                    )
                    scores[name]["stability"] = round(stab, 4)
                else:
                    warnings.warn(f"No stability_fn for '{name}'; skipping stability.")
            if "stability" not in metrics_used:
                metrics_used.append("stability")
        else:
            print("  [3/4] Stability skipped (no stability_fns provided).")

        # === LOCALIZATION ===
        if gt_mask is not None:
            print("  [4/4] Computing localization (IoU + Pointing Game)...")
            for name, cam in cams.items():
                scores[name]["iou"]           = round(self._m.iou(cam, gt_mask, iou_threshold), 4)
                scores[name]["pointing_game"] = round(self._m.pointing_game(cam, gt_mask), 4)
            metrics_used += ["iou", "pointing_game"]
        else:
            print("  [4/4] Localization skipped (no gt_mask provided).")

        # === COMPOSITE SCORE ===
        weights = self._default_weights(metrics_used)
        summary: dict[str, float] = {}
        for name in cams:
            method_scores = scores[name]
            composite = sum(
                weights.get(m, 0.0) * method_scores.get(m, 0.0)
                for m in metrics_used
            )
            summary[name] = round(composite, 4)

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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, results: dict, decimals: int = 4) -> None:
        """
        Print a formatted table of per-metric scores and a ranked summary.

        Parameters
        ----------
        results : dict
            Output of compare().
        decimals : int
            Decimal places to show. Default 4.
        """
        scores  = results["scores"]
        ranked  = results["ranked"]
        winner  = results["winner"]
        metrics = results["metrics_used"]
        weights = results["weights"]

        methods = list(scores.keys())
        col_w   = max(14, max(len(m) for m in methods) + 2)
        met_w   = max(18, max(len(m) for m in metrics) + 2)

        # --- Header ---
        print("\n" + "═" * (met_w + col_w * len(methods) + 4))
        print("  EXACT — Heatmap Comparison Report")
        print("═" * (met_w + col_w * len(methods) + 4))

        header = f"{'Metric':<{met_w}}" + "".join(f"{m:>{col_w}}" for m in methods)
        print(header)
        print("─" * (met_w + col_w * len(methods) + 4))

        # --- Per-metric rows ---
        for met in metrics:
            row_vals = [scores[m].get(met, float("nan")) for m in methods]
            best_val = max(v for v in row_vals if not np.isnan(v))
            row = f"{'  ' + met:<{met_w}}"
            for v in row_vals:
                cell = f"{v:.{decimals}f}" if not np.isnan(v) else "  N/A  "
                mark = " ★" if abs(v - best_val) < 1e-9 else "  "
                row += f"{(cell + mark):>{col_w}}"
            print(row)

        print("─" * (met_w + col_w * len(methods) + 4))

        # --- Composite ---
        comp_row = f"{'  COMPOSITE':< {met_w}}"
        for m in methods:
            v = results["summary"][m]
            comp_row += f"{v:>{col_w}.{decimals}f}"
        print(comp_row)
        print("═" * (met_w + col_w * len(methods) + 4))

        # --- Ranked summary ---
        print("\n  📊 RANKED SUMMARY  (higher = better, all metrics ↑)")
        print("  " + "─" * 40)
        medals = ["🥇", "🥈", "🥉"] + ["  "] * 10
        for i, (name, score) in enumerate(ranked):
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {medals[i]} #{i+1}  {name:<20}  {score:.4f}  {bar}")

        print(f"\n  🏆 Winner: {winner}")
        print(f"  Weights used: { {k: round(v,2) for k,v in weights.items()} }")
        print()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        results: dict,
        save_png: bool = False,
        filename: str = "comparison.png",
        figsize_scale: float = 1.0,
    ) -> None:
        """
        Render four panels:
          1. Side-by-side heatmap grid (original + one overlay per method)
          2. Numeric score table (colour-coded)
          3. Bar chart of composite scores
          4. Radar chart of per-metric scores

        Parameters
        ----------
        results : dict
            Output of compare().
        save_png : bool
            Whether to save the figure. Default False.
        filename : str
            Filename to save under save_dir. Default 'comparison.png'.
        figsize_scale : float
            Scale factor for the figure. Default 1.0.
        """
        try:
            import matplotlib
            matplotlib.use("Agg") if save_png else None
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            raise ImportError("matplotlib is required for plot(). "
                              "Install with: pip install matplotlib")

        methods  = list(results["cams"].keys())
        n_methods = len(methods)
        cams     = results["cams"]
        img      = results["input_image"]
        metrics  = results["metrics_used"]
        scores   = results["scores"]
        ranked   = results["ranked"]
        winner   = results["winner"]

        # ── colour palette ───────────────────────────────────────────────
        PALETTE  = ["#4E9AF1","#F4845F","#54C27D","#A78BFA","#F7C948","#E879A0"]
        method_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}

        BG       = "#0D1117"
        CARD     = "#161B22"
        TEXT     = "#E6EDF3"
        SUBTEXT  = "#8B949E"
        GOLD     = "#F7C948"
        GREEN    = "#54C27D"
        RED      = "#F4845F"

        scale    = figsize_scale
        fig_w    = max(18, 5 + 3 * n_methods) * scale
        fig_h    = 22 * scale

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
        fig.suptitle(
            "EXACT  ·  Heatmap Comparison Report",
            color=TEXT, fontsize=18 * scale, fontweight="bold",
            fontfamily="monospace", y=0.98,
        )

        # ── layout ───────────────────────────────────────────────────────
        outer = gridspec.GridSpec(
            4, 1, figure=fig,
            hspace=0.45,
            top=0.95, bottom=0.04,
            left=0.05, right=0.97,
            height_ratios=[2.5, 2.0, 1.5, 1.5],
        )

        # ================================================================
        # Panel 1 – Side-by-side heatmap grid
        # ================================================================
        gs1  = gridspec.GridSpecFromSubplotSpec(
            1, n_methods + 1, subplot_spec=outer[0], wspace=0.06
        )
        ax_orig = fig.add_subplot(gs1[0])
        ax_orig.imshow(img)
        ax_orig.set_title("Original", color=TEXT, fontsize=9 * scale,
                          fontweight="bold", pad=6)
        ax_orig.axis("off")

        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs1[i + 1])
            overlay = _overlay(img, cams[method])
            ax.imshow(overlay)
            label = method + (" 🏆" if method == winner else "")
            ax.set_title(label, color=method_colors[method],
                         fontsize=9 * scale, fontweight="bold", pad=6)
            ax.axis("off")

        _panel_label(fig, outer[0], "① Heatmap Overlay", color=SUBTEXT, scale=scale)

        # ================================================================
        # Panel 2 – Score table
        # ================================================================
        ax_tbl = fig.add_subplot(outer[1])
        ax_tbl.set_facecolor(CARD)
        ax_tbl.axis("off")

        n_rows = len(metrics) + 1   # metrics + composite
        n_cols = n_methods + 1      # label col + one per method

        cell_texts = []
        cell_colors = []

        for met in metrics:
            row_vals = [scores[m].get(met, float("nan")) for m in methods]
            valid    = [v for v in row_vals if not np.isnan(v)]
            best     = max(valid) if valid else None
            worst    = min(valid) if valid else None

            row_text  = [f"  {met}"]
            row_color = [CARD]
            for v in row_vals:
                if np.isnan(v):
                    row_text.append("N/A")
                    row_color.append(CARD)
                else:
                    row_text.append(f"{v:.4f}")
                    if best is not None and abs(v - best) < 1e-9:
                        row_color.append("#1A3A1A")   # dark green tint
                    elif worst is not None and abs(v - worst) < 1e-9 and best != worst:
                        row_color.append("#3A1A1A")   # dark red tint
                    else:
                        row_color.append(CARD)

            cell_texts.append(row_text)
            cell_colors.append(row_color)

        # Composite row
        comp_vals  = [results["summary"][m] for m in methods]
        best_comp  = max(comp_vals)
        worst_comp = min(comp_vals)
        comp_row_text  = ["  COMPOSITE"]
        comp_row_color = ["#1C1F26"]
        for v in comp_vals:
            comp_row_text.append(f"{v:.4f}")
            if abs(v - best_comp) < 1e-9:
                comp_row_color.append("#2A4A1A")
            elif abs(v - worst_comp) < 1e-9 and best_comp != worst_comp:
                comp_row_color.append("#4A1A1A")
            else:
                comp_row_color.append("#1C1F26")
        cell_texts.append(comp_row_text)
        cell_colors.append(comp_row_color)

        col_labels = ["Metric"] + methods
        tbl = ax_tbl.table(
            cellText=cell_texts,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9 * scale)
        tbl.scale(1, 1.6)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#30363D")
            cell.set_text_props(color=TEXT if row > 0 else GOLD,
                                fontweight="bold" if row == 0 else "normal",
                                fontfamily="monospace")

        _panel_label(fig, outer[1], "② Per-metric Score Table  (★ = best per row)", color=SUBTEXT, scale=scale)

        # ================================================================
        # Panel 3 – Bar chart (composite)
        # ================================================================
        ax_bar = fig.add_subplot(outer[2])
        ax_bar.set_facecolor(CARD)
        ax_bar.spines[["top","right","left","bottom"]].set_color("#30363D")
        ax_bar.tick_params(colors=TEXT)

        sorted_methods = [r[0] for r in ranked]
        sorted_scores  = [r[1] for r in ranked]
        bar_colors = [
            GOLD if m == winner else method_colors[m] for m in sorted_methods
        ]

        bars = ax_bar.barh(
            sorted_methods[::-1], sorted_scores[::-1],
            color=bar_colors[::-1], height=0.55, edgecolor="#30363D",
        )
        for bar, val in zip(bars, sorted_scores[::-1]):
            ax_bar.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color=TEXT, fontsize=8 * scale, fontfamily="monospace",
            )

        ax_bar.set_xlim(0, max(sorted_scores) * 1.20 + 0.01)
        ax_bar.set_xlabel("Composite Score (higher = better)", color=SUBTEXT,
                          fontsize=8 * scale)
        ax_bar.set_facecolor(CARD)
        ax_bar.tick_params(axis="y", labelcolor=TEXT, labelsize=9 * scale)
        ax_bar.tick_params(axis="x", labelcolor=SUBTEXT, labelsize=8 * scale)
        ax_bar.set_title(f"🏆 Winner: {winner}", color=GOLD,
                         fontsize=10 * scale, fontweight="bold", pad=8)

        _panel_label(fig, outer[2], "③ Composite Score Ranking", color=SUBTEXT, scale=scale)

        # ================================================================
        # Panel 4 – Radar chart
        # ================================================================
        ax_radar_host = fig.add_subplot(outer[3])
        ax_radar_host.axis("off")

        if len(metrics) >= 3:
            radar_gs = gridspec.GridSpecFromSubplotSpec(
                1, 1, subplot_spec=outer[3]
            )
            ax_r = fig.add_subplot(radar_gs[0], polar=True)
            _draw_radar(ax_r, methods, metrics, scores, method_colors, TEXT, SUBTEXT, CARD, scale)
            _panel_label(fig, outer[3], "④ Radar Chart (per-metric profile)", color=SUBTEXT, scale=scale)
        else:
            ax_radar_host.text(
                0.5, 0.5, "Radar chart requires ≥ 3 metrics",
                ha="center", va="center", color=SUBTEXT,
                fontsize=10 * scale, transform=ax_radar_host.transAxes
            )

        # ── save / show ──────────────────────────────────────────────────
        plt.tight_layout()
        if save_png:
            out = self.save_dir / filename
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"✓ Saved: {out}")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_weights(metrics: list[str]) -> dict[str, float]:
        """
        Assign per-metric weights that sum to 1.0.

        Faithfulness carries the most weight (it directly tests whether the
        heatmap reflects the model's true decision process), followed by
        sharpness, then stability, then localization.
        """
        base = {
            "deletion_auc":  0.20,
            "insertion_auc": 0.20,
            "sparsity":      0.12,
            "concentration": 0.13,
            "stability":     0.15,
            "iou":           0.12,
            "pointing_game": 0.08,
        }
        active = {m: base[m] for m in metrics if m in base}
        total  = sum(active.values())
        if total == 0:
            return {m: 1.0 / len(metrics) for m in metrics}
        return {m: v / total for m, v in active.items()}


# ---------------------------------------------------------------------------
# Plotting helpers (module-level to keep class readable)
# ---------------------------------------------------------------------------

def _overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Apply a jet heatmap overlay on a float32 RGB image."""
    import matplotlib.cm as cm
    heatmap = cm.jet(cam)[..., :3].astype(np.float32)
    overlay = 0.55 * heatmap + 0.45 * img
    return np.clip(overlay, 0, 1)


def _panel_label(fig, subplot_spec, text: str, color: str, scale: float):
    """Draw a small label above a grid-spec panel."""
    import matplotlib.transforms as mtransforms
    bbox = subplot_spec.get_position(fig)
    fig.text(
        bbox.x0, bbox.y1 + 0.005 * scale,
        text, color=color,
        fontsize=8 * scale, fontfamily="monospace",
        transform=fig.transFigure,
    )


def _draw_radar(ax, methods, metrics, scores, colors, text_color, sub_color, bg, scale):
    """Draw a radar/spider chart for per-metric scores."""
    import matplotlib.pyplot as plt

    N    = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    ax.set_facecolor(bg)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color=text_color, fontsize=8 * scale,
                       fontfamily="monospace")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(color="#30363D", linewidth=0.8)
    ax.spines["polar"].set_color("#30363D")

    for method in methods:
        vals = [scores[method].get(m, 0.0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[method], linewidth=1.8,
                linestyle="solid", label=method)
        ax.fill(angles, vals, color=colors[method], alpha=0.10)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        labelcolor=text_color, facecolor=bg,
        edgecolor="#30363D", fontsize=8 * scale,
    )