# explainers/ig_image_explainer.py

"""
IGImageExplainer
================
Integrated Gradients explainer for any PyTorch image model.

Produces 4 attribution maps:
    1. Magnitude  — overall importance  (which regions matter)
    2. Positive   — regions that support the prediction
    3. Negative   — regions that suppress the prediction
    4. Contour    — boundary of the most important region

EXACT compatibility
-------------------
explain() returns a standardised result dict with a 'cam' key containing
the magnitude heatmap as a (H, W) float32 array in [0, 1].  This makes
IGImageExplainer a drop-in for HeatmapComparator alongside GradCAM, LIME, etc.

    explainer = IGImageExplainer(model)
    result    = explainer.explain(input_tensor, input_image=img_np)

    # Direct use with HeatmapComparator:
    cmp.compare(
        entries={
            "GradCAM": (gradcam_result, gradcam_exp, {"method": "gradcam"}),
            "IG":      (ig_result,      ig_exp,      {}),
        },
        input_tensor=input_tensor,
        input_image=img_np,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class IGImageExplainer:
    """
    Integrated Gradients explainer for any PyTorch image model.

    Supported models
    ----------------
    Any differentiable PyTorch model with output shape [batch, num_classes]
    or [batch, 1] (sigmoid binary). Works with ResNet, VGG, EfficientNet,
    DenseNet, MobileNet, ViT, custom CNNs, InceptionV3, GoogLeNet, and more.

    Does NOT work with
    ------------------
    Tree models (Random Forest, XGBoost), models with argmax/hard one-hot
    layers, or any operation that breaks gradient flow.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model. Automatically put in eval() mode.
    device : torch.device, optional
        Target device. Defaults to the model's current device.
    save_dir : str, optional
        Directory for saved outputs. Default 'user_saves/ig_saves'.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        save_dir: str = "user_saves/ig_saves",
    ):
        self.model    = model.eval()
        self.device   = device or next(model.parameters()).device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        input_tensor: torch.Tensor,
        input_image: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 200,
        batch_size: int = 32,
        alpha: float = 0.5,
        save_png: bool = False,
        class_name: str = "",
    ) -> Dict:
        """
        Run Integrated Gradients and return a standardised EXACT result dict.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor [1, C, H, W]. Must be float, normalised
            the same way as during training.
        input_image : np.ndarray, optional
            Original image for overlay rendering. Accepts any of:
              - uint8  (H, W, 3) in [0, 255]   — RGB or BGR
              - float32 (H, W, 3) in [0, 1]
              - float32 (H, W, 3) in [0, 255]
              - (3, H, W) channel-first arrays
            If None, derived from input_tensor.
        target_class : int, optional
            Class index to explain. None = model's top predicted class.
        baseline : torch.Tensor, optional
            Reference tensor [1, C, H, W]. None = black image (zeros).
            Must use the same normalisation as input_tensor.
        steps : int
            Riemann sum steps. More = more accurate. 100–300 is ideal.
        batch_size : int
            Steps per forward pass. Reduce if VRAM is limited.
        alpha : float
            Heatmap blend strength [0.0–1.0]. Default 0.5.
        save_png : bool
            Whether to save the dashboard PNG. Default False.
        class_name : str
            Human-readable class label for the dashboard title and filename.

        Returns
        -------
        dict with keys:
            'heatmap'          : np.ndarray (H, W) float32 in [0, 1]
                                 Magnitude attribution map — used by HeatmapComparator.
            'visualization'    : np.ndarray (H, W, 3) float32 in [0, 1]
                                 Magnitude heatmap blended over input_image (RGB).
            'filepath'         : Path or None
            'target_class'     : int
            'convergence_delta': float  (<0.05 excellent, <0.15 ok, >0.15 bad)
            'overlay_magnitude': np.ndarray (H, W, 3) uint8 RGB
            'overlay_positive' : np.ndarray (H, W, 3) uint8 RGB
            'overlay_negative' : np.ndarray (H, W, 3) uint8 RGB
            'overlay_contour'  : np.ndarray (H, W, 3) uint8 RGB
        """
        # ── Validate ──────────────────────────────────────────────────
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be shape [1, C, H, W], got {tuple(input_tensor.shape)}."
            )
        if steps < 20:
            raise ValueError(f"steps={steps} is too low. Use at least 50 (recommended: 100-300).")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha={alpha} must be between 0.0 and 1.0.")

        # ── Prepare display image (RGB float32 in [0,1]) ──────────────
        if input_image is None:
            input_image = _tensor_to_display(input_tensor)
        else:
            input_image = _to_display_image(input_image)
        h, w = input_image.shape[:2]

        # ── Device & baseline ─────────────────────────────────────────
        input_tensor = input_tensor.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            if baseline.shape != input_tensor.shape:
                raise ValueError(
                    f"baseline shape {tuple(baseline.shape)} must match "
                    f"input_tensor shape {tuple(input_tensor.shape)}."
                )
        baseline = baseline.to(self.device)

        # ── Auto-detect target class ──────────────────────────────────
        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.dim() == 1:
                    raise ValueError(
                        f"Model output shape {tuple(logits.shape)} is 1-D. "
                        f"Expected [batch, num_classes]."
                    )
                target_class = (
                    int(logits.squeeze() > 0.5)
                    if logits.shape[1] == 1
                    else logits.argmax(dim=1).item()
                )

        # ── Compute IG attributions ───────────────────────────────────
        attributions, delta = self._compute_attributions(
            input_tensor, baseline, target_class, steps, batch_size
        )

        # ── Build attribution maps ────────────────────────────────────
        mag_map = self._magnitude_map(attributions)   # torch (H_t, W_t)
        pos_map = self._positive_map(attributions)
        neg_map = self._negative_map(attributions)

        # ── 'cam' — magnitude map resized to display image, normalised ─
        # This is what HeatmapComparator reads for all metrics.
        cam = self._map_to_float(mag_map, h, w)       # (H, W) float32 [0,1]

        # ── RGB overlays (consistent with rest of EXACT) ──────────────
        # All overlays are RGB float32 [0,1]
        ov_mag = self._heatmap_overlay_rgb(mag_map, input_image, alpha, "jet")
        ov_pos = self._heatmap_overlay_rgb(pos_map, input_image, alpha, "hot")
        ov_neg = self._heatmap_overlay_rgb(neg_map, input_image, alpha, "winter")
        ov_cnt = self._contour_overlay_rgb(mag_map, input_image)

        # ── Save dashboard ────────────────────────────────────────────
        filepath = None
        if save_png:
            suffix   = f"_{class_name}" if class_name else ""
            filepath = self.save_dir / f"ig{suffix}.png"
            self._save_dashboard(
                overlays={
                    "Magnitude  (Overall Importance)":    ov_mag,
                    "Positive   (Supports Prediction)":   ov_pos,
                    "Negative   (Suppresses Prediction)": ov_neg,
                    "Contour    (Important Region)":      ov_cnt,
                },
                target_class=target_class,
                class_name=class_name,
                delta=delta,
                filepath=filepath,
            )
            print(f"✓ Saved: {filepath}")

        return {
            # ── EXACT standard keys ───────────────────────────────────
            "heatmap":           cam,          # (H,W) float32 [0,1] — for comparator
            "visualization":     ov_mag,       # (H,W,3) float32 [0,1] RGB — primary overlay
            "filepath":          filepath,
            # ── IG-specific keys ──────────────────────────────────────
            "target_class":      target_class,
            "convergence_delta": delta,
            "overlay_magnitude": ov_mag,
            "overlay_positive":  ov_pos,
            "overlay_negative":  ov_neg,
            "overlay_contour":   ov_cnt,
        }

    # ------------------------------------------------------------------
    # Core IG computation  (private)
    # ------------------------------------------------------------------

    def _compute_attributions(self, input_tensor, baseline, target_class, steps, batch_size):
        """
        Riemann sum approximation of the IG path integral.

            IG(x_i) = (x_i - b_i) * (1/steps) * SUM_a [ dF(b + a*(x-b)) / dx_i ]

        We skip alpha=0 (the baseline) because its gradient contributes
        nothing to the path difference.
        """
        delta_path = input_tensor - baseline
        alphas     = torch.linspace(0, 1, steps + 1)[1:].to(self.device)
        accum_grad = torch.zeros_like(input_tensor)

        for start in range(0, steps, batch_size):
            batch_alphas = alphas[start: start + batch_size]
            scaled = (
                baseline + batch_alphas.view(-1, 1, 1, 1) * delta_path
            ).clone().detach().requires_grad_(True)

            output = self.model(scaled)
            if isinstance(output, tuple):
                output = output[0]
            if output.shape[1] == 1:
                output = torch.cat([1 - output, output], dim=1)

            score = output[:, target_class].sum()
            grads = torch.autograd.grad(score, scaled)[0]
            accum_grad += grads.detach().sum(dim=0, keepdim=True)

        attributions = delta_path * (accum_grad / steps)
        delta        = self._convergence_delta(
            attributions, input_tensor, baseline, target_class
        )
        return attributions.detach(), delta

    def _convergence_delta(self, attributions, input_tensor, baseline, target_class) -> float:
        """
        Completeness check: |sum(attr) - (F(input) - F(baseline))|
        < 0.05 excellent  |  < 0.15 acceptable  |  > 0.15 increase steps
        """
        with torch.no_grad():
            out_in = self.model(input_tensor)
            out_bl = self.model(baseline)
            if isinstance(out_in, tuple):
                out_in, out_bl = out_in[0], out_bl[0]
            if out_in.shape[1] == 1:
                out_in = torch.cat([1 - out_in, out_in], dim=1)
                out_bl = torch.cat([1 - out_bl, out_bl], dim=1)
        return abs(attributions.sum().item() - (
            out_in[0, target_class].item() - out_bl[0, target_class].item()
        ))

    # ------------------------------------------------------------------
    # Attribution map builders  (private)
    # ------------------------------------------------------------------

    def _magnitude_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Sum of |attr| across channels → (H, W). Answers: which pixels matter?"""
        return torch.sum(torch.abs(attr.squeeze(0)), dim=0)

    def _positive_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Sum of positive attr across channels → (H, W). Supports prediction."""
        return torch.sum(torch.clamp(attr.squeeze(0), min=0), dim=0)

    def _negative_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Abs sum of negative attr across channels → (H, W). Suppresses prediction."""
        return torch.abs(torch.sum(torch.clamp(attr.squeeze(0), max=0), dim=0))

    # ------------------------------------------------------------------
    # Visualisation helpers  (private, all output RGB float32 [0,1])
    # ------------------------------------------------------------------

    def _map_to_float(self, attr_map: torch.Tensor, h: int, w: int) -> np.ndarray:
        """
        Convert a torch attribution map to a normalised (H, W) float32 [0,1]
        numpy array resized to (h, w). Used to produce the 'cam' key.
        """
        arr = attr_map.cpu().numpy().astype(np.float32)
        upper = np.percentile(arr, 99)
        if upper <= 0:
            return np.zeros((h, w), dtype=np.float32)
        arr = np.clip(arr, 0, upper)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32)

    def _to_uint8_map(self, attr_map: torch.Tensor) -> np.ndarray:
        """Normalise attribution map to (H, W) uint8 [0,255] for colourmap application."""
        arr   = attr_map.cpu().numpy().astype(np.float32)
        upper = np.percentile(arr, 99)
        if upper <= 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip(arr, 0, upper)
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        return np.uint8(255 * arr)

    def _heatmap_overlay_rgb(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,          # RGB float32 [0,1]
        alpha: float,
        colormap: str,              # matplotlib colormap name
    ) -> np.ndarray:
        """
        Blend a colourised attribution heatmap over the display image.
        Returns RGB float32 [0,1] — consistent with all other EXACT overlays.
        """
        import matplotlib.cm as cm_mod
        h, w = image.shape[:2]

        gray = self._to_uint8_map(attr_map)                         # (H_t, W_t) uint8
        if gray.shape != (h, w):
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply matplotlib colormap → RGBA float [0,1] → drop alpha → RGB
        cmap    = cm_mod.get_cmap(colormap)
        colored = cmap(gray / 255.0)[..., :3].astype(np.float32)   # (H, W, 3) RGB [0,1]

        return np.clip((1 - alpha) * image + alpha * colored, 0, 1)

    def _contour_overlay_rgb(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,          # RGB float32 [0,1]
        threshold_pct: float = 90.0,
        color: tuple = (0.0, 1.0, 0.0),   # RGB [0,1] — bright green
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw contours around the most important regions on the display image.
        Returns RGB float32 [0,1].
        """
        h, w = image.shape[:2]
        gray = self._to_uint8_map(attr_map)
        if gray.shape != (h, w):
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

        gray    = cv2.GaussianBlur(gray, (25, 25), 0)
        thresh  = int(np.percentile(gray, threshold_pct))
        if thresh == 0:
            return image.copy()

        _, binary  = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            min_area = h * w * 0.005
            contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # Draw on a uint8 copy then convert back to float32 [0,1]
        overlay_u8 = np.uint8(image * 255)
        color_u8   = tuple(int(c * 255) for c in color)
        cv2.drawContours(overlay_u8, contours, -1, color_u8, thickness)
        return overlay_u8.astype(np.float32) / 255.0

    def _save_dashboard(
        self,
        overlays: dict,
        target_class: int,
        class_name: str,
        delta: float,
        filepath: Path,
        dpi: int = 150,
    ) -> None:
        """Save a 2×2 dashboard of the four RGB overlays."""
        quality = (
            "[EXCELLENT]"       if delta < 0.05 else
            "[OK]"              if delta < 0.15 else
            "[!!] Increase steps"
        )
        cls_str = f"Class: {class_name}" if class_name else f"Class: {target_class}"

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#111122")
        fig.suptitle(
            f"Integrated Gradients — Image Explanation\n"
            f"{cls_str}  |  Convergence Δ = {delta:.4f}  {quality}",
            color="white", fontsize=12, fontweight="bold", y=0.98,
        )
        for ax, (title, overlay) in zip(axes.flat, overlays.items()):
            ax.imshow(overlay)    # overlay is already RGB float32 [0,1]
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(str(filepath), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)


# ---------------------------------------------------------------------------
# Display image helpers  (module-level, mirrors comparator convention)
# ---------------------------------------------------------------------------

def _to_display_image(img: np.ndarray) -> np.ndarray:
    """
    Convert any image array to RGB float32 in [0, 1].
    Handles: uint8 [0,255], float [0,255], float [0,1], channel-first (C,H,W),
    and ImageNet-normalised arrays with negative values.
    """
    img = np.array(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] == 3:       # (C,H,W) → (H,W,C)
        img = np.transpose(img, (1, 2, 0))
    if img.max() > 1.0:
        img = img / 255.0
    if img.min() < 0.0:                            # normalised tensor → shift to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip(img, 0.0, 1.0)


def _tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) or (C,H,W) tensor → RGB float32 (H,W,3) in [0,1]."""
    img = t[0] if t.ndim == 4 else t
    return _to_display_image(img.cpu().numpy())