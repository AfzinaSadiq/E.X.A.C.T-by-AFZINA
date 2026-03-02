import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg") # Agg stands for Anti-Grain Geometry # used when we dont need to display plots but only want to save them to files
import matplotlib.pyplot as plt
from typing import Dict, Optional


class IGImageExplainer:
    """
    Integrated Gradients explainer for any PyTorch image model.

    Produces 4 visualizations:
        1. Magnitude overlay   — overall importance (which regions matter)
        2. Positive overlay    — regions that support the prediction
        3. Negative overlay    — regions that suppress the prediction
        4. Contour overlay     — boundary of the most important region

    Quick start:
        explainer = IGImageExplainer(model)
        results   = explainer.explain(img_tensor, img_bgr)
        explainer.save_dashboard(results, "explanation.png")
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model.eval() # switches the model to evaluation mode
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_bgr: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 200,
        batch_size: int = 32,
        alpha: float = 0.5, # how strongly to blend the heatmap over the original image, 0.5 means 50% heatmap, 50% original.
    ) -> Dict:
        """
        Run IG and return all visualizations.

        Args:
            input_tensor  : Preprocessed image tensor [1, C, H, W]
            original_bgr  : Original image as BGR numpy array [H, W, 3]
            target_class  : Class index to explain. None = predicted class.
            baseline      : Reference tensor [1, C, H, W]. None = black image.
            steps         : Riemann sum steps. More = more accurate (100-300).
            batch_size    : Steps processed per forward pass (tune for VRAM).
            alpha         : Heatmap blend strength (0 = image, 1 = heatmap).

        Returns:
            dict:
                target_class       : int
                convergence_delta  : float  (< 0.05 is good)
                overlay_magnitude  : BGR ndarray — main importance heatmap
                overlay_positive   : BGR ndarray — what supports the prediction
                overlay_negative   : BGR ndarray — what suppresses the prediction
                overlay_contour    : BGR ndarray — boundary of important region
        """
        input_tensor = input_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        baseline = baseline.to(self.device)

        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_tensor).argmax(dim=1).item()

        attributions, delta = self._compute_attributions(
            input_tensor, baseline, target_class, steps, batch_size
        )
 
        mag_map = self._magnitude_map(attributions)
        pos_map = self._positive_map(attributions)
        neg_map = self._negative_map(attributions)

        return {
            "target_class":      target_class,
            "convergence_delta": delta,
            "overlay_magnitude": self._heatmap_overlay(mag_map, original_bgr, alpha, cv2.COLORMAP_JET),
            "overlay_positive":  self._heatmap_overlay(pos_map, original_bgr, alpha, cv2.COLORMAP_HOT),
            "overlay_negative":  self._heatmap_overlay(neg_map, original_bgr, alpha, cv2.COLORMAP_WINTER),
            "overlay_contour":   self._contour_overlay(mag_map, original_bgr, threshold_pct=90.0),
        }

    def save_dashboard(
        self,
        results: Dict,
        save_path: str,
        class_name: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """
        Saves a 2x2 explanation dashboard as a PNG.

        Layout:
            [ Magnitude  |  Positive ]
            [ Negative   |  Contour  ]

        Args:
            results    : Output dict from .explain()
            save_path  : Where to save (e.g. "explanation.png")
            class_name : Human-readable class label for the title
            dpi        : Output resolution
        """
        delta = results["convergence_delta"]
        cls_str = f"Class: {class_name}" if class_name else f"Class: {results['target_class']}"
        quality = "[OK]" if delta < 0.15 else "[!!] Increase steps"

        panels = [
            ("Magnitude  (Overall Importance)",    "overlay_magnitude"),
            ("Positive   (Supports Prediction)",   "overlay_positive"),
            ("Negative   (Suppresses Prediction)", "overlay_negative"),
            ("Contour    (Important Region)",       "overlay_contour"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#111122")
        fig.suptitle(
            f"Integrated Gradients - Explanation\n"
            f"{cls_str} | Convergence Δ = {delta:.4f}  {quality}",
            color="white", fontsize=12, fontweight="bold", y=0.98,
        )

        for ax, (title, key) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    # ------------------------------------------------------------------
    # Core IG computation  (private)
    # ------------------------------------------------------------------

    def _compute_attributions(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        steps: int,
        batch_size: int,
    ):
        delta = input_tensor - baseline # the total difference between input and baseline. This is the "path" we will walk along.
        alphas = torch.linspace(0, 1, steps + 1)[1:].to(self.device) # Creates steps evenly-spaced values from just above 0 to 1. linspace(0, 1, 201) gives 201 values; [1:] removes the first one (alpha=0, which is just the baseline itself). So we get alphas: [0.005, 0.010, ..., 1.0]. Each alpha represents a point along the path
        accum_grad = torch.zeros_like(input_tensor)

        for start in range(0, steps, batch_size): # Instead of computing one interpolated image at a time (slow), we process batch_size steps at once. range(0, 200, 32) gives [0, 32, 64, 96, 128, 160, 192] — seven batches.
            batch_alphas = alphas[start : start + batch_size]
            scaled = (baseline + batch_alphas.view(-1, 1, 1, 1) * delta).clone().detach().requires_grad_(True)

            output = self.model(scaled)
            score = output[:, target_class].sum()
            grads = torch.autograd.grad(score, scaled)[0]
            accum_grad += grads.detach().sum(dim=0, keepdim=True)

        attributions = delta * (accum_grad / steps) # Final IG formula: divide accumulated gradients by total steps to get the average gradient, then multiply by delta (the input-baseline difference). This gives the final attribution for each pixel.
        delta_val = self._convergence_delta(attributions, input_tensor,
                                               baseline, target_class)
        return attributions.detach(), delta_val

    def _convergence_delta( # this function is basically completeness check
        self,
        attributions: torch.Tensor,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
    ) -> float:
        with torch.no_grad():
            f_in = self.model(input_tensor)[0, target_class].item()
            f_bl = self.model(baseline)[0, target_class].item()
        return abs(attributions.sum().item() - (f_in - f_bl))

    # ------------------------------------------------------------------
    # Attribution maps  (private)
    # ------------------------------------------------------------------

    def _magnitude_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Sum of |attributions| across channels - overall importance."""
        return torch.sum(torch.abs(attr.squeeze(0)), dim=0)
        # attr is [1, 3, H, W]. .squeeze(0) removes the batch dimension → [3, H, W]. torch.abs takes absolute value of every element. torch.sum(..., dim=0) sums across the 3 color channels → [H, W]. 
        # Result: one importance score per pixel combining all channels

    def _positive_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Sum of positive attributions - what supports the prediction."""
        return torch.sum(torch.clamp(attr.squeeze(0), min=0), dim=0)

    def _negative_map(self, attr: torch.Tensor) -> torch.Tensor:
        """Abs sum of negative attributions - what suppresses the prediction."""
        return torch.abs(torch.sum(torch.clamp(attr.squeeze(0), max=0), dim=0))

    # ------------------------------------------------------------------
    # Visualization helpers  (private)
    # ------------------------------------------------------------------

    def _to_uint8(self, attr_map: torch.Tensor, clip_pct: float = 99.0) -> np.ndarray:
        # This function is for Normalizaing visualization
        arr = attr_map.cpu().numpy() # moves tensor to CPU and converts to numpy array
        upper = np.percentile(arr, clip_pct) # finds the 99th percentile value. Pixels above this are extreme outliers
        arr = np.clip(arr, 0, upper)
        arr = arr - arr.min() # shifts minimum to 0
        arr = arr / (arr.max() + 1e-8) # divides by maximum to get range [0, 1]. The 1e-8 prevents division by zero if the map is all zeros
        return np.uint8(255 * arr) # scales to [0, 255] and converts to unsigned 8-bit integer

    def _resize(self, heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        return heatmap

    def _heatmap_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        alpha: float,
        colormap: int,
    ) -> np.ndarray:
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)
        colored = cv2.applyColorMap(heatmap, colormap) # cv2.applyColorMap converts the grayscale [0-255] heatmap into a color image using the specified colormap - JET gives the classic rainbow. 
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0) # cv2.addWeighted blends two images: image × (1-alpha) + colored × alpha. At alpha=0.5 you see 50% original image and 50% heatmap color.

    def _contour_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        threshold_pct: float = 90.0,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)

        # Smooth before thresholding - removes pixel-level IG noise so
        # contours outline regions rather than individual noisy pixels
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

        threshold = int(np.percentile(heatmap, threshold_pct))
        _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Only draw the largest contours - skip tiny noisy fragments
        if contours:
            min_area = image.shape[0] * image.shape[1] * 0.005  # 0.5% of image
            contours = [c for c in contours if cv2.contourArea(c) > min_area]

        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness)
        return overlay