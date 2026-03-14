import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
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

    Supported models:
        Any differentiable PyTorch model with output shape [batch, num_classes]
        or [batch, 1] (sigmoid binary). Works with: ResNet, VGG, EfficientNet,
        DenseNet, MobileNet, ViT, custom CNNs, InceptionV3, GoogLeNet, and more.

    Does NOT work with:
        Tree models (Random Forest, XGBoost), models with argmax/hard one-hot
        layers, or any operation that breaks gradient flow.

    Baseline:
        Default is a black image (all zeros). This represents zero visual
        information. Attributions measure pixel contribution vs. seeing nothing.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Args:
            model  : A trained PyTorch model. Automatically put in eval() mode.
            device : Target device. Defaults to the model's current device.

        Note:
            We do NOT set requires_grad=False on model parameters here.
            IG gradients are computed w.r.t. the INTERPOLATED INPUT, not the
            model weights. Disabling parameter gradients would permanently break
            the model for the user (fine-tuning, continued training would fail).
            torch.no_grad() is used only where model forward passes do not need
            gradient tracking (target-class detection, convergence check).
        """
        self.model = model.eval()
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
        alpha: float = 0.5,
    ) -> Dict:
        """
        Run Integrated Gradients and return all 4 visualizations.

        Args:
            input_tensor  : Preprocessed image tensor [1, C, H, W].
                            Must be float, normalised the same way as during training.
            original_bgr  : Original image as BGR numpy array [H, W, 3].
                            Used only for overlay rendering, not for IG computation.
            target_class  : Class index to explain. None = model's top prediction.
            baseline      : Reference tensor [1, C, H, W]. None = black image (zeros).
                            MUST use the same normalisation as input_tensor.
            steps         : Riemann sum steps. More = more accurate. 100-300 is ideal.
            batch_size    : Steps processed per forward pass. Lower if VRAM is limited.
            alpha         : Heatmap blend strength [0.0-1.0].
                            0 = original image only, 1 = heatmap only. Default 0.5.

        Returns:
            dict with keys:
                target_class       : int   — the class index that was explained
                convergence_delta  : float — <0.05 excellent, <0.15 ok, >0.15 bad
                overlay_magnitude  : BGR ndarray — overall importance heatmap
                overlay_positive   : BGR ndarray — pixels supporting the prediction
                overlay_negative   : BGR ndarray — pixels suppressing the prediction
                overlay_contour    : BGR ndarray — contour of most important region

        Raises:
            ValueError : if input_tensor is not shape [1, C, H, W]
            ValueError : if baseline shape does not match input_tensor
            ValueError : if steps < 20 (too few for reliable approximation)
            ValueError : if alpha is outside [0.0, 1.0]
        """
        # ── Input validation ──────────────────────────────────────────
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be shape [1, C, H, W], got {tuple(input_tensor.shape)}. "
                f"If you have a batch, use input_tensor[i:i+1] to select one image."
            )
        if steps < 20:
            raise ValueError(
                f"steps={steps} is too low. Use at least 50 (recommended: 100-300)."
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha={alpha} is out of range. Must be between 0.0 and 1.0."
            )

        # ── Device & baseline setup ───────────────────────────────────
        input_tensor = input_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)  # black image = no visual signal
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

                # Some models (InceptionV3, GoogLeNet) return (output, aux) tuple
                # in train mode. eval() mode returns a plain tensor, but guard anyway.
                if isinstance(logits, tuple):
                    logits = logits[0]

                # A 1D output means the batch dimension was accidentally squeezed.
                # We cannot safely determine class count, so raise clearly.
                if logits.dim() == 1:
                    raise ValueError(
                        f"Model output has shape {tuple(logits.shape)} (1D). "
                        f"Expected [batch, num_classes] or [batch, 1]. "
                        f"Check your model's forward() — it may be squeezing the batch dim."
                    )

                if logits.shape[1] == 1:          # sigmoid binary output [batch, 1]
                    target_class = int(logits.squeeze() > 0.5)
                else:
                    target_class = logits.argmax(dim=1).item()

        # ── Run IG ────────────────────────────────────────────────────
        attributions, delta = self._compute_attributions(
            input_tensor, baseline, target_class, steps, batch_size
        )

        # ── Build attribution maps ────────────────────────────────────
        mag_map = self._magnitude_map(attributions)
        pos_map = self._positive_map(attributions)
        neg_map = self._negative_map(attributions)

        return {
            "target_class":      target_class,
            "convergence_delta": delta,
            "overlay_magnitude": self._heatmap_overlay(
                mag_map, original_bgr, alpha, cv2.COLORMAP_JET
            ),
            "overlay_positive":  self._heatmap_overlay(
                pos_map, original_bgr, alpha, cv2.COLORMAP_HOT
            ),
            "overlay_negative":  self._heatmap_overlay(
                neg_map, original_bgr, alpha, cv2.COLORMAP_WINTER
            ),
            "overlay_contour":   self._contour_overlay(
                mag_map, original_bgr, threshold_pct=90.0
            ),
        }

    def save_dashboard(
        self,
        results: Dict,
        save_path: str,
        class_name: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """
        Save a 2x2 explanation dashboard as a PNG.

        Layout:
            [ Magnitude  |  Positive ]
            [ Negative   |  Contour  ]

        Args:
            results    : Output dict from .explain()
            save_path  : File path (e.g. "explanation.png")
            class_name : Human-readable class label for the dashboard title
            dpi        : Output resolution (150 for screens, 300 for print)
        """
        delta = results["convergence_delta"]
        cls_str = f"Class: {class_name}" if class_name else f"Class: {results['target_class']}"
        quality = (
            "[EXCELLENT]" if delta < 0.05 else
            "[OK]"        if delta < 0.15 else
            "[!!] Increase steps"
        )

        panels = [
            ("Magnitude  (Overall Importance)",    "overlay_magnitude"),
            ("Positive   (Supports Prediction)",   "overlay_positive"),
            ("Negative   (Suppresses Prediction)", "overlay_negative"),
            ("Contour    (Important Region)",       "overlay_contour"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#111122")
        fig.suptitle(
            f"Integrated Gradients \u2014 Image Explanation\n"
            f"{cls_str}  |  Convergence \u0394 = {delta:.4f}  {quality}",
            color="white", fontsize=12, fontweight="bold", y=0.98,
        )

        for ax, (title, key) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
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
        """
        Riemann sum approximation of the IG path integral.

        Math:
            IG(pixel_i) = (input_i - baseline_i)
                          * (1/steps) * SUM_a [ dF(baseline + a*(input-baseline)) / d(pixel_i) ]

        We skip alpha=0 (the baseline itself) because its gradient contributes
        nothing to the path difference. So alphas runs from 1/steps to 1.0.
        """
        delta_path = input_tensor - baseline  # [1, C, H, W]

        # Skip alpha=0 (baseline has zero path contribution)
        alphas = torch.linspace(0, 1, steps + 1)[1:].to(self.device)  # [steps]
        accum_grad = torch.zeros_like(input_tensor)

        for start in range(0, steps, batch_size):
            batch_alphas = alphas[start: start + batch_size]  # [B]

            # Interpolate: baseline + alpha * (input - baseline)
            # view(-1,1,1,1) broadcasts alpha across [C, H, W] dimensions
            scaled = (
                baseline + batch_alphas.view(-1, 1, 1, 1) * delta_path
            ).clone().detach().requires_grad_(True)

            output = self.model(scaled)

            # Handle tuple output (InceptionV3, GoogLeNet in train mode)
            if isinstance(output, tuple):
                output = output[0]

            # Handle sigmoid [batch,1] output: convert to 2-class for indexing
            if output.shape[1] == 1:
                output = torch.cat([1 - output, output], dim=1)

            score = output[:, target_class].sum()
            grads = torch.autograd.grad(score, scaled)[0]
            accum_grad += grads.detach().sum(dim=0, keepdim=True)

        # Final IG formula
        attributions = delta_path * (accum_grad / steps)

        delta_val = self._convergence_delta(
            attributions, input_tensor, baseline, target_class
        )
        return attributions.detach(), delta_val

    def _convergence_delta(
        self,
        attributions: torch.Tensor,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
    ) -> float:
        """
        Completeness check (the Completeness Axiom of IG):

            |sum(attributions)  -  (F(input) - F(baseline))|

        A perfect continuous integral gives 0. Our Riemann sum gives a small
        positive error. Guidelines:
            < 0.05  = excellent
            < 0.15  = acceptable
            > 0.15  = increase `steps`
        """
        with torch.no_grad():
            out_in = self.model(input_tensor)
            out_bl = self.model(baseline)

            # Handle tuple output (InceptionV3, GoogLeNet in train mode)
            if isinstance(out_in, tuple):
                out_in = out_in[0]
                out_bl = out_bl[0]

            # Apply same sigmoid conversion for consistency
            if out_in.shape[1] == 1:
                out_in = torch.cat([1 - out_in, out_in], dim=1)
                out_bl = torch.cat([1 - out_bl, out_bl], dim=1)
            f_in = out_in[0, target_class].item()
            f_bl = out_bl[0, target_class].item()
        return abs(attributions.sum().item() - (f_in - f_bl))

    # ------------------------------------------------------------------
    # Attribution map builders  (private)
    # ------------------------------------------------------------------

    def _magnitude_map(self, attr: torch.Tensor) -> torch.Tensor:
        """
        Sum of |attributions| across colour channels.
        Result shape: [H, W]

        Answers: which pixels matter at all, in any direction?
        Red on this map = important pixel. Blue = unimportant background.
        """
        return torch.sum(torch.abs(attr.squeeze(0)), dim=0)

    def _positive_map(self, attr: torch.Tensor) -> torch.Tensor:
        """
        Sum of only positive attributions across channels.
        Result shape: [H, W]

        Answers: which pixels PUSHED the model TOWARD this class?
        Hot pixels on this map = "this is why the model said cat/dog/disease".
        """
        return torch.sum(torch.clamp(attr.squeeze(0), min=0), dim=0)

    def _negative_map(self, attr: torch.Tensor) -> torch.Tensor:
        """
        Absolute sum of negative attributions across channels.
        Result shape: [H, W]

        Answers: which pixels PUSHED the model AWAY from this class?
        Useful for debugging: "what made the model doubt this prediction?"
        We take abs so the map is positive (needed for heatmap rendering).
        """
        return torch.abs(torch.sum(torch.clamp(attr.squeeze(0), max=0), dim=0))

    # ------------------------------------------------------------------
    # Visualization helpers  (private)
    # ------------------------------------------------------------------

    def _to_uint8(self, attr_map: torch.Tensor, clip_pct: float = 99.0) -> np.ndarray:
        """
        Normalise attribution map to [0, 255] uint8 for OpenCV rendering.

        Steps:
          1. Cast to float32 (safe for numpy operations)
          2. Clip at 99th percentile (removes extreme outlier pixels)
          3. Shift minimum to 0
          4. Normalise maximum to 1
          5. Scale to [0, 255] and cast to uint8

        The 99th percentile clip prevents one extremely hot pixel from
        washing out the entire heatmap to a uniform colour.
        """
        arr = attr_map.cpu().numpy().astype(np.float32)
        upper = np.percentile(arr, clip_pct)

        # Guard: if all attributions are zero (e.g. no positive signal),
        # return a black map rather than dividing by zero.
        if upper <= 0:
            return np.zeros(arr.shape, dtype=np.uint8)

        arr = np.clip(arr, 0, upper)
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        return np.uint8(255 * arr)

    def _resize(self, heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Resize heatmap to match image spatial dimensions if they differ.

        This is needed when the model's internal resolution differs from the
        display image resolution (e.g. ViT with patch-level attributions,
        or when the original BGR was not resized to the model's input size).
        Note: cv2.resize takes (width, height), opposite of numpy (height, width).
        """
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (image.shape[1], image.shape[0]),   # (width, height) for cv2
                interpolation=cv2.INTER_LINEAR,
            )
        return heatmap

    def _heatmap_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        alpha: float,
        colormap: int,
    ) -> np.ndarray:
        """
        Blend a colourised attribution heatmap over the original image.

        Pipeline:
            attribution map [H,W] float
            → _to_uint8: normalise to [0,255] grayscale
            → _resize: match image spatial dims
            → cv2.applyColorMap: grayscale → 3-channel colour (JET/HOT/WINTER)
            → cv2.addWeighted: blend with original image at alpha strength
        """
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)
        colored = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

    def _contour_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        threshold_pct: float = 90.0,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw green contours around the most important image regions.

        Pipeline:
            attribution map
            → _to_uint8: normalise
            → _resize: match image dims
            → GaussianBlur(25,25): smooth out pixel-level IG noise so contours
              outline regions, not individual speckled pixels
            → threshold at 90th percentile: keep only top 10% important pixels
            → findContours: detect connected important regions
            → filter by area (> 0.5% of image): remove tiny noise fragments
            → drawContours on image copy

        If the entire map is zero (no attributions), returns image unchanged.
        """
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

        threshold = int(np.percentile(heatmap, threshold_pct))

        # Guard: all-zero map (no attributions found)
        if threshold == 0:
            return image.copy()

        _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            min_area = image.shape[0] * image.shape[1] * 0.005
            contours = [c for c in contours if cv2.contourArea(c) > min_area]

        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness)
        return overlay