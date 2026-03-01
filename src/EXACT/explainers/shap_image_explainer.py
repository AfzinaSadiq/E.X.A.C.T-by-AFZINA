"""
shap_image_explainer.py
-----------------------
SHAP-based explainability for PyTorch image classification models.
Part of the EXACT XAI library alongside shap_tabular_explainer.py,
shap_text_explainer.py, and shap_kernel_explainer.py.

Supports
--------
- RGB images          (H, W, 3)
- Grayscale images    (H, W) or (H, W, 1)
- Binary classifiers  (1 output neuron  -> sigmoid)
- Multi-class models  (N output neurons -> softmax)
- Any PyTorch model   CNN, ViT, EfficientNet, DenseNet, ResNet, etc.

SHAP Backends
-------------
"deep"      -> DeepExplainer. Fast, accurate. Best for CNNs.
               Automatically falls back to "partition" if the model is
               incompatible (e.g. Vision Transformers with attention layers).
"partition" -> PartitionExplainer. Slower, model-agnostic. Works with anything.

Auto-Save
---------
Every visualize() call automatically saves the plot as a .png to user_saves/.
Filename includes class name and timestamp so nothing is ever overwritten.

Quick Start
-----------
    from shap_image_explainer import SHAPImageExplainer

    explainer = SHAPImageExplainer(
        model,
        background_data = train_images[:50],
        explainer_type  = "deep",
        target_size     = (224, 224),
        normalize       = True,
        class_names     = ["no_tumor", "tumor"],
    )

    result = explainer.explain(image)
    explainer.visualize(result, image, style="heatmap")
    explainer.visualize(result, image, style="masked")
    explainer.visualize(result, image, style="signed")
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalization — used when normalize=True
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# All visualization PNGs are saved here automatically
_SAVE_DIR = "user_saves"

# Known PyTorch layer types that break DeepExplainer
_INCOMPATIBLE_LAYER_TYPES = (
    nn.MultiheadAttention,
)

# Class name substrings that signal attention / transformer architecture.
# Covers timm, HuggingFace, Swin, and custom ViT implementations.
_INCOMPATIBLE_NAME_PATTERNS = (
    "attention",
    "transformer",
    "selfattention",
    "multiheadattention",
    "windowattention",      # Swin Transformer
    "vit",
)


# ---------------------------------------------------------------------------
# SHAPImageExplainer
# ---------------------------------------------------------------------------

class SHAPImageExplainer:
    """
    SHAP explainer for PyTorch image classifiers.

    Parameters
    ----------
    model : torch.nn.Module
        Any PyTorch image classification model. Internally set to eval mode.

    background_data : list or np.ndarray, optional
        Reference images for DeepExplainer (ignored for "partition").
        Pass 50-100 real training images for best explanation quality.
        If None, a black image is used as fallback (less accurate).

    explainer_type : str
        "deep"      -> DeepExplainer (default). Auto-falls back to
                       "partition" if the model is incompatible (e.g. ViT).
        "partition" -> PartitionExplainer. Always safe with any model.

    target_size : tuple of (int, int)
        (height, width) your model expects as input.

    normalize : bool
        True  -> apply ImageNet mean/std normalization.
                 Use for models pretrained on ImageNet.
        False -> only scale pixels to [0, 1].
                 Use for custom-trained models (e.g. medical imaging CNNs).

    custom_mean : list of float, optional
        Per-channel mean. Overrides ImageNet default when normalize=True.
        Example: [0.5, 0.5, 0.5]

    custom_std : list of float, optional
        Per-channel std. Overrides ImageNet default when normalize=True.
        Example: [0.5, 0.5, 0.5]

    max_evals : int
        Evaluations for PartitionExplainer. Higher = more accurate, slower.
        Default: 500.

    class_names : list of str, optional
        Human-readable class labels used in plots and explain() output.
        Example: ["benign", "malignant"]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: Optional[Union[List, np.ndarray]] = None,
        explainer_type: str = "deep",
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = False,
        custom_mean: Optional[List[float]] = None,
        custom_std: Optional[List[float]] = None,
        max_evals: int = 500,
        class_names: Optional[List[str]] = None,
    ):
        if explainer_type.lower() not in ("deep", "partition"):
            raise ValueError("explainer_type must be 'deep' or 'partition'.")

        self.target_size = tuple(target_size)
        self.normalize   = normalize
        self.max_evals   = max_evals
        self.class_names = class_names

        # Normalization parameters
        if normalize:
            self.norm_mean = (
                np.array(custom_mean, dtype=np.float32)
                if custom_mean else _IMAGENET_MEAN
            )
            self.norm_std = (
                np.array(custom_std, dtype=np.float32)
                if custom_std else _IMAGENET_STD
            )
        else:
            self.norm_mean = None
            self.norm_std  = None

        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.eval().to(self.device)

        # ── DeepExplainer compatibility check ────────────────────────────────
        # Runs at init so the user knows immediately, before any explain() call.
        # If "deep" is requested but the model is incompatible, we silently
        # switch to "partition" and print a clear message explaining why.
        if explainer_type.lower() == "deep":
            compatible, reason = self._check_deep_compatibility(
                self.model, self.target_size
            )
            if not compatible:
                print(
                    "[SHAPImageExplainer] WARNING: DeepExplainer is not compatible "
                    "with this model.\n"
                    f"  Reason : {reason}\n"
                    "  Action : Automatically switching to PartitionExplainer.\n"
                    "  Note   : Explanation quality is equivalent. "
                    "PartitionExplainer is slightly slower."
                )
                self.explainer_type = "partition"
            else:
                self.explainer_type = "deep"
        else:
            self.explainer_type = "partition"

        # Background data is only meaningful for DeepExplainer
        if self.explainer_type == "partition" and background_data is not None:
            warnings.warn(
                "background_data is unused by PartitionExplainer and will be ignored.",
                stacklevel=2,
            )
            self.background_data = None
        else:
            self.background_data = background_data

        # Internal state — explainer is built lazily on first explain() call
        self._explainer        = None
        self._last_image_shape = None

        # Create save directory upfront
        os.makedirs(_SAVE_DIR, exist_ok=True)

        print(
            f"[SHAPImageExplainer] Ready.\n"
            f"  Backend     : {self.explainer_type}\n"
            f"  Target size : {self.target_size}\n"
            f"  Normalize   : {self.normalize}\n"
            f"  Device      : {self.device}\n"
            f"  Plots saved : {os.path.abspath(_SAVE_DIR)}"
        )

    # -------------------------------------------------------------------------
    # DeepExplainer compatibility check
    # -------------------------------------------------------------------------

    @staticmethod
    def _check_deep_compatibility(
        model: nn.Module,
        target_size: Tuple[int, int],
    ) -> Tuple[bool, str]:
        """
        Check whether DeepExplainer can safely be used with this model.

        Uses three passes in order:

        Pass 1 — Type check
            Scans every module against known incompatible nn.Module types
            (e.g. nn.MultiheadAttention used in standard PyTorch ViTs).

        Pass 2 — Name pattern check
            Scans every module's class name against known incompatible
            string patterns. Catches third-party ViT implementations
            (timm, HuggingFace, custom) that don't use standard PyTorch types.

        Pass 3 — Live dry-run
            Actually tries to build shap.DeepExplainer and run one forward pass
            with a dummy input. Catches any model that slipped through passes 1
            and 2 (e.g. a custom attention layer with a completely non-standard name).
            This is the final safety net that prevents crashes at explain() time.

        Returns
        -------
        (True,  "")            compatible — safe to use DeepExplainer
        (False, reason_str)    incompatible — reason describes what was found
        """
        # ── Pass 1: known layer type check ───────────────────────────────────
        for name, module in model.named_modules():
            if isinstance(module, _INCOMPATIBLE_LAYER_TYPES):
                return (
                    False,
                    f"Found {type(module).__name__} at '{name}'. "
                    "This is a transformer attention layer incompatible with DeepExplainer."
                )

        # ── Pass 2: class name pattern check ─────────────────────────────────
        for name, module in model.named_modules():
            cls_name = type(module).__name__.lower()
            for pattern in _INCOMPATIBLE_NAME_PATTERNS:
                if pattern in cls_name:
                    return (
                        False,
                        f"Found layer '{type(module).__name__}' at '{name}' "
                        f"matching pattern '{pattern}'. "
                        "Model appears to contain attention or transformer blocks."
                    )

        # ── Pass 3: live dry-run ──────────────────────────────────────────────
        # Use a small dummy that matches the expected channel count.
        # We detect channel count by checking if the first Conv layer is 1 or 3.
        # Defaults to 3 (RGB) if nothing is found.
        try:
            device    = next(model.parameters()).device
            in_ch     = 3   # default RGB
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    in_ch = module.in_channels
                    break

            dummy_input = torch.zeros(2, in_ch, 32, 32, device=device)
            dummy_bg    = torch.zeros(1, in_ch, 32, 32, device=device)

            test_exp = shap.DeepExplainer(model, dummy_bg)
            _        = test_exp.shap_values(dummy_input[:1])

        except Exception as e:
            return (
                False,
                f"DeepExplainer dry-run failed with {type(e).__name__}: "
                f"{str(e)[:150]}. Model is not compatible with DeepExplainer."
            )

        return (True, "")

    # -------------------------------------------------------------------------
    # Image preprocessing
    # -------------------------------------------------------------------------

    def _preprocess(self, image: Any, for_display: bool = False) -> np.ndarray:
        """
        Convert any image input to a float32 numpy array of shape (H, W, C).

        Accepts
        -------
        str         : file path — loaded via PIL
        PIL.Image   : converted to numpy
        np.ndarray  : (H, W) or (H, W, C), uint8 or float32

        Steps
        -----
        1. Load from path or PIL if needed
        2. Add channel dimension if grayscale (H, W) -> (H, W, 1)
        3. Scale to [0, 1]
        4. Resize to target_size if needed
        5. Apply mean/std normalization only when for_display=False and
           normalize=True. Display images are always kept in [0, 1].
        """
        # Load
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: '{image}'")
            image = Image.open(image)

        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected str path, PIL.Image, or np.ndarray. "
                f"Got: {type(image).__name__}"
            )

        image = image.copy().astype(np.float32)

        # Ensure channel dim
        if image.ndim == 2:
            image = image[:, :, np.newaxis]     # (H, W) -> (H, W, 1)

        # Scale to [0, 1]
        if image.max() > 2.0:
            image /= 255.0
        image = np.clip(image, 0.0, 1.0)

        # Resize if needed
        H_t, W_t = self.target_size
        C = image.shape[2]
        if image.shape[:2] != (H_t, W_t):
            image = self._resize(image, H_t, W_t, C)

        # Normalize for model input (never for display)
        if not for_display and self.normalize and self.norm_mean is not None:
            image = self._apply_norm(image)

        return image    # (H, W, C)  float32

    @staticmethod
    def _resize(image: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
        """Resize (H, W, C) array to (H, W, C) using PIL. Channel-aware."""
        if C == 1:
            pil = Image.fromarray(
                (image[:, :, 0] * 255).astype(np.uint8), mode="L"
            )
            pil = pil.resize((W, H), Image.BILINEAR)
            return np.array(pil, dtype=np.float32)[:, :, np.newaxis] / 255.0

        mode = {3: "RGB", 4: "RGBA"}.get(C)
        if mode:
            pil = Image.fromarray((image * 255).astype(np.uint8), mode=mode)
            pil = pil.resize((W, H), Image.BILINEAR)
            return np.array(pil, dtype=np.float32) / 255.0

        # Any other channel count: resize each channel independently
        channels = []
        for c in range(C):
            pil = Image.fromarray(
                (image[:, :, c] * 255).astype(np.uint8), mode="L"
            )
            pil = pil.resize((W, H), Image.BILINEAR)
            channels.append(np.array(pil, dtype=np.float32) / 255.0)
        return np.stack(channels, axis=-1)

    def _apply_norm(self, image: np.ndarray) -> np.ndarray:
        """
        Apply per-channel (image - mean) / std normalization.

        Handles both:
            (H, W, C)    — single image from _preprocess()
            (N, H, W, C) — batch from PartitionExplainer's masker
        """
        # Channel count is always the last axis regardless of batch or not
        C = image.shape[-1]

        mean = (
            self.norm_mean
            if self.norm_mean.shape[0] == C
            else np.full(C, self.norm_mean.mean(), dtype=np.float32)
        )
        std = (
            self.norm_std
            if self.norm_std.shape[0] == C
            else np.full(C, self.norm_std.mean(), dtype=np.float32)
        )

        # Reshape to broadcast correctly over any leading dimensions
        # (H, W, C) -> mean shape (C,)   broadcasts fine
        # (N, H, W, C) -> mean shape (C,) also broadcasts fine
        # because numpy aligns from the right
        return (image - mean) / (std + 1e-8)

    def _to_tensor(self, image_HWC: np.ndarray) -> torch.Tensor:
        """Convert (H, W, C) numpy array to (1, C, H, W) tensor on device."""
        return (
            torch.tensor(image_HWC)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    # -------------------------------------------------------------------------
    # Model inference helpers
    # -------------------------------------------------------------------------

    def _get_probs(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run model forward pass and return a (num_classes,) probability array.
        Handles binary (1 neuron) and multi-class (N neurons) outputs.
        """
        with torch.no_grad():
            logits = self.model(tensor)
            if logits.shape[-1] == 1:
                # Binary classifier with single output neuron
                p = torch.sigmoid(logits).item()
                return np.array([1.0 - p, p], dtype=np.float32)
            # Multi-class classifier
            return torch.softmax(logits, dim=1).cpu().numpy()[0]

    def _predict_fn(self):
        """
        Return a numpy-in / numpy-out predict function for PartitionExplainer.

        PartitionExplainer masks regions of the image and calls this function
        repeatedly to see how predictions change. The masker always sends
        unnormalized [0, 1] images, so normalization is applied here before
        passing to the model.
        """
        def predict(X: np.ndarray) -> np.ndarray:
            X = np.array(X, dtype=np.float32)
            # Apply normalization here — masker works in [0,1] pixel space
            if self.normalize and self.norm_mean is not None:
                X = self._apply_norm(X)
            tensor = (
                torch.tensor(X)
                .permute(0, 3, 1, 2)
                .float()
                .to(self.device)
            )
            with torch.no_grad():
                logits = self.model(tensor)
                if logits.shape[-1] == 1:
                    p = torch.sigmoid(logits)
                    return torch.cat([1.0 - p, p], dim=-1).cpu().numpy()
                return torch.softmax(logits, dim=1).cpu().numpy()

        return predict

    # -------------------------------------------------------------------------
    # Background tensor (DeepExplainer only)
    # -------------------------------------------------------------------------

    def _build_background(self, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Build a (N, C, H, W) background reference tensor for DeepExplainer.

        DeepExplainer computes SHAP values relative to a baseline:
            shap(pixel) ≈ f(image) - f(background)

        Real training images as background produce more meaningful explanations
        than the black image fallback.
        """
        H, W, C = image_shape

        if self.background_data is not None:
            bg_arrays = [
                self._preprocess(img, for_display=False)
                for img in self.background_data
            ]
            bg = np.stack(bg_arrays, axis=0)    # (N, H, W, C)
        else:
            warnings.warn(
                "No background_data provided. Using a black image as background.\n"
                "For better SHAP accuracy, pass 50-100 real training images.",
                stacklevel=3,
            )
            bg = np.zeros((1, H, W, C), dtype=np.float32)
            # If normalizing, the black image must also be normalized
            if self.normalize and self.norm_mean is not None:
                bg = self._apply_norm(bg)

        return torch.tensor(bg).permute(0, 3, 1, 2).float().to(self.device)

    # -------------------------------------------------------------------------
    # SHAP value extraction — handles all SHAP version output formats
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_deep_shap(
        raw: Any,
        explain_class: int,
        image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Extract (H, W, C) SHAP values for one class from DeepExplainer output.

        SHAP has changed its output format across versions. This method handles
        all known formats so the code works regardless of which SHAP version
        the user has installed:

            list of arrays        old SHAP (<=0.39): one array per class
            ndarray (N, C, H, W)  single-class 4D output
            ndarray (N, K, C, H, W)  multi-class, classes on axis 1
            ndarray (N, C, H, W, K)  multi-class, classes on last axis
        """
        H_img, W_img, C_img = image_shape

        # Old SHAP: list where index = class
        if isinstance(raw, list):
            if explain_class >= len(raw):
                raise IndexError(
                    f"class_index {explain_class} out of range. "
                    f"Model has {len(raw)} classes (0 to {len(raw)-1})."
                )
            # raw[class] shape: (N, C, H, W) — take first sample
            return raw[explain_class][0].transpose(1, 2, 0)    # -> (H, W, C)

        if isinstance(raw, np.ndarray):
            if raw.ndim == 4:
                # (N, C, H, W) — single class output
                return raw[0].transpose(1, 2, 0)               # -> (H, W, C)

            if raw.ndim == 5:
                # Distinguish (N, K, C, H, W) vs (N, C, H, W, K)
                # by checking if the last axis is small (class count)
                # or large (spatial dimension)
                spatial_max = max(H_img, W_img)
                if raw.shape[-1] < spatial_max and raw.shape[-1] != C_img:
                    # (N, C, H, W, K) — classes on last axis
                    return raw[0, :, :, :, explain_class].transpose(1, 2, 0)
                else:
                    # (N, K, C, H, W) — classes on second axis
                    return raw[0, explain_class].transpose(1, 2, 0)

        raise TypeError(
            f"Unrecognized DeepExplainer output: type={type(raw)}, "
            f"shape={getattr(raw, 'shape', 'N/A')}.\n"
            "Please report this with: import shap; print(shap.__version__)"
        )

    # -------------------------------------------------------------------------
    # Build SHAP explainer — called lazily on first explain() call
    # -------------------------------------------------------------------------

    def _build_explainer(self, shape: Tuple[int, int, int]) -> None:
        """
        Initialize the underlying SHAP explainer for the given image shape.
        Called automatically on the first explain() call or when image shape
        changes (PartitionExplainer only — masker is shape-specific).
        """
        if self.explainer_type == "deep":
            bg = self._build_background(shape)
            self._explainer = shap.DeepExplainer(self.model, bg)
            print(
                f"[SHAPImageExplainer] DeepExplainer built. "
                f"Background shape: {tuple(bg.shape)}  Device: {self.device}"
            )
        else:
            self._explainer = shap.PartitionExplainer(
                self._predict_fn(),
                shap.maskers.Image("inpaint_telea", shape),
            )
            print(
                f"[SHAPImageExplainer] PartitionExplainer built. "
                f"Image shape: {shape}  Max evals: {self.max_evals}"
            )

    # -------------------------------------------------------------------------
    # Core public method: explain()
    # -------------------------------------------------------------------------

    def explain(
        self,
        image: Any,
        class_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute pixel-level SHAP values for a single image.

        Each pixel receives a SHAP value:
            Positive  -> pixel pushed the model score for the explained class HIGHER
            Negative  -> pixel pushed the model score for the explained class LOWER
            ~Zero     -> pixel had little or no influence on the prediction

        Parameters
        ----------
        image : str | PIL.Image | np.ndarray
            Input image in any supported format:
                str          -> file path
                PIL.Image    -> any mode
                np.ndarray   -> (H, W), (H, W, 1), or (H, W, C)

        class_index : int, optional
            Class to explain.
            None (default) -> explain the model's top predicted class.
            int            -> explain a specific class index.

        Returns
        -------
        dict with keys:
            shap_values     : np.ndarray (H, W, C)
                              Pixel-level SHAP values.
            predicted_class : int
                              Model's argmax prediction (highest probability class).
            explained_class : int
                              Class whose SHAP values were computed.
            class_name      : str
                              Human-readable label if class_names was provided,
                              otherwise str(predicted_class).
            probabilities   : np.ndarray
                              Full probability vector across all classes.
            confidence      : float
                              Probability of the predicted class (0.0 to 1.0).
        """
        # Preprocess image two ways:
        # img_model -> normalized, ready for model inference
        # img_disp  -> always [0, 1], used by masker and for visualization
        img_model = self._preprocess(image, for_display=False)
        img_disp  = self._preprocess(image, for_display=True)

        H, W, C = img_model.shape
        tensor  = self._to_tensor(img_model)

        # Get prediction
        probs           = self._get_probs(tensor)
        predicted_class = int(np.argmax(probs))
        explained_class = (
            class_index if class_index is not None else predicted_class
        )

        if not (0 <= explained_class < len(probs)):
            raise IndexError(
                f"class_index={explained_class} out of range. "
                f"Model has {len(probs)} classes (0 to {len(probs)-1})."
            )

        # Build or rebuild explainer if needed
        shape = (H, W, C)
        needs_build = (
            self._explainer is None
            or (self.explainer_type == "partition"
                and shape != self._last_image_shape)
        )
        if needs_build:
            self._build_explainer(shape)
        self._last_image_shape = shape

        # Compute SHAP values
        if self.explainer_type == "deep":
            raw         = self._explainer.shap_values(tensor)
            shap_values = self._extract_deep_shap(raw, explained_class, shape)

        else:   # partition
            # Pass img_disp (unnormalized) to the masker.
            # Normalization happens inside _predict_fn for every model call.
            raw = self._explainer(
                img_disp[np.newaxis],       # shape: (1, H, W, C)
                max_evals=self.max_evals,
                outputs=[explained_class],  # explain the correct target class
            )
            sv = raw.values                 # (1, H, W, C, 1) or (1, H, W, C)
            shap_values = sv[0, ..., 0] if sv.ndim == 5 else sv[0]

        # Resolve class name
        class_name = (
            self.class_names[predicted_class]
            if self.class_names and predicted_class < len(self.class_names)
            else str(predicted_class)
        )

        return {
            "shap_values":     shap_values.astype(np.float32),
            "predicted_class": predicted_class,
            "explained_class": explained_class,
            "class_name":      class_name,
            "probabilities":   probs,
            "confidence":      float(probs[predicted_class]),
        }

    # -------------------------------------------------------------------------
    # Visualization helpers (private)
    # -------------------------------------------------------------------------

    @staticmethod
    def _heatmap_2d(shap_values: np.ndarray) -> np.ndarray:
        """
        Collapse (H, W, C) SHAP values to a (H, W) signed heatmap in [-1, 1].
        Mean across channels preserves sign: positive = supports, negative = opposes.
        """
        h = shap_values.mean(axis=-1)
        m = np.abs(h).max()
        return (h / m).astype(np.float32) if m > 0 else h.astype(np.float32)

    @staticmethod
    def _build_masked(
        image: np.ndarray,
        shap_values: np.ndarray,
        percentile: int,
    ) -> np.ndarray:
        """
        Keep the top (100 - percentile)% most important pixels at full brightness.
        Everything else is blacked out for maximum contrast.
        """
        importance = np.abs(shap_values).mean(axis=-1)
        mask = (
            importance >= np.percentile(importance, percentile)
        )[:, :, np.newaxis]
        rgb = (
            np.repeat(image, 3, axis=-1)
            if image.shape[-1] == 1
            else image[..., :3].copy()
        )
        return np.clip(np.where(mask, rgb, 0.0), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _build_signed(
        image: np.ndarray,
        shap_values: np.ndarray,
        percentile: int,
    ) -> np.ndarray:
        """
        Color-code important pixels by SHAP sign:
            Green channel boosted -> positive SHAP (pixel supports prediction)
            Red channel boosted   -> negative SHAP (pixel contradicts prediction)
            Black                 -> below importance threshold
        """
        sv_mean = shap_values.mean(axis=-1)
        abs_sv  = np.abs(sv_mean)
        mask    = abs_sv >= np.percentile(abs_sv, percentile)
        rgb = (
            np.repeat(image, 3, axis=-1)
            if image.shape[-1] == 1
            else image[..., :3].copy()
        )
        out = np.zeros_like(rgb)

        pos = mask & (sv_mean > 0)
        neg = mask & (sv_mean < 0)

        out[pos]    = rgb[pos]
        out[pos, 1] = np.clip(out[pos, 1] * 1.5, 0.0, 1.0)    # boost green

        out[neg]    = rgb[neg]
        out[neg, 0] = np.clip(out[neg, 0] * 1.5, 0.0, 1.0)    # boost red

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _autosave(self, fig: plt.Figure, style: str, class_name: str) -> str:
        """
        Save a visualization figure to user_saves/ as a PNG.
        Filename format: shap_{style}_{class_name}_{YYYYMMDD_HHMMSS}.png
        Returns the full saved file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize class_name for use in filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in class_name)
        fname = os.path.join(_SAVE_DIR, f"shap_{style}_{safe_name}_{timestamp}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"[SHAPImageExplainer] Plot saved -> {fname}")
        return fname

    # -------------------------------------------------------------------------
    # Core public method: visualize()
    # -------------------------------------------------------------------------

    def visualize(
        self,
        result: Dict[str, Any],
        original_image: Any,
        style: str = "heatmap",
        percentile: int = 70,
        alpha: float = 0.6,
        cmap: str = "RdBu_r",
        figsize: Tuple[int, int] = (10, 4),
        show: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Visualize SHAP pixel-level explanations.

        The figure is automatically saved to user_saves/ as a PNG.
        Filename includes the class name and timestamp — nothing is ever overwritten.

        Styles
        ------
        "heatmap"  (default)
            Red/blue colormap overlaid on the original image.
            Red  = pixels that pushed the model score UP   (+SHAP)
            Blue = pixels that pushed the model score DOWN  (-SHAP)
            Best all-round view — shows both supporting and contradicting pixels.

        "masked"
            Only the top (100 - percentile)% most influential pixels are shown
            at full brightness. Everything else is blacked out.
            Best for confirming what exact region the model focused on.
            Example: did the CNN look at the tumor or the background?

        "signed"
            Green pixels = positive SHAP  (support the prediction)
            Red pixels   = negative SHAP  (contradict the prediction)
            Best for separating evidence for vs against the prediction.

        "shap_native"
            SHAP library's own built-in image_plot renderer.
            Good for publication-quality output.

        Parameters
        ----------
        result         : dict returned by explain()
        original_image : original image (str path, PIL.Image, or np.ndarray)
        style          : "heatmap" | "masked" | "signed" | "shap_native"
        percentile     : importance threshold for "masked" and "signed".
                         70 means top 30% most influential pixels are shown.
        alpha          : heatmap overlay transparency. 0=invisible, 1=opaque.
        cmap           : colormap for "heatmap" style. Default: "RdBu_r".
        figsize        : (width, height) in inches.
        show           : call plt.show() to display the figure interactively.

        Returns
        -------
        np.ndarray or None
            "heatmap"     -> (H, W)    normalized signed heatmap
            "masked"      -> (H, W, 3) masked RGB image
            "signed"      -> (H, W, 3) signed masked RGB image
            "shap_native" -> None
        """
        valid_styles = ("heatmap", "masked", "signed", "shap_native")
        if style not in valid_styles:
            raise ValueError(
                f"style must be one of {valid_styles}. Got '{style}'."
            )

        sv       = result["shap_values"]
        cls_name = result["class_name"]
        conf     = result["confidence"]
        exp_cls  = result["explained_class"]
        pred_cls = result["predicted_class"]

        # Always use the display image (un-normalized, [0, 1]) for visualization
        img  = self._preprocess(original_image, for_display=True)
        disp = img[:, :, 0] if img.shape[-1] == 1 else img[..., :3]
        dcm  = "gray" if img.shape[-1] == 1 else None

        if sv.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"Shape mismatch: shap_values {sv.shape[:2]} vs "
                f"image {img.shape[:2]}. "
                "Make sure you use the same target_size in explain() and visualize()."
            )

        title = f"Predicted: {cls_name}  |  Confidence: {conf:.1%}"
        if exp_cls != pred_cls:
            title += f"  (explaining class {exp_cls})"

        # ── shap_native ───────────────────────────────────────────────────────
        if style == "shap_native":
            shap.image_plot(sv[np.newaxis], img[np.newaxis], show=False)
            fig = plt.gcf()
            fig.suptitle(title, fontsize=10, y=1.01)
            self._autosave(fig, style, cls_name)
            plt.show() if show else plt.close(fig)
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(disp, cmap=dcm)
        axes[0].set_title("Original Image", fontsize=10)
        axes[0].axis("off")

        # ── heatmap ───────────────────────────────────────────────────────────
        if style == "heatmap":
            H, W = disp.shape[:2]
            hm   = self._heatmap_2d(sv)
            axes[1].imshow(disp, cmap=dcm)
            im = axes[1].imshow(
                hm,
                cmap=cmap,
                alpha=alpha,
                vmin=-1, vmax=1,
                extent=[0, W, H, 0],
                aspect="auto",
                origin="upper",
            )
            plt.colorbar(
                im, ax=axes[1],
                fraction=0.046, pad=0.04,
                label="SHAP  (+red / -blue)",
            )
            axes[1].set_title(f"SHAP Heatmap\n{title}", fontsize=9)
            axes[1].axis("off")
            plt.tight_layout()
            self._autosave(fig, style, cls_name)
            plt.show() if show else plt.close(fig)
            return hm

        # ── masked ────────────────────────────────────────────────────────────
        if style == "masked":
            out = self._build_masked(img, sv, percentile)
            axes[1].imshow(out)
            axes[1].set_title(
                f"Top {100 - percentile}% Important Pixels\n{title}",
                fontsize=9,
            )
            axes[1].axis("off")
            plt.tight_layout()
            self._autosave(fig, style, cls_name)
            plt.show() if show else plt.close(fig)
            return out

        # ── signed ────────────────────────────────────────────────────────────
        if style == "signed":
            out = self._build_signed(img, sv, percentile)
            gp  = mpatches.Patch(color="green", label="Supports prediction  (+SHAP)")
            rp  = mpatches.Patch(color="red",   label="Contradicts prediction  (-SHAP)")
            axes[1].imshow(out)
            axes[1].legend(handles=[gp, rp], loc="lower right", fontsize=8)
            axes[1].set_title(f"Signed SHAP\n{title}", fontsize=9)
            axes[1].axis("off")
            plt.tight_layout()
            self._autosave(fig, style, cls_name)
            plt.show() if show else plt.close(fig)
            return out