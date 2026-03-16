# explainers/lime_image_explainer.py

"""
LimeExplainer
=============
LIME-based saliency heatmap explainer for image classifiers.

Returns a standardised result dict — identical contract to all CAM-based
explainers — so its output can be passed directly into HeatmapComparator
alongside GradCAM, ViTGradCAM, DFF, etc.

Result dict keys
----------------
  'heatmap'       : np.ndarray (H, W) float32 in [0, 1]
                    The raw saliency heatmap at original image resolution.
                    HeatmapComparator accepts this key as an alias for 'cam'.
  'visualization' : np.ndarray (H, W, 3) float32 in [0, 1]
                    Jet-coloured heatmap blended over the original image.
  'filepath'      : Path or None
                    Path where the overlay was saved (None if save_png=False).
  'explanation'   : lime.explanation.ImageExplanation
                    The raw LIME explanation object, in case you need it.
  'boundary'      : np.ndarray (H, W, 3) or None
                    Boundary-marked visualisation (None if boundary_marking=False).
  'boundary_filepath' : Path or None

Usage
-----
    from EXACT.explainers.lime_explainer import LimeExplainer

    exp = LimeExplainer(model, num_samples=1000, target_size=(128, 128))

    result = exp.explain(image="path/to/img.jpg", save_png=True)

    # Use with HeatmapComparator:
    from EXACT.comparators import HeatmapComparator
    cmp = HeatmapComparator(model)
    results = cmp.compare(
        entries={
            "GradCAM": (gradcam_result, gradcam_exp, {"method": "gradcam"}),
            "LIME":    (lime_result,    lime_exp,    {}),
        },
        input_tensor=input_tensor,
        input_image=img_np,
    )
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from EXACT.utils import predict_proba_fn


class LimeExplainer_Image:
    """
    LIME heatmap explainer for image classifiers.

    Wraps lime.lime_image.LimeImageExplainer and returns a standardised
    result dict compatible with HeatmapComparator and the rest of EXACT.

    Parameters
    ----------
    model : torch.nn.Module
        The classifier to explain.
    num_samples : int
        Number of perturbed samples LIME generates. Higher = more accurate
        but slower. Default 1000.
    target_size : tuple (W, H)
        Spatial size images are resized to before being fed to the model.
        Must match what the model expects. Default (224, 224).
    smoothing_sigma : float
        Gaussian smoothing applied to the raw LIME heatmap to reduce
        superpixel blockiness. Default 2. Set to 0 to disable.
    random_state : int
        Seed for reproducibility. Default 42.
    save_dir : str
        Directory for saved outputs. Default 'user_saves/lime_saves'.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_samples: int = 1000,
        target_size: tuple = (224, 224),
        smoothing_sigma: float = 2.0,
        random_state: int = 42,
        save_dir: str = "user_saves/lime_saves",
    ):
        try:
            from lime import lime_image as _lime_image
        except ImportError:
            raise ImportError(
                "lime is required for LimeExplainer. "
                "Install with: pip install lime"
            )

        self.model           = model
        self.num_samples     = num_samples
        self.target_size     = target_size
        self.smoothing_sigma = smoothing_sigma
        self.random_state    = random_state
        self.save_dir        = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lime           = _lime_image.LimeImageExplainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        image: Union[str, np.ndarray, Image.Image],
        top_labels: int = 1,
        label: Optional[int] = None,
        boundary_marking: bool = False,
        num_features: int = 5,
        positive_only: bool = True,
        hide_rest: bool = False,
        alpha: float = 0.5,
        save_png: bool = False,
        tag: str = "",
    ) -> dict:
        """
        Generate a LIME saliency heatmap for the given image.

        Parameters
        ----------
        image : str, np.ndarray, or PIL.Image
            Input image. Accepts a file path, a uint8/float32 numpy array,
            or a PIL Image. Any size is accepted — it is resized to
            target_size for the model and back to original size for output.
        top_labels : int
            Number of top predicted classes LIME analyses. Default 1.
        label : int, optional
            Class index to explain. If None, uses the top predicted class.
        boundary_marking : bool
            Whether to also produce a LIME boundary-marked visualisation.
            Default False.
        num_features : int
            Number of superpixel regions shown in the boundary visualisation.
            Default 5.
        positive_only : bool
            In boundary mode, show only positively-contributing regions.
            Default True.
        hide_rest : bool
            In boundary mode, grey out non-highlighted regions. Default False.
        alpha : float
            Heatmap blend strength over the original image. Default 0.5.
        save_png : bool
            Whether to save output images. Default False.
        tag : str
            Optional string appended to saved filenames. Default ''.

        Returns
        -------
        dict with keys:
            'heatmap'            : np.ndarray (H, W) float32 in [0, 1]
            'visualization'      : np.ndarray (H, W, 3) float32 in [0, 1]
            'filepath'           : Path or None
            'explanation'        : lime ImageExplanation object
            'boundary'           : np.ndarray (H, W, 3) or None
            'boundary_filepath'  : Path or None
        """
        # ── load and prepare image ────────────────────────────────────
        original_pil  = self._load_image(image)
        original_size = original_pil.size                          # (W, H)
        original_np   = np.array(original_pil, dtype=np.float32) / 255.0  # [0,1]

        lime_input = self._resize_for_model(original_pil)         # (H, W, 3) [0,1]

        # ── LIME explanation ──────────────────────────────────────────
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        explanation = self._lime.explain_instance(
            image=lime_input,
            classifier_fn=self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=self.num_samples,
        )

        # ── heatmap ───────────────────────────────────────────────────
        heatmap_small = self._build_heatmap(explanation, label)   # (H_model, W_model)

        # Resize heatmap back to original image resolution
        heatmap_full = np.array(
            Image.fromarray((heatmap_small * 255).astype(np.uint8))
                 .resize(original_size, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0                                                  # (H_orig, W_orig)

        visualization = self._blend(original_np, heatmap_full, alpha)

        # ── save overlay ──────────────────────────────────────────────
        filepath = None
        if save_png:
            suffix   = f"_{tag}" if tag else f"_{int(time.time())}"
            filepath = self.save_dir / f"lime_heatmap{suffix}.png"
            Image.fromarray((visualization * 255).astype(np.uint8)).save(filepath)
            print(f"✓ Saved: {filepath}")

        # ── boundary visualisation (optional) ────────────────────────
        boundary           = None
        boundary_filepath  = None
        if boundary_marking:
            boundary = self._boundary_vis(
                explanation, label, positive_only, num_features, hide_rest
            )
            if save_png:
                suffix            = f"_{tag}" if tag else f"_{int(time.time())}"
                boundary_filepath = self.save_dir / f"lime_boundary{suffix}.png"
                Image.fromarray((boundary * 255).astype(np.uint8)).save(boundary_filepath)
                print(f"✓ Saved: {boundary_filepath}")

        return {
            "heatmap":           heatmap_full,   # (H, W)   — consumed by HeatmapComparator
            "visualization":     visualization,  # (H, W, 3)
            "filepath":          filepath,
            "explanation":       explanation,    # raw LIME object if needed
            "boundary":          boundary,
            "boundary_filepath": boundary_filepath,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image) -> Image.Image:
        import torch as _torch
        # torch.Tensor — comes from stability re-runs in HeatmapComparator
        if isinstance(image, _torch.Tensor):
            img = image[0] if image.ndim == 4 else image   # (1,C,H,W) or (C,H,W)
            img = img.cpu().numpy()
            if img.shape[0] == 3:                           # (C,H,W) → (H,W,C)
                img = np.transpose(img, (1, 2, 0))
            if img.min() < 0 or img.max() > 1.0:           # un-normalise if needed
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            image = (img * 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _resize_for_model(self, image: Image.Image) -> np.ndarray:
        """Resize to target_size and return float32 (H, W, 3) in [0, 1]."""
        return np.array(image.resize(self.target_size), dtype=np.float32) / 255.0

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        LIME-compatible predict function: (N, H, W, 3) -> (N, num_classes).

        LIME generates perturbed images at whatever spatial size it was given
        (lime_input size), but the internal LIME segmentation pipeline can
        produce batches at a slightly different resolution than target_size.
        We explicitly resize every image in the batch to self.target_size
        here so the model always receives the exact dimensions it expects.
        This is the single authoritative resize point -- predict_proba_fn
        does no resizing of its own.
        """
        images = np.array(images)
        tw, th = self.target_size   # target_size is (W, H) — PIL convention

        # Only resize if the batch spatial dims don't already match target_size
        if images.shape[1] != th or images.shape[2] != tw:
            resized = np.stack([
                np.array(
                    Image.fromarray((img * 255).astype(np.uint8)).resize(
                        self.target_size, Image.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
                for img in images
            ])
        else:
            resized = images.astype(np.float32)

        return predict_proba_fn.predict_proba(resized, model=self.model)

    def _build_heatmap(self, explanation, label=None) -> np.ndarray:
        """
        Convert LIME superpixel weights → smooth float32 (H, W) heatmap in [0, 1].
        """
        if label is None:
            label = explanation.top_labels[0]

        segments = explanation.segments
        weights  = dict(explanation.local_exp[label])

        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for sp, w in weights.items():
            heatmap[segments == sp] = w

        # Normalise before smoothing so sigma is scale-independent
        mn, mx = heatmap.min(), heatmap.max()
        heatmap = (heatmap - mn) / (mx - mn + 1e-8)

        if self.smoothing_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.smoothing_sigma)
            # Re-normalise after smoothing (gaussian_filter can shift the range)
            mn, mx = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)

        return heatmap.astype(np.float32)

    @staticmethod
    def _blend(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        """Blend a jet-coloured heatmap over a [0,1] RGB image."""
        import matplotlib.cm as cm
        heatmap_rgb = cm.jet(heatmap)[..., :3].astype(np.float32)
        return np.clip(image * (1 - alpha) + heatmap_rgb * alpha, 0, 1)

    def _boundary_vis(
        self,
        explanation,
        label,
        positive_only: bool,
        num_features: int,
        hide_rest: bool,
    ) -> np.ndarray:
        from skimage.segmentation import mark_boundaries
        if label is None:
            label = explanation.top_labels[0]
        lime_img, mask = explanation.get_image_and_mask(
            label=label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest,
        )
        return mark_boundaries(lime_img, mask).astype(np.float32)