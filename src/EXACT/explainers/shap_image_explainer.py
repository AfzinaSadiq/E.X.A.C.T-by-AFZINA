import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt


class ShapExplainer_Image:
    """
    SHAP Image Explainer for EXACT Library  (PyTorch-specific)

    Responsibilities:
        - Generate SHAP explanations for ANY PyTorch image classification model
        - Return per-superpixel SHAP attribution values
        - Automatically display a heatmap after every explain() call
        - Optionally save the heatmap to disk (save_png=True)
        - Provide visualization utilities (heatmap_plot, summary_plot)

    Explainer used:
        PartitionExplainer — hierarchical Owen values approach.
        Specifically designed for structured inputs like images and text.
        Works with ALL PyTorch image model types:
            - Custom CNNs
            - Pretrained models (ResNet, VGG, EfficientNet, etc.)
            - Any nn.Module that accepts (B, C, H, W) float tensors

    Why PartitionExplainer over KernelExplainer for images?
        KernelExplainer treats all superpixels as independent features and
        uses random coalitions to approximate Shapley values. It was designed
        for tabular data and ignores spatial structure.

        PartitionExplainer uses Owen values — a hierarchical variant of
        Shapley values that respects the spatial structure of images by
        recursively partitioning regions in a tree. This makes it:
            - More accurate for spatially structured inputs
            - Faster — tree-based recursion instead of random sampling
            - The approach recommended by the shap library for images

    Supports:
        - RGB images   (C=3)
        - Grayscale images  (C=1)
        - Binary classification  (2-class output)
        - Multi-class classification  (N-class output)

    How it works:
        1. Input image is wrapped with shap.maskers.Image which handles
           superpixel segmentation and masking internally.
        2. A prediction wrapper converts masked numpy images to tensors
           and runs them through the PyTorch model.
        3. PartitionExplainer recursively evaluates which image regions
           matter most using a hierarchical partition tree.
        4. SHAP values per pixel are returned and used for visualization.
        5. A heatmap is automatically displayed after explain() completes.
        6. If save_png=True, the heatmap is saved to the user_saves/ folder.

    Notes:
        - Model is automatically set to eval() mode
        - Input image must be a numpy array: (H, W, C) or (H, W) for grayscale
        - background_color controls what masked regions look like
    """

    def __init__(
        self,
        model,
        class_names=None,
        n_segments=50,
        background_color="mean",
        max_evals=500,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch image classification model.
            Must accept input of shape (batch_size, C, H, W) as FloatTensor
            and return logits of shape (batch_size, num_classes).

        class_names : list[str], optional
            Names of output classes (used in visualization only).
            If None, auto-labeled as Class_0, Class_1, ...

        n_segments : int
            Number of superpixels to segment the image into.
            More segments = finer-grained explanation but slower.
            Recommended: 30–100. Default: 50.

        background_color : str or float or list
            What color to use for masked (absent) superpixel regions.
            Options:
                'mean'  → mean color of the input image  (default, recommended)
                'black' → fill with zeros
                'white' → fill with ones
                float   → fill with this value  (e.g. 0.5)
                list    → per-channel fill  (e.g. [0.485, 0.456, 0.406]
                          for ImageNet mean)
            Default: 'mean'

        max_evals : int
            Maximum number of model evaluations PartitionExplainer performs.
            Higher → more accurate but slower.
            Recommended: 200–1000. Default: 500.
        """
        self.model            = model
        self.class_names      = class_names
        self.n_segments       = n_segments
        self.background_color = background_color
        self.max_evals        = max_evals

        # Resolve device from model
        self.device = next(model.parameters()).device

        # Always eval mode
        self.model.eval()

    # ──────────────────────────────────────────────────────────────────────
    # Internal: predict proba  (self-contained, image-specific)
    # ──────────────────────────────────────────────────────────────────────

    def _predict_proba(self, images_np):
        """
        Prediction wrapper for PartitionExplainer.

        PartitionExplainer passes masked images as numpy arrays in HWC format.
        This function converts them to (B, C, H, W) tensors, runs the model,
        and returns softmax probabilities.

        Self-contained because image models need float32 tensors in CHW format
        — different from tabular (float features) and text (long token IDs).

        Parameters
        ----------
        images_np : np.ndarray
            Shape: (batch_size, H, W, C) or (batch_size, H, W)
            dtype: float32, values in [0, 1] or model's expected range.

        Returns
        -------
        np.ndarray   shape: (batch_size, num_classes)  — softmax probabilities
        """
        self.model.eval()

        # Convert HWC → CHW for PyTorch
        if images_np.ndim == 4:
            # (B, H, W, C) → (B, C, H, W)
            batch = images_np.transpose(0, 3, 1, 2)
        elif images_np.ndim == 3:
            # (B, H, W) grayscale → (B, 1, H, W)
            batch = images_np[:, np.newaxis, :, :]
        else:
            raise ValueError(
                f"[ShapExplainer_Image] Unexpected image batch shape: {images_np.shape}. "
                f"Expected (B, H, W, C) or (B, H, W)."
            )

        input_tensor = torch.tensor(
            batch.astype(np.float32), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)              # (B, num_classes)

            if logits.ndim == 2 and logits.shape[-1] == 1:
                # Binary classification with single output neuron
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()                         # (B, num_classes)

    # ──────────────────────────────────────────────────────────────────────
    # Internal: compute background fill value
    # ──────────────────────────────────────────────────────────────────────

    def _get_background_value(self, image_np):
        """
        Compute the scalar or per-channel fill value for masked regions.

        Parameters
        ----------
        image_np : np.ndarray   shape: (H, W, C) or (H, W)

        Returns
        -------
        float or np.ndarray   — fill value passed to shap.maskers.Image
        """
        if self.background_color == "mean":
            return image_np.mean()

        elif self.background_color == "black":
            return 0.0

        elif self.background_color == "white":
            return 1.0

        elif isinstance(self.background_color, (int, float)):
            return float(self.background_color)

        elif isinstance(self.background_color, (list, np.ndarray)):
            return float(np.mean(self.background_color))

        else:
            raise ValueError(
                f"[ShapExplainer_Image] Invalid background_color: "
                f"'{self.background_color}'. "
                f"Choose from: 'mean', 'black', 'white', float, or list."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Internal: render and optionally save the heatmap figure
    # ──────────────────────────────────────────────────────────────────────

    def _render_heatmap(self, explanation, class_index, save_png, save_dir):
        """
        Build the 3-panel heatmap figure, display it, and optionally save it.

        Called automatically at the end of explain() so the user always sees
        the heatmap immediately without needing to call heatmap_plot() manually.

        Parameters
        ----------
        explanation : dict   — output of explain() (built so far)
        class_index : int    — which class to visualize
        save_png    : bool   — whether to save the figure to disk
        save_path   : str    — full file path to save to (used when save_png=True)
        """
        shap_values  = explanation["shap_values"].values   # (1, H, W, C, num_classes)
        image_np     = explanation["image_np"]
        is_grayscale = explanation["is_grayscale"]

        # Extract for this class — (H, W, C)
        values_hwc = shap_values[0, :, :, :, class_index]

        # Average over channels → (H, W)
        pixel_map  = values_hwc.mean(axis=-1)

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original image
        axes[0].set_title("Original Image", fontsize=12)
        if is_grayscale:
            axes[0].imshow(image_np, cmap="gray")
        else:
            axes[0].imshow(image_np)
        axes[0].axis("off")

        # Panel 2: SHAP heatmap
        axes[1].set_title(f"SHAP Heatmap  [{class_label}]", fontsize=12)
        vmax = np.abs(pixel_map).max()
        im   = axes[1].imshow(
            pixel_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        axes[2].set_title(f"Overlay  [{class_label}]", fontsize=12)
        if is_grayscale:
            axes[2].imshow(image_np, cmap="gray")
        else:
            axes[2].imshow(image_np)
        axes[2].imshow(
            pixel_map, cmap="RdBu_r", alpha=0.5, vmin=-vmax, vmax=vmax
        )
        axes[2].axis("off")

        plt.suptitle(
            f"SHAP Image Explanation  —  Class: {class_label}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"shap_heatmap_{class_label}.png")

        # ── Save to disk if requested ──
        if save_png:
            save_path = os.path.join(save_dir, f"shap_heatmap_{class_label}.png")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[ShapExplainer_Image] Heatmap saved to: {save_path}")

        plt.show()
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    # Core explanation logic
    # ──────────────────────────────────────────────────────────────────────

    def explain(self, image, class_index=0, save_png=False, save_dir="user_saves"):
        """
        Generate SHAP values for a single image using PartitionExplainer.

        After computing SHAP values, a heatmap is automatically displayed
        showing the original image, the SHAP heatmap, and an overlay.

        If save_png=True, the heatmap is saved to the save_dir folder as a
        .png file named after the class being explained.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to explain.

            Accepted formats:
                np.ndarray  (H, W, C)   — RGB,        values in [0, 1]
                np.ndarray  (H, W)      — Grayscale,   values in [0, 1]
                np.ndarray  (C, H, W)   — CHW format,  auto-converted to HWC
                torch.Tensor (C, H, W)  — auto-converted to HWC numpy

        class_index : int
            Which class to visualize in the auto-generated heatmap.
            Default: 0.

        save_png : bool
            If True, saves the heatmap as a .png file inside save_dir.
            File is named: shap_heatmap_class_<class_label>.png
            Default: False.

        save_dir : str
            Directory where the heatmap .png is saved when save_png=True.
            Created automatically if it does not exist.
            Default: 'user_saves'

        Returns
        -------
        explanation : dict with keys:
            'shap_values'    : shap.Explanation object
                               .values shape: (1, H, W, C, num_classes)
                               Per-pixel SHAP values — PartitionExplainer
                               returns pixel-level attributions directly.
            'expected_value' : np.ndarray
                               Base model output (before any features present).
            'image_np'       : np.ndarray  shape: (H, W, C) or (H, W)
                               Original image in HWC format.
            'image_hwc'      : np.ndarray  shape: (H, W, C)
                               Image with grayscale expanded to (H, W, 1).
            'is_grayscale'   : bool
        """
        # ── Step 1: Normalize input to HWC numpy ──
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # CHW → HWC
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)

        image_np     = image.astype(np.float32)
        is_grayscale = image_np.ndim == 2

        # PartitionExplainer needs HWC — expand grayscale to (H, W, 1)
        if is_grayscale:
            image_hwc = image_np[:, :, np.newaxis]         # (H, W, 1)
        else:
            image_hwc = image_np                           # (H, W, C)

        # ── Step 2: Compute background fill value ──
        bg_value = self._get_background_value(image_np)

        # ── Step 3: Build shap.maskers.Image ──
        masker = shap.maskers.Image(
            mask_value = bg_value,
            shape      = image_hwc.shape,                  # (H, W, C)
        )

        # ── Step 4: Build PartitionExplainer ──
        explainer = shap.explainers.Partition(
            self._predict_proba,
            masker,
        )

        # ── Step 5: Compute SHAP values ──
        shap_values = explainer(
            image_hwc[np.newaxis, :, :, :],                # (1, H, W, C)
            max_evals  = self.max_evals,
            batch_size = 50,
        )

        # ── Step 6: Build explanation dict ──
        explanation = {
            "shap_values"    : shap_values,
            "expected_value" : shap_values.base_values,
            "image_np"       : image_np,
            "image_hwc"      : image_hwc,
            "is_grayscale"   : is_grayscale,
        }

        # ── Step 7: Auto-display heatmap (always) ──
        # Build save path if needed
        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"class_{class_index}"
        )
        save_path = os.path.join(
            save_dir, f"shap_heatmap_{class_label}.png"
        )

        self._render_heatmap(
            explanation = explanation,
            class_index = class_index,
            save_png    = save_png,
            save_dir   = save_dir,
        )

        return explanation

    # ──────────────────────────────────────────────────────────────────────
    # Raw explanation data  (mirrors LIME's get_explanation_data pattern)
    # ──────────────────────────────────────────────────────────────────────

    def get_explanation_data(self, explanation, class_index=0, num_segments=10):
        """
        Extract top pixel regions by SHAP value magnitude.

        Since PartitionExplainer returns pixel-level SHAP values directly,
        this method flattens and ranks pixels by absolute SHAP value.

        Parameters
        ----------
        explanation  : dict  — direct output of explain()
        class_index  : int   — which class to extract. Default: 0.
        num_segments : int   — how many top regions to return. Default: 10.

        Returns
        -------
        list of (str, float)
            [('pixel_(r,c)', shap_value), ...] sorted by |shap_value| descending.
        """
        shap_values = explanation["shap_values"].values    # (1, H, W, C, num_classes)

        # Extract for this class — shape: (H, W, C)
        values_hwc = shap_values[0, :, :, :, class_index]

        # Average over channels to get (H, W)
        values_hw  = values_hwc.mean(axis=-1)

        # Flatten and rank
        H, W    = values_hw.shape
        flat    = values_hw.flatten()
        indices = np.argsort(np.abs(flat))[::-1][:num_segments]

        result = []
        for idx in indices:
            r, c = divmod(idx, W)
            result.append((f"pixel_({r},{c})", float(flat[idx])))

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Console visualization
    # ──────────────────────────────────────────────────────────────────────

    def visualize(self, explanation, class_index=0, num_segments=10):
        """
        Print SHAP pixel attributions in readable console format.

        Parameters
        ----------
        explanation  : dict  — direct output of explain()
        class_index  : int   — which class to display. Default: 0.
        num_segments : int   — how many top pixels to show. Default: 10.

        Returns
        -------
        segment_scores : list of (str, float)
        """
        segment_scores = self.get_explanation_data(
            explanation,
            class_index  = class_index,
            num_segments = num_segments,
        )

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        print(f"\nSHAP Image Explanation  [Class: {class_label}]")
        print("-" * 48)
        for region, score in segment_scores:
            sign = "+" if score >= 0 else "-"
            bar  = "|" * min(int(abs(score) * 80), 20)
            print(f"  {region:<30s}  {sign}  {abs(score):.6f}  {bar}")
        print("-" * 48)

        return segment_scores

    # ──────────────────────────────────────────────────────────────────────
    # SHAP plots
    # ──────────────────────────────────────────────────────────────────────

    def heatmap_plot(self, explanation, class_index=0, save_png=False, save_dir="user_saves"):
        """
        Overlay SHAP heatmap on the original image.

        Red regions push the prediction toward the class.
        Blue regions push the prediction away from the class.

        Shows 3 panels: original image | SHAP heatmap | overlay.

        Can also be called manually after explain() if the user wants to
        visualize a different class_index without re-running explain().

        Parameters
        ----------
        explanation : dict  — output of explain()
        class_index : int   — which class to visualize. Default: 0.
        save_png    : bool  — if True, saves the figure to save_dir. Default: False.
        save_dir    : str   — directory to save the figure. Default: 'user_saves'.
        """
        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"class_{class_index}"
        )
        save_path = os.path.join(
            save_dir, f"shap_heatmap_{class_label}.png"
        )

        self._render_heatmap(
            explanation = explanation,
            class_index = class_index,
            save_png    = save_png,
            save_dir   = save_dir,
        )

    def summary_plot(self, explanation, class_index=0, save_png=False, save_dir="user_saves"):
        """
        SHAP image summary plot — shows pixel attributions on the image.
        Uses heatmap_plot internally since shap.image_plot does not
        support single-channel (grayscale) images reliably.

        Parameters
        ----------
        explanation : dict  — output of explain()
        class_index : int   — which class to visualize. Default: 0.
        save_png    : bool  — if True, saves the figure to save_dir. Default: False.
        save_dir    : str   — directory to save the figure. Default: 'user_saves'.
        """
        self.heatmap_plot(
            explanation = explanation,
            class_index = class_index,
            save_png    = save_png,
            save_dir    = save_dir,
        )