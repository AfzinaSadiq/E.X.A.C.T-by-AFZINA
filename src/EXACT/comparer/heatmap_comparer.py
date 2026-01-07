import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from PIL import Image


class HeatmapComparer:
    """
    Compares XAI explanations (heatmaps) from different explainers.
    Uses SSIM (Structural Similarity) and Cosine Similarity metrics.

    Accepts heatmaps directly as numpy arrays or loads from image files.
    Automatically handles preprocessing of LIME explanations.
    """

    def __init__(self):
        pass

    @staticmethod
    def _load_heatmap(heatmap_input):
        """
        Load heatmap from various input types.

        Args:
            heatmap_input: numpy array, PIL Image, image path (str), or LIME explanation object

        Returns:
            numpy array: Normalized heatmap as 2D array
        """
        # If it's a string, assume it's a file path
        if isinstance(heatmap_input, str):
            img = Image.open(heatmap_input)
            heatmap = np.array(img)
            if len(heatmap.shape) == 3:
                heatmap = np.mean(heatmap, axis=2)
            return heatmap.astype(np.float32) / 255.0

        # If it's a PIL Image
        if isinstance(heatmap_input, Image.Image):
            heatmap = np.array(heatmap_input)
            if len(heatmap.shape) == 3:
                heatmap = np.mean(heatmap, axis=2)
            return heatmap.astype(np.float32) / 255.0

        # If it's a LIME explanation object
        if hasattr(heatmap_input, "get_image_and_mask"):
            # This is a LIME explanation
            try:
                label = heatmap_input.top_labels[0]
                lime_viz, _ = heatmap_input.get_image_and_mask(
                    label=label, positive_only=True, num_features=10, hide_rest=False
                )
                # Convert LIME visualization to grayscale
                if len(lime_viz.shape) == 3:
                    heatmap = np.mean(lime_viz, axis=2)
                else:
                    heatmap = lime_viz
                return heatmap.astype(np.float32)
            except:
                raise ValueError("Could not extract heatmap from LIME explanation")

        # If it's already a numpy array
        if isinstance(heatmap_input, np.ndarray):
            return np.array(heatmap_input, dtype=np.float32)

        raise TypeError(f"Unsupported heatmap input type: {type(heatmap_input)}")

    @staticmethod
    def normalize_heatmap(heatmap):
        """
        Normalize heatmap to [0, 1] range.
        """
        heatmap = np.array(heatmap, dtype=np.float32)
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)

        if heatmap_max - heatmap_min == 0:
            return np.zeros_like(heatmap)

        normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        return normalized

    @staticmethod
    def _resize_to_match(heatmap1, heatmap2):
        """
        Resize heatmaps to the same shape if they differ.
        Uses the smaller dimensions to avoid interpolation artifacts.
        """
        if heatmap1.shape == heatmap2.shape:
            return heatmap1, heatmap2

        from PIL import Image

        h1, w1 = heatmap1.shape[:2]
        h2, w2 = heatmap2.shape[:2]

        target_h = min(h1, h2)
        target_w = min(w1, w2)

        hm1_resized = (
            np.array(
                Image.fromarray((heatmap1 * 255).astype(np.uint8)).resize(
                    (target_w, target_h)
                )
            )
            / 255.0
        )

        hm2_resized = (
            np.array(
                Image.fromarray((heatmap2 * 255).astype(np.uint8)).resize(
                    (target_w, target_h)
                )
            )
            / 255.0
        )

        return hm1_resized, hm2_resized

    def compute_ssim(self, heatmap1, heatmap2):
        """
        Compute Structural Similarity Index (SSIM) between two heatmaps.
        Measures perceptual similarity considering spatial structure.

        Returns:
            float: SSIM score in range [-1, 1]. Higher is better (1 = identical).
        """
        hm1 = self.normalize_heatmap(heatmap1)
        hm2 = self.normalize_heatmap(heatmap2)

        hm1, hm2 = self._resize_to_match(hm1, hm2)

        # SSIM requires 2D or 3D input
        if len(hm1.shape) == 2:
            return ssim(hm1, hm2, data_range=1.0)
        else:
            # For multi-channel, compute SSIM for each channel and average
            return ssim(hm1, hm2, data_range=1.0, channel_axis=-1)

    def compute_cosine_similarity(self, heatmap1, heatmap2):
        """
        Compute Cosine Similarity between two heatmaps.
        Treats heatmaps as vectors and measures directional similarity.

        Returns:
            float: Cosine similarity in range [-1, 1]. Higher is better (1 = identical direction).
        """
        hm1 = self.normalize_heatmap(heatmap1)
        hm2 = self.normalize_heatmap(heatmap2)

        hm1, hm2 = self._resize_to_match(hm1, hm2)

        # Flatten to 1D vectors
        vec1 = hm1.flatten()
        vec2 = hm2.flatten()

        # Compute cosine similarity (returns distance, so convert to similarity)
        cosine_dist = cosine(vec1, vec2)
        cosine_sim = 1 - cosine_dist

        return cosine_sim

    def compare(
        self,
        heatmap1,
        heatmap2,
        explainer1_name="Explainer1",
        explainer2_name="Explainer2",
    ):
        """
        Compare two heatmaps using both SSIM and Cosine Similarity.

        Args:
            heatmap1: numpy array, PIL Image, file path (str), or LIME explanation
            heatmap2: numpy array, PIL Image, file path (str), or LIME explanation
            explainer1_name: Name of first explainer (for reporting)
            explainer2_name: Name of second explainer (for reporting)

        Returns:
            dict: Dictionary containing:
                - 'ssim': SSIM score
                - 'cosine_similarity': Cosine similarity score
                - 'interpretation': Human-readable interpretation
        """
        # Load heatmaps from various input types
        hm1 = self._load_heatmap(heatmap1)
        hm2 = self._load_heatmap(heatmap2)

        ssim_score = self.compute_ssim(hm1, hm2)
        cosine_score = self.compute_cosine_similarity(hm1, hm2)

        # Generate interpretation
        avg_score = (ssim_score + cosine_score) / 2
        if avg_score > 0.8:
            interpretation = "VERY HIGH - Explanations are highly similar"
        elif avg_score > 0.6:
            interpretation = "HIGH - Explanations are moderately similar"
        elif avg_score > 0.4:
            interpretation = "MODERATE - Explanations show some agreement"
        elif avg_score > 0.2:
            interpretation = "LOW - Explanations have limited agreement"
        else:
            interpretation = "VERY LOW - Explanations are dissimilar"

        return {
            "ssim": float(ssim_score),
            "cosine_similarity": float(cosine_score),
            "average_similarity": float(avg_score),
            "interpretation": interpretation,
            "explainer1": explainer1_name,
            "explainer2": explainer2_name,
        }

    def visualize_comparison(
        self,
        heatmap1,
        heatmap2,
        explainer1_name="Explainer1",
        explainer2_name="Explainer2",
        show=True,
        save_path=None,
    ):
        """
        Visualize comparison of two heatmaps side-by-side with metrics.

        Args:
            heatmap1: numpy array, PIL Image, file path (str), or LIME explanation
            heatmap2: numpy array, PIL Image, file path (str), or LIME explanation
            explainer1_name: Name of first explainer
            explainer2_name: Name of second explainer
            show: Whether to display the plot
            save_path: Path to save the figure (optional)
        """
        # Load heatmaps
        hm1 = self._load_heatmap(heatmap1)
        hm2 = self._load_heatmap(heatmap2)

        hm1 = self.normalize_heatmap(hm1)
        hm2 = self.normalize_heatmap(hm2)
        hm1, hm2 = self._resize_to_match(hm1, hm2)

        comparison = self.compare(heatmap1, heatmap2, explainer1_name, explainer2_name)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot first heatmap
        im1 = axes[0].imshow(hm1, cmap="jet")
        axes[0].set_title(
            f"{explainer1_name}\n(SSIM: {comparison['ssim']:.3f})", fontsize=12
        )
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot second heatmap
        im2 = axes[1].imshow(hm2, cmap="jet")
        axes[1].set_title(
            f"{explainer2_name}\n(Cosine Sim: {comparison['cosine_similarity']:.3f})",
            fontsize=12,
        )
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Add overall comparison text
        fig.suptitle(
            f"Similarity: {comparison['interpretation']} (Avg: {comparison['average_similarity']:.3f})",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison visualization saved to {save_path}")

        if show:
            plt.show()

        return comparison

    def print_report(self, comparison_result):
        """
        Print a detailed comparison report.

        Args:
            comparison_result: Dictionary returned by compare() method
        """
        print("\n" + "=" * 60)
        print("HEATMAP COMPARISON REPORT")
        print("=" * 60)
        print(
            f"Comparing: {comparison_result['explainer1']} vs {comparison_result['explainer2']}"
        )
        print("-" * 60)
        print(f"SSIM (Structural Similarity):     {comparison_result['ssim']:.4f}")
        print(
            f"Cosine Similarity:                 {comparison_result['cosine_similarity']:.4f}"
        )
        print(
            f"Average Similarity:                {comparison_result['average_similarity']:.4f}"
        )
        print("-" * 60)
        print(f"Interpretation: {comparison_result['interpretation']}")
        print("=" * 60 + "\n")
