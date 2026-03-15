import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch
from EXACT.utils import predict_proba_fn
from PIL import Image
import os
import time
from scipy.ndimage import gaussian_filter


class LimeExplainer_Image:

    def __init__(self, model, num_samples=1000, target_size=(224,224), smoothing_sigma=2, random_state = 42):

        self.model = model
        self.num_samples = num_samples
        self.target_size = target_size
        self.smoothing_sigma = smoothing_sigma
        self.random_state = random_state

        self.explainer = lime_image.LimeImageExplainer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Image loader
    # --------------------------------------------------

    @staticmethod
    def _load_image(image):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, np.ndarray):

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            image = Image.fromarray(image)

        return image

    # --------------------------------------------------
    # Resize image for LIME
    # --------------------------------------------------

    def _resize_for_model(self, image):

        resized = image.resize(self.target_size)

        img_array = np.array(resized, dtype=np.float32) / 255.0

        return img_array

    # --------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------

    def _generate_heatmap(self, explanation, label=None):

        if label is None:
            label = explanation.top_labels[0]

        segments = explanation.segments
        weights = dict(explanation.local_exp[label])

        heatmap = np.zeros(segments.shape)

        for superpixel, weight in weights.items():
            heatmap[segments == superpixel] = weight

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        heatmap = gaussian_filter(heatmap, sigma=self.smoothing_sigma)

        return heatmap

    # --------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------

    def _overlay_heatmap(self, image, heatmap, alpha=0.5):

        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]

        overlay = image * (1 - alpha) + heatmap_color * alpha

        overlay = np.clip(overlay, 0, 1)

        return overlay

    # --------------------------------------------------
    # Boundary visualization
    # --------------------------------------------------

    def _generate_boundary_visualization(self, explanation, label=None,
                                         positive_only=True,
                                         num_features=5,
                                         hide_rest=False):

        if label is None:
            label = explanation.top_labels[0]

        lime_image, mask = explanation.get_image_and_mask(
            label=label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest
        )

        boundary_image = mark_boundaries(lime_image, mask)

        return boundary_image

    # --------------------------------------------------
    # MAIN EXPLAIN FUNCTION
    # --------------------------------------------------

    def explain(self, image, top_labels=1, boundary_marking=False):

        # Load original image
        original_image_pil = self._load_image(image)

        original_size = original_image_pil.size

        original_image_np = np.array(original_image_pil) / 255.0

        # Resize for model
        lime_input_image = self._resize_for_model(original_image_pil)

        # Prediction function
        def predict_fn(images):

            images = np.array(images)

            return predict_proba_fn.predict_proba(images, model=self.model)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image=lime_input_image,
            classifier_fn=predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=self.num_samples
        )

        # --------------------------------------------------
        # Heatmap generation (ALWAYS)
        # --------------------------------------------------

        heatmap = self._generate_heatmap(explanation)

        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))

        heatmap_resized = heatmap_pil.resize(original_size, Image.BILINEAR)

        heatmap_resized = np.array(heatmap_resized) / 255.0

        overlay = self._overlay_heatmap(original_image_np, heatmap_resized)

        # Save heatmap
        save_dir = "user_saves"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = int(time.time())

        heatmap_path = os.path.join(save_dir, f"lime_heatmap_{timestamp}.png")

        overlay_uint8 = (overlay * 255).astype(np.uint8)

        Image.fromarray(overlay_uint8).save(heatmap_path)

        print(f"LIME heatmap saved to {heatmap_path}")

        # --------------------------------------------------
        # Boundary visualization (OPTIONAL)
        # --------------------------------------------------

        if boundary_marking:

            boundary_image = self._generate_boundary_visualization(explanation)

            boundary_image_uint8 = (boundary_image * 255).astype(np.uint8)

            boundary_path = os.path.join(save_dir, f"lime_boundary_{timestamp}.png")

            Image.fromarray(boundary_image_uint8).save(boundary_path)

            print(f"LIME boundary explanation saved to {boundary_path}")

        return explanation