import cv2
import numpy as np
import torch
from pathlib import Path
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
from EXACT.utils import get_last_conv_layer


class DFF:
    """
    Deep Feature Factorization (DFF) for visual concept discovery.

    Instead of asking "where does the model see a cat?", DFF asks:
    "what are ALL the concepts the model detects in this image, and how
    do they relate to the model's predictions?"

    It factorizes the feature activations at a target layer using Non-Negative
    Matrix Factorization (NMF) into N concepts. Each concept gets a spatial
    heatmap showing where it appears in the image, and optionally a label
    showing what classes it scores highest for.

    Parameters
    ----------
    model : torch.nn.Module
        The model to explain.
    target_layer : torch.nn.Module, optional
        Layer whose activations are factorized. If None, uses the last
        convolutional layer. Best results come from the last feature layer
        (e.g. model.layer4 for ResNet, model.features[-1] for VGG).
    computation_on_concepts : torch.nn.Module, optional
        Sub-network applied to each concept embedding to produce class scores.
        For ResNet this is model.fc; for VGG it could be model.classifier.
        If None, concept class scores won't be computed.
    n_components : int, optional
        Number of concepts to extract. Default is 5.
        - Too low  -> distinct concepts get merged together
        - Too high -> over-segmentation into esoteric fragments
        Tune this based on the complexity of your images.
    save_dir : str, optional
        Directory to save output visualizations. Default is 'user_saves/dff_saves'.
    """

    def __init__(
        self,
        model,
        target_layer=None,
        computation_on_concepts=None,
        n_components=5,
        save_dir="user_saves/dff_saves",
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or get_last_conv_layer(model)
        self.computation_on_concepts = computation_on_concepts
        self.n_components = n_components
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._dff = DeepFeatureFactorization(
            model=self.model,
            target_layer=self.target_layer,
            computation_on_concepts=self.computation_on_concepts,
        )

    def explain(
        self,
        input_tensor,
        input_image=None,
        n_components=None,
        class_names=None,
        top_k=2,
        image_weight=0.3,
        save_png=False,
        tag="",
    ):
        """
        Discover and visualize the visual concepts in an image.

        Runs NMF on the model's internal activations to find N concepts,
        then overlays them on the image with distinct colors. If class names
        are provided, each concept is also labeled with its top predicted classes.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed model input, shape (1, C, H, W).
        input_image : np.ndarray or torch.Tensor, optional
            Original image for visualization, shape (H, W, 3) or (3, H, W),
            in [0, 1] or [0, 255]. If None, input_tensor[0] is used.
        n_components : int, optional
            Number of concepts to extract. Overrides the instance default.
        class_names : list of str, optional
            Class name strings indexed by class index. Used to label each
            concept with its top predicted categories. Requires
            computation_on_concepts to be set.
        top_k : int, optional
            Number of top classes to display per concept label. Default is 2.
        image_weight : float, optional
            How much of the original image to blend into the visualization.
            0.0 = only concept colors, 1.0 = only original image. Default 0.3.
        save_png : bool, optional
            Whether to save the visualization as a PNG. Default is False.
        tag : str, optional
            Optional string appended to the saved filename. Default is ''.

        Returns
        -------
        dict with keys:
            'visualization'    : np.ndarray (H, W, 3) - concept overlay image
            'concept_heatmaps' : np.ndarray (n_components, H, W) - per-concept spatial maps
            'concept_scores'   : np.ndarray (n_components, num_classes) or None
                                 Raw class scores per concept (before softmax).
                                 None if computation_on_concepts was not set.
            'concept_labels'   : list of str or None - labels shown in the legend
            'filepath'         : Path or None - where the image was saved
        """
        n = n_components or self.n_components

        concepts, batch_explanations, concept_scores = self._dff(input_tensor, n)
        concept_heatmaps = batch_explanations[0]  # shape: (n_components, H, W)

        # Build concept labels if we have class scores and class names
        concept_labels = None
        if concept_scores is not None and class_names is not None:
            concept_labels = self._build_labels(concept_scores, class_names, top_k)

        # Prepare the image for visualization
        if input_image is None:
            input_image = input_tensor[0]
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.cpu().numpy()
        if input_image.ndim == 3 and input_image.shape[0] == 3:
            input_image = np.transpose(input_image, (1, 2, 0))
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        input_image = np.float32(input_image)

        visualization = show_factorization_on_image(
            input_image,
            concept_heatmaps,
            image_weight=image_weight,
            concept_labels=concept_labels,
        )

        filepath = None
        if save_png:
            suffix = f"_{tag}" if tag else ""
            filepath = self.save_dir / f"dff_n{n}{suffix}.png"
            cv2.imwrite(str(filepath), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filepath}")

        return {
            "visualization": visualization,
            "concept_heatmaps": concept_heatmaps,
            "concept_scores": concept_scores,
            "concept_labels": concept_labels,
            "filepath": filepath,
        }

    def _build_labels(self, concept_scores, class_names, top_k):
        """
        Convert raw concept scores into readable legend label strings.

        Applies softmax across classes for each concept, picks the top_k
        highest scoring classes, and formats them as "classname:score".

        Parameters
        ----------
        concept_scores : np.ndarray
            Raw scores, shape (n_components, num_classes).
        class_names : list of str
            Class name strings indexed by class index.
        top_k : int
            Number of top classes to show per concept.

        Returns
        -------
        list of str
            One label string per concept, newline-separated top-k entries.
        """
        scores = torch.softmax(torch.from_numpy(concept_scores), dim=-1).numpy()
        top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]

        labels = []
        for concept_idx in range(top_indices.shape[0]):
            parts = []
            for class_idx in top_indices[concept_idx]:
                score = scores[concept_idx, class_idx]
                name = class_names[class_idx].split(",")[0][:20]
                parts.append(f"{name}:{score:.2f}")
            labels.append("\n".join(parts))

        return labels