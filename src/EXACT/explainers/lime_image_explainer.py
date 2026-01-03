import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

class LimeImageExplainer:
    """
    LIME Image Explainer for Exact Library 
    Works with Pytorch and Tensorflow
    
    """

    def __init__(self, wrapped_model, num_samples = 1000):
        self.model = wrapped_model
        self.num_samples = num_samples # Number of perturbed samples LIME generates, More samples = better explanation but slower
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image, top_labels = 1):
        """
        image: numpy array (H, W, C)

        """
        def predict_fn(images):
            # Lime gives a list of images -> convert to numpy
            images = np.array(images)
            return self.model.predict_proba(images)
        
        explanation = self.explainer.explain_instance(
            image = image,
            classifier_fn = predict_fn,
            top_labels = top_labels,
            hide_color=0,
            num_samples=self.num_samples
        )

        return explanation
    
    def get_visualization(self, explanation, label = None, positive_only=True, num_features = 3, hide_rest=True):
        """
        Returns LIME visualization

        """

        if label is None:
            label = explanation.top_labels[0]

        image, mask = explanation.get_image_and_mask(
            label = label,
            positive_only = positive_only,
            num_features = num_features, # How many features we should show
            hide_rest = hide_rest # Hide irrelevant regions
        )

        return mark_boundaries(image, mask)