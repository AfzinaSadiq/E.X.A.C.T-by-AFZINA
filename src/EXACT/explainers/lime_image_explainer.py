import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch
from EXACT.utils import predict_proba_fn
from PIL import Image
import os

class LimeExplainer_Image:
    """
    LIME Image Explainer for Exact Library 
    Main responsibilities:
        - Generate LIME explanations
        - Return raw visualization data (image + mask)
        - Provide optional plotting utilites for  users
    """

    def __init__(self, model, num_samples = 1000, target_size = (128,128)):
        self.model = model
        self.num_samples = num_samples # Number of perturbed samples LIME generates, More samples = better explanation but slower
        self.target_size = target_size
        self.explainer = lime_image.LimeImageExplainer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _preprocess_image(image,target_size):
        """
        Preprocess image: Covert to PIL, resize and normalize

        Args: 
            image: PIL image, numpy array or file path (str)
            target_size: Tuple (height, width) to resize to

        Returns: 
            numpy array: Normalized image in range [0, 1] with shape (H, W, C)
        """

        # Load image if it's  a file path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert numpy array to PIL image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Resize to target size
        image = image.resize(target_size)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image,dtype=np.float32) / 255.0

        return img_array

    # Core explanation logic
    def explain(self, image, top_labels = 1, preprocess = True):
        """
        Generate a LIME explanation for a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image in (H, W, C) format
        top_labels :int
            Number of top predicted classes to explain

        Returns
        -------
        explanation: lime.explanation.Explanation
            LIME explanation object

        """

        # Preprocess image
        if preprocess:
            image = self._preprocess_image(image, self.target_size)

        def predict_fn(images):
            # Lime gives a list of images -> convert to numpy
            images = np.array(images)
            return predict_proba_fn.predict_proba(images,model=self.model)
        
        explanation = self.explainer.explain_instance(
            image = image,
            classifier_fn = predict_fn,
            top_labels = top_labels,
            hide_color=0,
            num_samples=self.num_samples
        )

        return explanation
    
    # RAW visualization data (NO plotting)
    def get_visualization_data(self, explanation, label = None, positive_only=True, num_features = 3, hide_rest=True):
        """
        Extract visualization data (image + mask) from LIME explanation.

        Parameters
        ----------
        explanation : lime.explanation.Explanation
            Lime explanation object
        label : int, optional
            Class label to visualize (defaults to top predicted label)
        positive_only: bool
            Show only positive contributing regions
        num_features: int
            Number of superpixels to displa y
        hide_rest : bool
            Hide non-important  regions

        Returns
        --------
        lime_image : np.ndarray
            Image with important region retained
        mask : np.ndarray
            Binary mask indicating important superpixels
        
        """

        if label is None:
            label = explanation.top_labels[0]

        lime_image, mask = explanation.get_image_and_mask(
            label = label,
            positive_only = positive_only,
            num_features = num_features, # How many features we should show
            hide_rest = hide_rest # Hide irrelevant regions
        )

        
        return lime_image, mask # We return Lime image + mask without plotting
    
    # Ploting utility (Optional for users)
    def overlay_heatmap(self, explanation, original_image = None, label = None, positive_only = False, num_features = 3, hide_rest = False, figsize=(8,4), title = "LIME Explanation", show = True, save_png = None):
        """
        Plot and optionally save the LIME image explanation.

        Parameters
        ----------
        explanation : lime.explanation.Explanation
            LIME explanation object
        
        original_image : np.ndarray or PIL.Image, optional
            Original image for side by side comparison
        
        label: int, optional
            Class label to visualize
        
        positive_only : bool
            Show only positive contributions

        num_features : int
            Number of superpixels to display

        hide_rest : bool
            Hide non-important regions

        figsize : tuple
            Figure size

        title : str
            Plot title

        show : bool
            Whether to display the plot

        save_path : str, optional
            Path to save the plotted image

        Returns
        -------
        overlay_image : np.ndarray
            LIME visualization image with boundaries
        
        """   

        lime_image, mask = self.get_visualization_data(
            explanation = explanation,
            label = label,
            positive_only = positive_only,
            num_features = num_features,
            hide_rest = hide_rest
        )

        overlay_image = mark_boundaries(lime_image, mask)

        plt.figure(figsize = figsize)

        if original_image is not None:
            original_image = self._preprocess_image(original_image,self.target_size)
            plt.subplot(1,2,1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.imshow(overlay_image)
            plt.title(title)
            plt.axis("off")
        else:
            plt.imshow(overlay_image)
            plt.title(title)
            plt.axis("off")

        if save_png:
            save_dir = "user_saves"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "lime_image_explanation_2.0.png")

            # Convert visualization to uint8 and save
            vizualization_uint8 = (overlay_image * 255).astype(np.uint8)
            Image.fromarray(vizualization_uint8).save(save_path)
            print(f"LIME visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

        
        return overlay_image
