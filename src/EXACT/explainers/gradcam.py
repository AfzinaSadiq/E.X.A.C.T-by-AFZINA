import cv2
import numpy as np
import torch
from pathlib import Path
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
    GradCAMElementWise,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from EXACT.utils import get_last_conv_layer


class GradCAM:
    """
    A user-friendly wrapper for pytorch-grad-cam XAI methods.

    Supports multiple Gradient-based Class Activation Map techniques:
    - GradCAM
    - HiResCAM
    - ScoreCAM
    - GradCAMPlusPlus
    - AblationCAM
    - XGradCAM
    - EigenCAM
    - FullGrad
    - GradCAMElementWise

    Parameters
    ----------
    model : torch.nn.Module
    target_layers : list, optional
        List of target layers for CAM computation. If None, uses last conv layer
    save_dir : str, optional
        Default is 'user_saves/'
    """

    METHODS = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
    }

    def __init__(self, model, target_layers=None, save_dir="user_saves/"):
        self.model = model
        self.model.eval()
        self.target_layers = target_layers or get_last_conv_layer(model)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cam_objects = {}

    def generate_cam(
        self,
        method="gradcam",
        input_tensor=None,
        targets=None,
    ):
        """
        Generate Class Activation Map using the specified method.

        Parameters
        ----------
        method : str, optional
            The CAM method to use. Options: 'gradcam', 'hirescam', 'scorecam',
            'gradcam++', 'ablationcam', 'xgradcam', 'eigencam', 'fullgrad',
            'gradcamelementwise'. Default is 'gradcam'.
        input_tensor : torch.Tensor
            Input tensor to generate CAM for. Shape: (B, C, H, W)
        targets : list, optional
            List of target class indices. If None, uses max probability class.

        Returns
        -------
        grayscale_cam : np.ndarray
            Grayscale CAM of shape (H, W)
        """
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(
                f"Method '{method}' not supported. "
                f"Available: {list(self.METHODS.keys())}"
            )

        # Initialize CAM object if not already done
        if method not in self.cam_objects:
            cam_class = self.METHODS[method]
            self.cam_objects[method] = cam_class(
                self.model,
                self.target_layers,
            )

        cam_obj = self.cam_objects[method]

        # Generate CAM (enable gradients for gradient-based methods)
        with torch.enable_grad():
            grayscale_cam = cam_obj(
                input_tensor=input_tensor,
                targets=targets,
            )

        return grayscale_cam[0]  # Return first batch item

    def visualize_and_save(
        self, input_image, cam, method="gradcam", class_name="", save_png=False
    ):
        """
        Visualize CAM overlay on the original image and optionally save it.

        Parameters
        ----------
        input_image : np.ndarray or torch.Tensor
            Original input image. Shape: (H, W, 3) or (3, H, W)
        cam : np.ndarray
            Grayscale CAM from generate_cam(). Shape: (H, W)
        method : str, optional
            Method name for filename. Default is 'gradcam'.
        class_name : str, optional
            Class name for filename. Default is empty.
        save_png : bool, optional
            Whether to save the visualization. Default is False.

        Returns
        -------
        visualization : np.ndarray
            RGB image with CAM overlay
        filepath : str or None
            Path where image was saved (None if save_png=False)
        """
        # Convert tensor to numpy if needed
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.cpu().numpy()

        # Handle channel-first format
        if input_image.shape[0] == 3:
            input_image = np.transpose(input_image, (1, 2, 0))

        # Normalize image if needed
        if input_image.max() > 1.0:
            input_image = input_image / 255.0

        # Normalize CAM to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Upscale CAM to match image dimensions (preserve image quality)
        input_h, input_w = input_image.shape[0], input_image.shape[1]
        cam_h, cam_w = cam.shape
        if (input_h, input_w) != (cam_h, cam_w):
            cam = cv2.resize(cam, (input_w, input_h))

        # Create overlay
        visualization = show_cam_on_image(input_image, cam, use_rgb=True)

        filepath = None
        if save_png:
            # Create filename
            class_suffix = f"_{class_name}" if class_name else ""
            filename = f"{method.lower()}{class_suffix}.png"
            filepath = self.save_dir / filename

            # Convert RGB to BGR for cv2.imwrite
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), visualization_bgr)
            print(f"✓ Saved: {filepath}")

        return visualization, filepath

    def explain(
        self,
        input_tensor,
        method="gradcam",
        targets=None,
        input_image=None,
        class_name="",
        save_png=False,
    ):
        """
        Complete pipeline: generate CAM and visualize with optional save.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor for model. Shape: (1, 3, H, W)
        method : str, optional
            CAM method to use. Default is 'gradcam'.
        targets : list, optional
            Target class indices. If None, uses max probability class.
        input_image : np.ndarray, optional
            Original image for visualization. If None, uses input_tensor.
        class_name : str, optional
            Class name for saving. Default is empty.
        save_png : bool, optional
            Whether to save visualization. Default is False.

        Returns
        -------
        dict
            Dictionary containing:
            - 'cam': grayscale CAM
            - 'visualization': RGB image with overlay
            - 'filepath': path to saved image (or None)
            - 'method': method used
        """
        # Generate CAM
        cam = self.generate_cam(
            method=method, input_tensor=input_tensor, targets=targets
        )

        # Use input_tensor for visualization if input_image not provided
        if input_image is None:
            input_image = input_tensor[0]

        # Visualize and save
        visualization, filepath = self.visualize_and_save(
            input_image=input_image,
            cam=cam,
            method=method,
            class_name=class_name,
            save_png=save_png,
        )

        return {
            "cam": cam,
            "visualization": visualization,
            "filepath": filepath,
            "method": method,
        }

    def use_all_methods(
        self, input_tensor, methods=None, input_image=None, targets=None, save_png=False
    ):
        """
        Generate CAM using multiple methods for comparison.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor for model. Shape: (1, 3, H, W)
        methods : list, optional
            List of methods to compare. If None, uses all available methods.
        input_image : np.ndarray, optional
            Original image for visualization.
        targets : list, optional
            Target class indices.
        save_png : bool, optional
            Whether to save visualizations. Default is False.

        Returns
        -------
        dict
            Dictionary with method names as keys and results as values
        """
        if methods is None:
            methods = list(self.METHODS.keys())

        results = {}
        for method in methods:
            try:
                result = self.explain(
                    input_tensor=input_tensor,
                    method=method,
                    targets=targets,
                    input_image=input_image,
                    save_png=save_png,
                )
                results[method] = result
                print(f"✓ {method.upper()} computed successfully")
            except Exception as e:
                print(f"✗ {method.upper()} failed: {str(e)}")
                results[method] = None

        return results

    def list_methods(self):
        """Print all available CAM methods."""
        print("Available CAM Methods:")
        for i, method in enumerate(self.METHODS.keys(), 1):
            print(f"  {i}. {method}")

    def get_model(self):
        """Get the underlying model."""
        return self.model