#explainers/vit_gradcame.py
import cv2
import numpy as np
import torch
from pathlib import Path
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    GradCAMPlusPlus,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image


# --- Reshape transforms ---
# ViT produces tokens of shape (B, num_patches + 1, embed_dim).
# The first token is the class token; the rest are spatial patches.
# We drop the class token and reshape the patches into a 2D spatial grid.

def _vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# Swin produces tokens of shape (B, num_patches, embed_dim) with no class token.
# We directly reshape into a 2D spatial grid.

def _swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class ViTGradCAM:
    """
    GradCAM for Vision Transformers (ViT and Swin).

    Standard GradCAM assumes CNN-style feature maps with spatial dimensions,
    but transformer layers produce token sequences. This class handles the
    reshape from token space back to 2D spatial maps automatically, and
    selects a sensible default target layer for each architecture.

    Supported architectures
    -----------------------
    'vit'  : Vision Transformer (timm / torchvision).
             Tokens: (B, num_patches + 1, embed_dim). Patch grid: 14x14.
             Target: model.blocks[-2].norm1  (second-to-last block)

    'swin' : Swin Transformer (timm / torchvision).
             Tokens: (B, num_patches, embed_dim). Patch grid: 7x7.
             Target: model.layers[-1].blocks[-1].norm1

    The final attention block is deliberately avoided as the target layer —
    the class token classification is not influenced by the spatial tokens
    there, so gradients with respect to them are zero.

    Parameters
    ----------
    model : torch.nn.Module
        A ViT or Swin model.
    arch : str
        Architecture type: 'vit' or 'swin'.
    target_layer : torch.nn.Module, optional
        The layer to compute CAM from. If None, a sensible default is
        chosen based on arch. Override this if using a non-standard model.
    patch_size : int, optional
        Spatial size of the patch grid (patch_size x patch_size).
        Default is 14 for ViT, 7 for Swin.
    save_dir : str, optional
        Directory to save visualizations. Default is 'user_saves/vit_cam_saves'.
    """

    # Methods that work well with transformers.
    # ScoreCAM, AblationCAM, and FullGrad are excluded — they either don't
    # support the reshape transform or are too slow for typical ViT usage.
    METHODS = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
    }

    def __init__(self, model, arch="vit", target_layer=None, patch_size=None, save_dir="user_saves/vit_cam_saves"):
        arch = arch.lower()
        if arch not in ("vit", "swin"):
            raise ValueError(f"arch must be 'vit' or 'swin', got '{arch}'")

        self.model = model
        self.model.eval()
        self.arch = arch
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.target_layer, self.reshape_transform = self._setup(
            arch, target_layer, patch_size
        )
        self._cam_objects = {}

    def _setup(self, arch, target_layer, patch_size):
        if arch == "vit":
            grid = patch_size or 14
            reshape = lambda t: _vit_reshape_transform(t, height=grid, width=grid)
            layer = target_layer or self.model.blocks[-2].norm1

        else:  # swin
            grid = patch_size or 7
            reshape = lambda t: _swin_reshape_transform(t, height=grid, width=grid)
            layer = target_layer or self.model.layers[-1].blocks[-1].norm1

        return layer, reshape

    def explain(self, input_tensor, input_image=None, targets=None, method="gradcam", class_name="", save_png=False):
        """
        Generate and visualize a CAM for a Vision Transformer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed input, shape (1, C, H, W).
        input_image : np.ndarray or torch.Tensor, optional
            Original image for the overlay, shape (H, W, 3) or (3, H, W),
            in [0, 1] or [0, 255]. If None, input_tensor[0] is used.
        targets : list, optional
            Target class indices. If None, uses the top predicted class.
        method : str, optional
            CAM method to use. One of: 'gradcam', 'hirescam', 'gradcam++',
            'xgradcam', 'eigencam'. Default is 'gradcam'.
        class_name : str, optional
            Appended to the saved filename. Default is ''.
        save_png : bool, optional
            Whether to save the visualization. Default is False.

        Returns
        -------
        dict with keys:
            'cam'           : np.ndarray (H, W) - grayscale activation map
            'visualization' : np.ndarray (H, W, 3) - overlay image
            'filepath'      : Path or None
            'method'        : method name used
        """
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(
                f"Method '{method}' not supported for transformers. "
                f"Available: {list(self.METHODS.keys())}"
            )

        # Cache CAM objects — reshape_transform is fixed per instance
        if method not in self._cam_objects:
            self._cam_objects[method] = self.METHODS[method](
                model=self.model,
                target_layers=[self.target_layer],
                reshape_transform=self.reshape_transform,
            )

        cam_obj = self._cam_objects[method]
        with torch.enable_grad():
            grayscale_cam = cam_obj(input_tensor=input_tensor, targets=targets)
        cam = grayscale_cam[0]

        # Prepare image
        if input_image is None:
            input_image = input_tensor[0]
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.cpu().numpy()
        if input_image.ndim == 3 and input_image.shape[0] == 3:
            input_image = np.transpose(input_image, (1, 2, 0))
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        input_image = np.float32(input_image)

        # Resize CAM to match image if needed
        h, w = input_image.shape[:2]
        if cam.shape != (h, w):
            cam = cv2.resize(cam, (w, h))

        visualization = show_cam_on_image(input_image, cam, use_rgb=True)

        filepath = None
        if save_png:
            suffix = f"_{class_name}" if class_name else ""
            filepath = self.save_dir / f"{self.arch}_{method}{suffix}.png"
            cv2.imwrite(str(filepath), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved: {filepath}")

        return {
            "cam": cam,
            "visualization": visualization,
            "filepath": filepath,
            "method": method,
        }