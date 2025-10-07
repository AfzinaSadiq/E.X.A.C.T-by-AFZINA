import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ..wrappers import BaseWrapper, TorchWrapper, TFWrapper
class GradCAM:
    """The Grad-CAM explainer module (original grad-cam).\n
       Supported Functionalities:\n
       -> .generate_heatmap(input_data, class_index) -> returns gradcam heatmap as a numpy object\n
       -> .overlay_heatmap(heatmap, image, alpha, show, cmap) -> returns upsampled/resized heatmap that can be overlayed onto input image. Generates plot with heatmap overlayed image if show = True"""

    def __init__(self, wrapped_model: BaseWrapper, target_layer = None):
        self.model = wrapped_model
        self.target_layer = target_layer or self.model.get_last_conv_layer()[0]

    def generate_heatmap(self, input_data, class_index = None):
        if isinstance(self.model, TorchWrapper):
            return self._torch_gradcam(input_data, class_index)
        elif isinstance(self.model, TFWrapper):
            return self._tf_gradcam(input_data, class_index)
        else:
            return NotImplementedError("GradCAM not supported for this backend")
        
    def overlay_heatmap(self, heatmap, image, alpha = 0.4, show = True, cmap = "jet", save_png = False):
        #heatmap = np.heatmap(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

        heatmap_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
        ) / 255.0

        colormap = plt.get_cmap(cmap)
        heatmap_color = colormap(heatmap_resized)[..., :3]

        image_np = np.array(image)

        overlay = (image_np * (1 - alpha) + heatmap_color * 255 *alpha).astype(np.uint8)
        
        if show:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].set_title("Original")
            ax[0].imshow(image_np[..., ::-1])  # BGR -> RGB
            ax[0].axis("off")

            ax[1].set_title("Heatmap")
            ax[1].imshow(heatmap_resized, cmap = cmap)
            ax[1].axis("off")

            ax[2].set_title("Overlay")
            ax[2].imshow(overlay)
            ax[2].axis("off")

            plt.show()

        if save_png:
            import os
            save_dir = "user_saves"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "gradcam_overlay.png")
            Image.fromarray(overlay).save(save_path)
            print(f"Overlay heatmap saved to {save_path}")

        return overlay
    
    def _torch_gradcam(self, input_data, class_index = None):
        """Implementation of core Grad-CAM computation using PyTorch. """
        import torch
        
        model = self.model.model
        model.eval()

        features = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal features
            features = out.detach()
        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        target_layer = self.target_layer
        handle_fw = target_layer.register_forward_hook(forward_hook)
        handle_bw = target_layer.register_full_backward_hook(backward_hook)

        if isinstance(input_data, torch.Tensor):
            inp = input_data.detach().clone().float().unsqueeze(0).to(self.model.device)
        elif isinstance(input_data, np.ndarray):
            inp = torch.from_numpy(input_data).float().unsqueeze(0).to(self.model.device)
        elif isinstance(input_data, Image.Image):
            inp = torch.from_numpy(np.array(input_data)).float().unsqueeze(0).to(self.model.device)
        else:
            raise TypeError("Unsupported input type for GradCAM: {}".format(type(input_data)))
        
        output = model(inp)

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        loss = output[:, class_index]
        model.zero_grad()
        loss.backward()

        weights = gradients.mean(dim = [2,3], keepdim = True)
        cam = (weights * features).sum(dim = 1).squeeze().cpu().numpy()
        cam = np.maximum(cam,0)
        cam = cam / (cam.max() + 1e-8)

        handle_fw.remove()
        handle_bw.remove()

        return cam

    def _tf_gradcam(self):
        import tensorflow as tf
        pass