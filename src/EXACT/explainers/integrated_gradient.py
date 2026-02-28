import torch
import numpy as np
import cv2


class IntegratedGradients:
    """
    Integrated Gradients Explainer
    Supports any differentiable PyTorch model.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.model.eval()
        self.device = device or next(model.parameters()).device

    # -----------------------------------------------------------
    # 1️⃣ Core Integrated Gradients Computation
    # -----------------------------------------------------------
    def generate_attributions(
        self,
        input_tensor,
        target_class=None,
        baseline=None,
        steps=200
    ):
        """
        Returns raw Integrated Gradients attribution tensor.
        Shape: same as input_tensor
        """

        input_tensor = input_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device) 
        else:
            baseline = baseline.to(self.device)

        # Get target class automatically if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()

        # Generate scaled inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(1, steps + 1)
        ]

        gradients = []

        for i, scaled_input in enumerate(scaled_inputs):
            scaled_input: torch.Tensor = scaled_input.clone().detach().requires_grad_(True)

            output = self.model(scaled_input)
            target = output[0, target_class]

            # self.model.zero_grad()
            target.backward()

            gradients.append(scaled_input.grad.detach())

             # ✅ Add debug print here
            if i % 50 == 0:
                print(f"Step {i} gradient norm: {scaled_input.grad.norm().item():.6f}")

        avg_gradients = torch.mean(torch.stack(gradients), dim=0)

        integrated_grads = (input_tensor - baseline) * avg_gradients

        return integrated_grads.detach()

    # -----------------------------------------------------------
    # 2️⃣ Convert to Positive Attribution Map
    # -----------------------------------------------------------
    def get_positive_attribution_map(self, attributions):
        """
        Keeps only positive contributions.
        Output shape: [H, W]
        """

        attributions = attributions.squeeze(0)  # remove batch

        # Sum across channels (RGB → single map)
        # Take absolute value BEFORE summing

        attr_map =  torch.sum(torch.abs(attributions), dim=0)

        return attr_map

    # -----------------------------------------------------------
    # 3️⃣ Generate Heatmap
    # -----------------------------------------------------------
    def generate_heatmap(self, attr_map):
        """
        Converts attribution map to normalized heatmap.
        """

        attr_map = attr_map.cpu().numpy()

        # Normalize
        # percentile = np.percentile(attr_map, 99)
        # attr_map = np.clip(attr_map, 0, percentile)
        attr_map = attr_map - attr_map.min()

        attr_map = attr_map / (attr_map.max() + 1e-8)

        heatmap = np.uint8(255 * attr_map)

        return heatmap


    # -----------------------------------------------------------
    # 4️⃣ Overlay Heatmap on Original Image
    # -----------------------------------------------------------
    def overlay_heatmap(self, heatmap, original_image, alpha=0.5):
        """
        Overlays heatmap on original image.
        original_image must be in OpenCV BGR format.
        """

        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            original_image,
            1 - alpha,
            colored_heatmap,
            alpha,
            0
        )

        return overlay

    # -----------------------------------------------------------
    # 5️⃣ Generate Binary Mask
    # -----------------------------------------------------------
    def generate_binary_mask(self, attr_map, threshold_percent=95):
        """
        Returns binary mask of top important regions.
        threshold_percent = keep top X% important pixels
        """

        attr_np = attr_map.cpu().numpy()

        threshold = np.percentile(attr_np, threshold_percent)

        mask = np.zeros_like(attr_np)
        mask[attr_np >= threshold] = 255

        mask = np.uint8(mask)

        return mask

    # -----------------------------------------------------------
    # 6️⃣ Full Explain Pipeline
    # -----------------------------------------------------------
    def explain(
        self,
        input_tensor,
        original_image,
        target_class=None,
        baseline=None,
        steps=200,
        return_mask=False
    ):
        """
        Full pipeline:
        - Compute IG
        - Extract positive attribution map
        - Generate heatmap
        - Overlay on image
        - Optionally return mask
        """

        attributions = self.generate_attributions(
            input_tensor=input_tensor,
            target_class=target_class,
            baseline=baseline,
            steps=steps
        )

        positive_map = self.get_positive_attribution_map(attributions)

        heatmap = self.generate_heatmap(positive_map)

        overlay = self.overlay_heatmap(heatmap, original_image)

        if return_mask:
            mask = self.generate_binary_mask(positive_map)
            return {
                "attributions": attributions,
                "positive_map": positive_map,
                "heatmap": heatmap,
                "overlay": overlay,
                "mask": mask
            }

        return {
            "attributions": attributions,
            "positive_map": positive_map,
            "heatmap": heatmap,
            "overlay": overlay
        }