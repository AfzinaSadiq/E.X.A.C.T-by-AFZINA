import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np


def test_comparer():
    """Test the HeatmapComparer with GradCAM and LIME explanations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare image
    tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = Image.open("models/Te-me_0010.jpg").convert("RGB")
    p_img = tf(img)

    # Define tumor model
    class tumor_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=256, out_features=4),
            ).to(device)

        def forward(self, x):
            return self.layer_stack(x)

    test_model = tumor_model()
    test_model.load_state_dict(torch.load("models/model_1.pth", map_location=device))

    # --------------------------USING THE EXACT PACKAGE----------------------------------------------------------

    from EXACT.wrappers import TorchWrapper
    from EXACT.explainers.gradcam import GradCAM
    from EXACT.explainers.lime_image_explainer import LimeImageExplainer
    from EXACT.comparer import HeatmapComparer

    wrapped_model = TorchWrapper(test_model)

    gradcam = GradCAM(wrapped_model)               # using the gradcam explainer
    gradcam_heatmap = gradcam.generate_heatmap(input_data=p_img)
    gradcam_overlay = gradcam.overlay_heatmap(
        heatmap=gradcam_heatmap, 
        image=img, 
        save_png=True
    )
    print("GradCAM heatmap generated")

    lime_explainer = LimeImageExplainer(
        wrapped_model, 
        num_samples=1000
    )                                              # using the lime explainer
    lime_explanation = lime_explainer.explain(
        img, 
        top_labels=1
    )
    lime_overlay = lime_explainer.overlay_heatmap(
        explanation=lime_explanation, 
        image=img, 
        save_png=True
    )
    print("Lime explanation generated")

    comparer = HeatmapComparer()                   # using the comparer

    comparison = comparer.compare(
        gradcam_heatmap,
        lime_explanation, 
        explainer1_name="GradCAM",
        explainer2_name="LIME",
    )

    comparer.print_report(comparison)

    comparer.visualize_comparison(
        gradcam_heatmap,
        lime_explanation,
        explainer1_name="Comparison 1",
        explainer2_name="Comparison 2",
        show=True,
        save_path="user_saves/gradcam_vs_lime_comparison.png",
    )

    return comparison


if __name__ == "__main__":
    test_comparer()
