import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

from EXACT.explainers import GradCAM
from EXACT.comparators import HeatmapComparator

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open("models/Te-me_0010.jpg").convert("RGB")
    img_np = np.array(Image.open("models/Te-me_0010.jpg"))
    p_img = tf(img).unsqueeze(0).to(device)

    class tumor_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.MaxPool2d(kernel_size = 2),
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                nn.MaxPool2d(kernel_size = 2),
                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                nn.MaxPool2d(kernel_size = 2),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features = 256, out_features = 4)
            ).to(device)
        def forward(self, x):
            return self.layer_stack(x)
        
    test_model = tumor_model()
    test_model.load_state_dict(torch.load("models/model_1.pth", map_location = device))
    explainer = GradCAM(test_model)
    gradcam_result = explainer.explain(
        input_tensor = p_img,
        input_image = img_np,
        method= "gradcam",
        save_png = True
    )
    gradcampp_result = explainer.explain(
        input_tensor = p_img,
        input_image = img_np,
        method= "gradcam++",
        save_png = True
    )
    xgradcam_result = explainer.explain(
        input_tensor = p_img,
        input_image = img_np,
        method= "xgradcam",
        save_png = True
    )
    eigencam_result = explainer.explain(
        input_tensor = p_img,
        input_image = img_np,
        method= "eigencam",
        save_png = True
    )

    cmp = HeatmapComparator(model = test_model, device=device, deletion_steps=10, faithfulness_enabled=True)
    
    results = cmp.compare(
    entries={
        # (result_dict,  explainer_object,  extra_kwargs_for_stability_reruns)
        "GradCAM":   (gradcam_result,   explainer, {"method": "gradcam"}),
        "GradCAM++": (gradcampp_result, explainer, {"method": "gradcam++"}),
        "EigenCAM":  (eigencam_result,  explainer, {"method": "eigencam"}),
        "XGradCAM":  (xgradcam_result,  explainer, {"method": "xgradcam"}),
        # ViT example:
        # "ViT-CAM": (vit_result, vit_exp, {}),
        #
        # Skip stability for a specific method by passing None as the explainer:
        # "EigenCAM": (eigencam_result, None, {}),
        },
        input_tensor=p_img,
        input_image=img_np,
        stability_runs=10,
        noise_std=0.05,
    )

    cmp.report(results)
    cmp.plot(results, save_png=True, filename="comparison_test.png")

if __name__ == "__main__":
    test()