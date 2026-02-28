import torch
from torch import nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from EXACT.explainers.integrated_gradient import IntegratedGradients


def test_integrated_gradient():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------- Load Image --------------------------------------

    img = Image.open("models\\Te-me_0010.jpg").convert("RGB")
    img = img.resize((128, 128))

    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    

    # -------------------------------------- Define CNN Architecture --------------------------------------

    class TumorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 4)
            )

        def forward(self, x):
            return self.layer_stack(x)

    model = TumorModel().to(device)
    model.load_state_dict(torch.load("models/model_1.pth", map_location=device))
    model.eval()

    # -------------------------------------- Preprocess Image --------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),   # Converts to [0,1] and shape (C,H,W)
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    # -------------------------------------- Integrated Gradients --------------------------------------

    ig = IntegratedGradients(model)

    # ✅ Check prediction BEFORE running IG
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(probs).item()
        print("Predicted class:", predicted)
        print("All probs:", probs)
        

    # ✅ Now explicitly pass target_class
    result = ig.explain(
        input_tensor=input_tensor,
        original_image=img_np,
        target_class=predicted,  # ← explicitly pass predicted class
        return_mask=True
    )

     # Check spatial variance of attributions
    attrs = result["attributions"].squeeze(0)  # [3, H, W]
    spatial_std = attrs.std(dim=[1,2])          # std per channel
    print("Spatial std per channel:", spatial_std)

    cv2.imwrite("user_saves/ig_overlay.png", result["overlay"])
    cv2.imwrite("user_saves/ig_heatmap.png", result["heatmap"])
    cv2.imwrite("user_saves/ig_mask.png", result["mask"])

    print("Integrated Gradients results saved in user_saves/")


if __name__ == "__main__":
    test_integrated_gradient()