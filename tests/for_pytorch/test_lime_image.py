import torch
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def test_lime_image():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    img = Image.open("models\\Te-me_0010.jpg").convert("RGB")
    # img_tensor = transform(img)

    # Lime needs numpy image (H, W, C)
    img_np = np.array(img.resize((128,128)))


    # -------------------------------------- Simple CNN Model --------------------------------------
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

        def forward(self,x):
            return self.layer_stack(x)
        
    
    model = TumorModel().to(device)
    model.load_state_dict(torch.load("models/model_1.pth",map_location=device))
    model.eval()

    # -------------------------------------- Wrap model --------------------------------------
   
    from src.EXACT.wrappers import TorchWrapper
    wrapped_model = TorchWrapper(model)

    # -------------------------------------- LIME Explainer --------------------------------------
    
    from src.EXACT.explainers.lime_image_explainer import LimeExplainer_Image

    lime_image_explainer = LimeExplainer_Image(wrapped_model, num_samples = 1000)

    explanation = lime_image_explainer.explain(img_np, top_labels=1) 

    # -------------------------------------- Visualization --------------------------------------

    os.makedirs("outputs",exist_ok=True)

    lime_image_explainer.plot_explanation(
        explanation = explanation,
        original_image = img_np,
        num_features = 3,
        save_path="outputs/lime_image_explanation2.0.png",
        show = True
    )

    print("LIME image explanation saved to outputs/lime_image_explanation.png")

if __name__ == '__main__':
    test_lime_image()
    