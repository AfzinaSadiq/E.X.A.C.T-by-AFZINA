import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def test_lime_image():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------- Image preprocessing --------------------------------------
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    img = Image.open("models/Te-me_0010.jpg").convert("RGB")
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
    
    from src.EXACT.explainers.lime_image_explainer import LimeImageExplainer

    lime_image_explainer = LimeImageExplainer(wrapped_model, num_samples = 1000)

    explanation = lime_image_explainer.explain(img_np, top_labels=1) 

    lime_image = lime_image_explainer.get_visualization(explanation)

    # -------------------------------------- Visualization --------------------------------------

    plt.figure(figsize=(10,4))

    # Original image
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis("off")

    #Limee explanation
    plt.subplot(1,2,2)
    plt.imshow(lime_image)
    plt.title("LIME Explanation")
    plt.axis("off")

    plt.suptitle("Tumor Detection - LIME Explanation", fontsize=14)

    os.makedirs("user_saves",exist_ok = True)
    plt.tight_layout()
    plt.savefig("user_saves/lime_image_explanation.png",bbox_inches="tight",dpi=300)
    plt.close()


    print("Saved the output image: outputs/lime_image_explanation.png")

if __name__ == '__main__':
    test_lime_image()
    