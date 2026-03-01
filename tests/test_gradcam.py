import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

def test1():
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
    
    ##------------------------------------------Actual testing code------------------
    from EXACT.explainers import GradCAM

    test_model = tumor_model()
    test_model.load_state_dict(torch.load("models/model_1.pth", map_location = device))
    explainer = GradCAM(test_model)
    result = explainer.use_all_methods(
        input_tensor = p_img,
        input_image = img_np,
        save_png = True
    )
    print(result)
    ##-------------------------------------------------------------------------------

if __name__ == "__main__":
    test1()
    