import torch
from torch import nn
from PIL import Image
from torchvision import transforms

def test1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open("models/Te-me_0010.jpg").convert("RGB")
    p_img = tf(img)


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
    from EXACT.wrappers import TorchWrapper
    from EXACT.explainers.gradcam import GradCAM

    test_model = tumor_model()
    test_model.load_state_dict(torch.load("models/model_1.pth", map_location = device))
    wrapped_model = TorchWrapper(test_model)
    gradcam = GradCAM(wrapped_model)
    heatmap = gradcam.generate_heatmap(input_data=p_img)
    final_heatmap = gradcam.overlay_heatmap(heatmap=heatmap, image = img, save_png=True)
    print(final_heatmap)
    ##-------------------------------------------------------------------------------

if __name__ == "__main__":
    test1()
    