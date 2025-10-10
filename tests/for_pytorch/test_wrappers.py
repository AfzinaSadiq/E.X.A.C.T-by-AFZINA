import torch
from torch import nn
from EXACT.wrappers import TorchWrapper

###To run any file under tests module type in the command : python -m tests.filename    here : (python -m tests.for_pytorch.test_wrappers)

def test1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ##Glenn Mathews Sep 21 - test status - Successfull!!
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
    wrapped_model = TorchWrapper(test_model)
    wrapped_model.load(path = "models/model_1.pth")
    print(wrapped_model.get_params())
    print(wrapped_model.get_last_conv_layer())


#nandu sep 23 test status: 
def test2():
    pass

if __name__ == "__main__":
    test1()



