import torch
import torch.nn as nn


def get_last_conv_layer(model):
    """
    This function traverses the model architecture and identifies the final
    convolutional layer (Conv1d, Conv2d, or Conv3d), returning it as a list
    suitable for use with pytorch-grad-cam.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch CNN model

    Returns
    -------
    list
        List containing the last convolutional layer. Returns empty list if
        no convolutional layer is found.
    """
    conv_layers = []

    # Recursively find all convolutional layers
    def find_conv_layers(module):
        for child in module.children():
            if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_layers.append(child)
            else:
                # Recursively search child modules
                find_conv_layers(child)

    find_conv_layers(model)

    # Return the last convolutional layer as a list
    if conv_layers:
        return [conv_layers[-1]]
    else:
        # Fallback: return empty list if no conv layer found
        print("Warning: No convolutional layer found in the model.")
        return []
