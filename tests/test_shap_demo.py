"""
test_shap_image.py
------------------
Example usage of EXACT library — SHAP Image Explainer.

Shows how a user would use SHAPImageExplainer with their own model.
Uses pretrained ResNet18 (ImageNet) as a placeholder for any real model.
"""

import os
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import urllib.request

from EXACT.explainers.shap_image_explainer import SHAPImageExplainer


# ---------------------------------------------------------------------------
# Step 1 — Load your model
# ---------------------------------------------------------------------------
# Replace this with your own model:
#   model = YourCNN()
#   model.load_state_dict(torch.load("your_weights.pth"))

print("Loading model...")

try:
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
except ImportError:
    model = models.resnet18(pretrained=True)

model.eval()
print("Model loaded.\n")


# ---------------------------------------------------------------------------
# Step 2 — Load a real image
# ---------------------------------------------------------------------------
# Replace this with your own image:
#   image = np.array(Image.open("your_image.jpg").convert("RGB"))

IMAGE_PATH = "test_dog.jpg"

if not os.path.exists(IMAGE_PATH):
    print("Downloading sample image...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        IMAGE_PATH,
    )
    print(f"Image saved: {IMAGE_PATH}")

image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
print(f"Image shape : {image.shape}")
print(f"Image dtype : {image.dtype}\n")


# ---------------------------------------------------------------------------
# Step 2b — Load ImageNet class names
# ---------------------------------------------------------------------------
# This makes predictions show "Samoyed" instead of just "258"

CLASSES_PATH = "imagenet_classes.txt"

if not os.path.exists(CLASSES_PATH):
    print("Downloading ImageNet class names...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        CLASSES_PATH,
    )

with open(CLASSES_PATH) as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Loaded {len(class_names)} class names.\n")


# ---------------------------------------------------------------------------
# Step 3 — Create the explainer
# ---------------------------------------------------------------------------
# NOTE: We use explainer_type="partition" directly here because ResNet18 on
# newer PyTorch versions has inplace operations that break DeepExplainer's
# gradient hooks. PartitionExplainer works with any model without this issue.
#
# max_evals controls explanation quality vs speed:
#   500  -> fast but coarse rectangular blocks (default)
#   3000 -> detailed pixel-level explanation (recommended)
#   5000 -> very detailed, slower

explainer = SHAPImageExplainer(
    model          = model,
    explainer_type = "partition",
    target_size    = (224, 224),
    normalize      = True,
    max_evals      = 5000,
    class_names    = class_names,
)


# ---------------------------------------------------------------------------
# Step 4 — Get the explanation
# ---------------------------------------------------------------------------

print("Running explanation (takes ~2-3 mins with max_evals=3000)...")
result = explainer.explain(image)

print(f"\nPredicted class : {result['predicted_class']}")
print(f"Class name      : {result['class_name']}")
print(f"Confidence      : {result['confidence']:.2%}")
print(f"SHAP values     : shape {result['shap_values'].shape}")


# ---------------------------------------------------------------------------
# Step 5 — Visualize
# ---------------------------------------------------------------------------
# All plots are automatically saved to user_saves/
#
# heatmap -> red = pixels that pushed prediction UP, blue = pushed it DOWN
# masked  -> shows only the pixels the model focused on to make its decision
# signed  -> green = supports prediction, red = contradicts prediction

print("\nGenerating visualizations...")

explainer.visualize(result, image, style="heatmap")
explainer.visualize(result, image, style="masked")
explainer.visualize(result, image, style="signed")

print("\nDone. Check user_saves/ for the output plots.")