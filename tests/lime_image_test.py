import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import EXACT LIME explainer
from EXACT.explainers.lime_image_explainer import LimeExplainer_Image


# -------------------------------------------------
# Load Pretrained ResNet50
# -------------------------------------------------

weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

imagenet_classes = weights.meta["categories"]


# -------------------------------------------------
# ImageNet Transform (for prediction check)
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -------------------------------------------------
# Load Image
# -------------------------------------------------

img_path = "sample_dog.jpg"   # change if needed
image = Image.open(img_path).convert("RGB")

image_np = np.array(image)


# -------------------------------------------------
# Model Prediction
# -------------------------------------------------

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)

probs = torch.softmax(outputs, dim=1)

top_prob, top_class = torch.topk(probs, 1)

predicted_label = imagenet_classes[top_class.item()]
confidence = top_prob.item()

print("\nPrediction Result")
print("---------------------")
print("Predicted Class :", predicted_label)
print("Confidence      :", round(confidence * 100, 2), "%")


# -------------------------------------------------
# Initialize LIME Explainer
# -------------------------------------------------

explainer = LimeExplainer_Image(
    model=model,
    num_samples=1000,
    target_size=(224,224),
    smoothing_sigma=2
)


# -------------------------------------------------
# Generate Explanation
# -------------------------------------------------

print("\nGenerating LIME explanation (heatmap + boundary)...")

explanation = explainer.explain(
    image=image_np,
    top_labels=1,
    boundary_marking=True   # <-- enables boundary visualization
)


# -------------------------------------------------
# Verify Explanation Object
# -------------------------------------------------

print("\nExplanation generated successfully.")

print("Top predicted label index:", explanation.top_labels[0])

print("Outputs saved inside folder: user_saves/")