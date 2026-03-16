import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import urllib.request
import os

from EXACT.explainers.shap_image_explainer import ShapExplainer_Image


def test_shap_image_pretrained():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load pretrained ResNet18
    #    Pretrained on ImageNet — 1000 classes, strong feature extractor.
    #    No training needed — weights are downloaded automatically.
    # ──────────────────────────────────────────────────────────────────────

    print("Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Download a sample image
    #    Using a freely available cat image from Wikipedia.
    #    Saved locally so it can be reused without re-downloading.
    # ──────────────────────────────────────────────────────────────────────

    print("Downloading sample image...")
    img_url  = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"
    img_path = "sample_dog.jpg"

    if not os.path.exists(img_path):
        urllib.request.urlretrieve(img_url, img_path)
        print(f"Image saved to: {img_path}")
    else:
        print(f"Image already exists: {img_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Preprocess image
    #    ResNet18 expects:
    #        - Input size: (3, 224, 224)
    #        - Normalized with ImageNet mean and std
    #    For SHAP we need the raw [0,1] numpy image (HWC) as well.
    # ──────────────────────────────────────────────────────────────────────

    # ImageNet normalization constants
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                             # → (C, H, W), [0, 1]
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Load image with PIL
    pil_image = Image.open(img_path).convert("RGB")

    # Preprocessed tensor for model prediction check
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Raw [0,1] numpy image in HWC format — this is what SHAP receives
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                             # → (C, H, W), [0, 1]
    ])
    image_tensor = raw_transform(pil_image)                # (3, 224, 224)
    image_np     = image_tensor.permute(1, 2, 0).numpy()  # (224, 224, 3) HWC

    # ──────────────────────────────────────────────────────────────────────
    # 4. Model prediction
    # ──────────────────────────────────────────────────────────────────────

    with torch.no_grad():
        logits    = model(input_tensor)
        probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted = int(probs.argmax())
        top5      = probs.argsort()[::-1][:5]

    print(f"\nTop-5 predictions:")
    for rank, idx in enumerate(top5):
        print(f"  {rank+1}. Class {idx:4d}  —  {probs[idx]*100:.2f}%")
    print(f"\nPredicted class index : {predicted}")
    print(f"Confidence            : {probs[predicted]*100:.2f}%")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Wrap model with ImageNet normalization + top-5 output only
    #
    #    Two reasons for this wrapper:
    #    (a) SHAP passes raw [0,1] masked images — ResNet18 needs normalized
    #        input, so normalization is applied inside forward().
    #    (b) SHAP allocates a (1, 224, 224, 3, num_classes) array — with
    #        1000 classes this requires 1.12 GB and causes a memory error.
    #        Returning only top-5 class logits reduces this to ~5.6 MB.
    # ──────────────────────────────────────────────────────────────────────

    top5_indices = probs.argsort()[::-1][:5].tolist()   # e.g. [208, 176, 207, ...]

    mean_t = torch.tensor(imagenet_mean, dtype=torch.float32).to(device).view(1, 3, 1, 1)
    std_t  = torch.tensor(imagenet_std,  dtype=torch.float32).to(device).view(1, 3, 1, 1)

    class NormalizedResNet(torch.nn.Module):
        """
        Wrapper that:
          1. Applies ImageNet normalization before forwarding.
          2. Returns only top-5 class logits to avoid SHAP memory overflow.

        SHAP allocates (1, H, W, C, num_classes) — with 1000 classes this
        is 1.12 GB. Returning 5 classes reduces it to ~5.6 MB.
        """
        def __init__(self, base_model, class_indices):
            super().__init__()
            self.base_model    = base_model
            self.class_indices = class_indices   # list of 5 class indices

        def forward(self, x):
            x      = (x - mean_t) / std_t
            logits = self.base_model(x)           # (B, 1000)
            return logits[:, self.class_indices]  # (B, 5) — only top-5 classes

    wrapped_model = NormalizedResNet(model, top5_indices).to(device)
    wrapped_model.eval()

    # class_names for the 5 output classes — labeled by ImageNet index
    top5_class_names = [str(i) for i in top5_indices]

    # ──────────────────────────────────────────────────────────────────────
    # 6. Build SHAP Image Explainer
    # ──────────────────────────────────────────────────────────────────────

    shap_image_explainer = ShapExplainer_Image(
        model            = wrapped_model,
        class_names      = top5_class_names,   # only 5 classes
        n_segments       = 50,
        background_color = imagenet_mean,       # ImageNet mean as background
        max_evals        = 500,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 7. explain() — heatmap auto-displayed and saved to user_saves/
    #    class_index=0 → top predicted class (index 0 in the top-5 list)
    # ──────────────────────────────────────────────────────────────────────

    print("\nComputing SHAP values (this may take ~1-2 min)...")
    explanation = shap_image_explainer.explain(
        image       = image_np,     # (224, 224, 3) raw [0, 1]
        class_index = 0,            # 0 = top predicted class in top-5 list
        save_png    = True,
        save_dir    = "user_saves",
    )

    # ──────────────────────────────────────────────────────────────────────
    # 8. Console visualization
    # ──────────────────────────────────────────────────────────────────────

    shap_image_explainer.visualize(
        explanation  = explanation,
        class_index  = 0,
        num_segments = 10,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 9. Verify user_saves/ folder and file were created
    # ──────────────────────────────────────────────────────────────────────

    expected_file = os.path.join("user_saves", f"shap_heatmap_{top5_indices[0]}.png")
    assert os.path.exists("user_saves"),  "user_saves/ folder was not created"
    assert os.path.exists(expected_file), f"Expected file not found: {expected_file}"
    print(f"\n✓ user_saves/ folder created")
    print(f"✓ Heatmap saved: {expected_file}")
    print("\nALL TESTS PASSED ✓")


if __name__ == "__main__":
    test_shap_image_pretrained()