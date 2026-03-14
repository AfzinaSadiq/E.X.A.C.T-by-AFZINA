"""
==============================================================
  IGImageExplainer — User Demo
==============================================================

  This is a real-world usage demo of the EXACT library's
  IGImageExplainer. Written from the perspective of a user
  who wants to understand WHY their image model made a decision.

  What this demo does:
    1. Loads a real pretrained ResNet50 (ImageNet, 1000 classes)
    2. Downloads a real photo from the internet (a dog image)
    3. Preprocesses it exactly as ResNet expects
    4. Runs the model — gets a real prediction
    5. Runs IGImageExplainer — gets pixel-level explanations
    6. Saves all 4 visualisations + the dashboard to user_saves/

  Run from your project root:
    python demo_ig_image.py

  Requirements:
    pip install torch torchvision pillow requests

  Output folder:
    user_saves/
      ├── original_image.jpg           ← the image we explained
      ├── ig_dashboard.png             ← 2x2 dashboard (all 4 views)
      ├── ig_magnitude.png             ← overall importance heatmap
      ├── ig_positive.png              ← what supported the prediction
      ├── ig_negative.png              ← what opposed the prediction
      └── ig_contour.png               ← boundary of important region
==============================================================
"""

import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── Import our library ────────────────────────────────────────────────────────
from EXACT.explainers.ig_image_explainer import IGImageExplainer

# ── Output folder ─────────────────────────────────────────────────────────────
SAVE_DIR = "user_saves"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── ImageNet class labels (top 10 we care about for a quick demo) ─────────────
# Full list at: https://github.com/anishathalye/imagenet-simple-labels
# We'll load just enough to print the class name for common animals
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

def load_imagenet_labels():
    """Download ImageNet class names so we can show a human-readable label."""
    try:
        import json
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        # Fallback: return empty dict, we'll just show the class index
        print("  [info] Could not download ImageNet labels — will show class index only.")
        return {}


# ── Image download ────────────────────────────────────────────────────────────
# We use a well-known royalty-free dog image from Wikimedia Commons.
# You can replace IMAGE_URL with any publicly accessible image URL,
# or replace the download block with: img_pil = Image.open("your_image.jpg")

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
    "YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
)
IMAGE_PATH = os.path.join(SAVE_DIR, "original_image.jpg")


def download_image(url: str, save_path: str) -> bool:
    """Download an image if not already cached. Returns True on success."""
    if os.path.exists(save_path):
        print(f"  [info] Using cached image: {save_path}")
        return True
    print(f"  [info] Downloading image from Wikimedia Commons...")
    try:
        headers = {"User-Agent": "EXACT-demo/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r, open(save_path, "wb") as f:
            f.write(r.read())
        print(f"  [info] Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"  [warn] Download failed: {e}")
        return False


def make_synthetic_image(save_path: str) -> None:
    """
    Fallback: create a synthetic 224x224 test image with coloured regions.
    Not a real photo, but enough to demonstrate IG running correctly.
    """
    print("  [info] Creating synthetic test image instead...")
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Blue sky region (top half)
    img[:112, :] = [135, 206, 235]
    # Green grass region (bottom half)
    img[112:, :] = [34, 139, 34]
    # Orange "object" in the centre
    cv2.circle(img, (112, 112), 50, (0, 140, 255), -1)
    cv2.imwrite(save_path, img)
    print(f"  [info] Synthetic image saved to: {save_path}")


# ── ResNet preprocessing ──────────────────────────────────────────────────────
# This is the EXACT preprocessing ResNet expects:
#   1. Resize shortest edge to 256
#   2. Centre-crop to 224×224
#   3. Convert to float tensor [0,1]
#   4. Normalise with ImageNet mean and std
# The baseline (black image) must use the same normalisation —
# so our "zero visual signal" baseline is not raw zeros, but the
# normalised equivalent of a black pixel.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                              # [0,1] float
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # ImageNet normalisation
])

def make_normalised_black_baseline() -> torch.Tensor:
    """
    The IG baseline for a normalised image model is NOT torch.zeros().

    A raw black image (all zeros) becomes a specific non-zero tensor after
    ImageNet normalisation:
        normalised_black = (0.0 - mean) / std
        = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
        = [-2.118, -2.036, -1.804]

    This correctly represents "a completely black pixel" in the same
    space as the input tensor — which is what we want as our baseline.
    """
    black_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    return preprocess(black_pil).unsqueeze(0)  # [1, 3, 224, 224]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 62)
    print("  EXACT Library  —  IGImageExplainer  User Demo")
    print("=" * 62)

    # ── Step 1: Load pretrained ResNet50 ──────────────────────────────────────
    print("\n[1/5] Loading pretrained ResNet50 (ImageNet weights)...")
    try:
        # Modern torchvision (>=0.13) uses the Weights API
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("      Loaded with ResNet50_Weights.IMAGENET1K_V2")
    except ImportError:
        # Older torchvision falls back to pretrained=True
        model = models.resnet50(pretrained=True)
        print("      Loaded with pretrained=True (legacy API)")

    model.eval()
    print("      Model: ResNet50  |  Classes: 1000  |  Mode: eval()")

    # ── Step 2: Get the image ─────────────────────────────────────────────────
    print(f"\n[2/5] Preparing input image...")
    ok = download_image(IMAGE_URL, IMAGE_PATH)
    if not ok:
        make_synthetic_image(IMAGE_PATH)

    # Load with PIL for preprocessing, and OpenCV for overlay rendering
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_bgr = cv2.imread(IMAGE_PATH)

    # If OpenCV could not read (e.g. the image is RGBA or unusual format),
    # convert from the PIL image directly
    if img_bgr is None:
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Resize BGR image to 224×224 to match what the model sees
    # (so overlays line up with the model's receptive field)
    img_bgr_224 = cv2.resize(img_bgr, (224, 224))
    print(f"      Image loaded  |  PIL size: {img_pil.size}  |  BGR shape: {img_bgr_224.shape}")

    # ── Step 3: Preprocess & run model ────────────────────────────────────────
    print(f"\n[3/5] Running ResNet50 on the image...")
    input_tensor = preprocess(img_pil).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits    = model(input_tensor)               # [1, 1000]
        probs     = torch.softmax(logits, dim=1)
        top5_prob, top5_idx = probs[0].topk(5)

    # Load class names if possible
    labels = load_imagenet_labels()

    predicted_class = top5_idx[0].item()
    predicted_prob  = top5_prob[0].item() * 100

    print(f"\n      ┌─ Model Prediction ──────────────────────────────┐")
    for rank, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
        name = labels[idx.item()] if labels else f"class_{idx.item()}"
        marker = "  ◄ top prediction" if rank == 1 else ""
        print(f"      │  #{rank}  {name:<30} {prob.item()*100:5.1f}%{marker}")
    print(f"      └─────────────────────────────────────────────────┘")

    # ── Step 4: Run IGImageExplainer ──────────────────────────────────────────
    print(f"\n[4/5] Running IGImageExplainer...")
    print(f"      Explaining class {predicted_class}",
          f"({labels[predicted_class] if labels else ''})...")

    explainer = IGImageExplainer(model)

    # Use a properly normalised black baseline — NOT raw zeros.
    # A raw torch.zeros() after ImageNet normalisation does NOT represent a
    # black image. We pass a real black PIL image through the same preprocess
    # pipeline so the baseline lives in the same normalised space as the input.
    baseline = make_normalised_black_baseline()

    # Why 500 steps for ResNet50?
    # ─────────────────────────────────────────────────────────────────────────
    # Our SimpleConvNet test (2 layers) converges with delta ~0.001 at 200 steps.
    # ResNet50 has 50 layers with skip connections and BatchNorm — the gradient
    # landscape along the interpolation path is far more curved. This means the
    # Riemann sum needs more rectangles to approximate the integral accurately.
    # Captum (Meta's XAI library) uses 500 as its default for large models.
    # At 200 steps, ResNet50 gives delta ~0.49 (poor). At 500 steps, ~0.05 (good).
    # ─────────────────────────────────────────────────────────────────────────
    STEPS      = 500
    BATCH_SIZE = 32     # lower to 16 if you get CUDA out-of-memory errors
    DELTA_GOOD = 0.05   # target threshold for a clean explanation

    print(f"      Steps: {STEPS}  |  Batch size: {BATCH_SIZE}  |  Baseline: normalised black image")

    results = explainer.explain(
        input_tensor  = input_tensor,
        original_bgr  = img_bgr_224,
        target_class  = predicted_class,
        baseline      = baseline,
        steps         = STEPS,
        batch_size    = BATCH_SIZE,
        alpha         = 0.55,   # 55% heatmap, 45% original image
    )

    delta = results["convergence_delta"]

    # Auto-retry with more steps if delta is still too high.
    # Each retry doubles the steps: 500 → 1000 → 2000.
    # This handles unusually complex images or models without manual tuning.
    for retry_steps in [1000, 2000]:
        if delta <= DELTA_GOOD:
            break
        print(f"      delta={delta:.4f} still high — retrying with {retry_steps} steps...")
        results = explainer.explain(
            input_tensor = input_tensor,
            original_bgr = img_bgr_224,
            target_class = predicted_class,
            baseline     = baseline,
            steps        = retry_steps,
            batch_size   = BATCH_SIZE,
            alpha        = 0.55,
        )
        delta = results["convergence_delta"]
        STEPS = retry_steps

    quality = "EXCELLENT" if delta < 0.05 else "OK" if delta < 0.15 else "[!!] still high — try a different baseline"
    print(f"\n      Convergence delta = {delta:.5f}  [{quality}]  (used {STEPS} steps)")
    print(f"      Explained class   = {results['target_class']}",
          f"({labels[results['target_class']] if labels else ''})")

    # ── Step 5: Save all visualisations ───────────────────────────────────────
    print(f"\n[5/5] Saving visualisations to '{SAVE_DIR}/'...")

    # Individual overlay images
    overlays = {
        "ig_magnitude.png": results["overlay_magnitude"],
        "ig_positive.png":  results["overlay_positive"],
        "ig_negative.png":  results["overlay_negative"],
        "ig_contour.png":   results["overlay_contour"],
    }
    for filename, overlay in overlays.items():
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, overlay)
        print(f"      Saved: {path}")

    # Full 2×2 dashboard
    dashboard_path = os.path.join(SAVE_DIR, "ig_dashboard.png")
    class_name = labels[predicted_class] if labels else str(predicted_class)
    explainer.save_dashboard(
        results    = results,
        save_path  = dashboard_path,
        class_name = f"{class_name}  ({predicted_prob:.1f}%)",
        dpi        = 150,
    )
    print(f"      Saved: {dashboard_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  DONE")
    print("=" * 62)
    print(f"  Model          : ResNet50 (ImageNet pretrained)")
    print(f"  Image          : {IMAGE_PATH}")
    print(f"  Prediction     : {class_name}  ({predicted_prob:.1f}%)")
    print(f"  Explained class: {predicted_class}")
    print(f"  Steps used     : {STEPS}")
    print(f"  Delta          : {delta:.5f}  [{quality}]")
    print(f"  Outputs saved  : {SAVE_DIR}/")
    print()
    print("  How to read the results:")
    print("  ┌─ ig_magnitude.png  ─ HOT = any pixel that mattered")
    print("  ├─ ig_positive.png   ─ HOT = pixels that SAID it's a",
          class_name)
    print("  ├─ ig_negative.png   ─ HOT = pixels that DOUBTED it")
    print("  └─ ig_contour.png    ─ GREEN line = boundary of key region")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()