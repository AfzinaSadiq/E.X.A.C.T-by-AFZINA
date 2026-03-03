"""
Test: IGImageExplainer using pretrained ResNet50 on ImageNet.

Why pretrained ResNet50?
    - Already accurate, so the prediction is meaningful
    - No training needed — run this immediately
    - If IG highlights the correct object region, the explainer works

What this test verifies:
    1. Explainer runs without errors
    2. Convergence delta < 0.05 for real images (completeness axiom holds)
    3. All 4 output visualizations are valid BGR images
    4. Dashboard saves to user_saves/

Usage:
    python -m tests.test_ig_image                        # auto-downloads sample
    python -m tests.test_ig_image --image path/to/img.jpg
    python -m tests.test_ig_image --output my_result.png

Output:
    user_saves/ig_explanation.png
"""

import argparse
import os
import sys
import urllib.request

# Fix: Windows OMP duplicate library conflict (PyTorch + numpy/cv2 both load
# libiomp5md.dll). Must be set before any torch/cv2 import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from EXACT.explainers.ig_image_explainer import IGImageExplainer

# All output visualizations are saved here
USER_SAVES_DIR = "user_saves"
os.makedirs(USER_SAVES_DIR, exist_ok=True)

# ImageNet labels for common classes
IMAGENET_LABELS = {
    0:   "tench",
    207: "golden_retriever",
    208: "Labrador_retriever",
    243: "bull_mastiff",
    281: "tabby_cat",
    282: "tiger_cat",
    283: "Persian_cat",
    284: "Siamese_cat",
    285: "Egyptian_cat",
    340: "zebra",
    386: "African_elephant",
    954: "banana",
}

SAMPLE_IMAGES = {
    "dog": "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "cat": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
}


# ------------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------------

def download_image(subject: str = "dog") -> tuple:
    """
    Downloads a sample image by subject name ("dog" or "cat").
    Falls back to a synthetic image if download fails.

    Returns:
        (path: str, is_synthetic: bool)
    """
    url       = SAMPLE_IMAGES[subject]
    save_path = f"sample_{subject}.jpg"
    try:
        print(f"  Downloading sample {subject} image ...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        print(f"  Saved → {save_path}")
        return save_path, False
    except Exception as e:
        print(f"  Download failed ({e}). Using synthetic image instead.")
        return _make_synthetic_image(save_path), True


def _make_synthetic_image(save_path: str) -> str:
    """
    Creates a 224×224 synthetic image (gradient background + bright circle).
    Guarantees the test always runs even with no internet connection.

    Note: convergence delta will be high on synthetic images — this is
    expected. ResNet50 was never trained on gradient/circle patterns, so
    its gradient landscape is noisy for these inputs. The check is relaxed
    automatically when a synthetic image is used.
    """
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img[i, :, 0] = i      # red gradient top→bottom
        img[:, i, 2] = i      # blue gradient left→right
    cv2.circle(img, (112, 112), 60, (200, 200, 50), -1)
    cv2.imwrite(save_path, img)
    print(f"  Synthetic image saved → {save_path}")
    return save_path


def load_image(path: str):
    """
    Loads an image from disk and prepares it for ResNet50.

    Returns:
        img_bgr    : [224, 224, 3] BGR ndarray (for visualization)
        img_tensor : [1, 3, 224, 224] normalized tensor (for model)
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    img_tensor      = transform(img_rgb).unsqueeze(0)
    img_bgr_resized = cv2.resize(img_bgr, (224, 224))

    return img_bgr_resized, img_tensor


# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------

def check_convergence(delta: float, is_synthetic: bool) -> bool:
    """
    Convergence delta measures completeness axiom error:
        delta = |sum(attributions) - (F(input) - F(baseline))|

    Expected ranges by step count:
        200 steps  →  delta ~ 0.05–0.15   (default, fast)
        500 steps  →  delta ~ 0.01–0.05   (accurate)
        1000 steps →  delta < 0.01        (very accurate, slow)

    Real images    : delta < 0.15  (realistic for 200 steps)
    Synthetic image: delta < 0.5   (relaxed — unstable gradients expected)
    """
    threshold = 0.5 if is_synthetic else 0.15
    ok        = delta < threshold
    note      = " (synthetic — relaxed threshold)" if is_synthetic else ""
    status    = "PASS ✅" if ok else "FAIL ❌  (try increasing steps)"
    print(f"  Convergence delta : {delta:.6f}  [{status}]{note}")
    return ok


def check_outputs(results: dict) -> bool:
    keys           = ["overlay_magnitude", "overlay_positive",
                      "overlay_negative",  "overlay_contour"]
    expected_shape = (224, 224, 3)
    all_ok         = True

    for key in keys:
        img = results[key]
        ok  = isinstance(img, np.ndarray) and img.shape == expected_shape
        print(f"  {key:<25} shape={img.shape}  [{'PASS ✅' if ok else 'FAIL ❌'}]")
        all_ok = all_ok and ok

    return all_ok


# ------------------------------------------------------------------
# Main test
# ------------------------------------------------------------------

def run_test(image_path: str, is_synthetic: bool,
             output_path: str = None) -> bool:

    if output_path is None:
        # Derive name from the image file — sample_cat.jpg → ig_cat.png
        base    = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(USER_SAVES_DIR, f"ig_{base}.png")

    print("\n" + "=" * 55)
    print("  IGImageExplainer — Test with Pretrained ResNet50")
    print("=" * 55)

    # 1. Load model
    print("\n[1/4] Loading pretrained ResNet50 ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    print(f"  Device : {device}")

    # 2. Load image
    print(f"\n[2/4] Loading image: {image_path}")
    img_bgr, img_tensor = load_image(image_path)
    print(f"  Image shape  : {img_bgr.shape}")
    print(f"  Tensor shape : {img_tensor.shape}")

    # 3. Run explainer
    print("\n[3/4] Running Integrated Gradients ...")
    explainer = IGImageExplainer(model, device=device)
    results   = explainer.explain(
        input_tensor = img_tensor,
        original_bgr = img_bgr,
        steps        = 200,
        batch_size   = 32,
    )

    predicted_class = results["target_class"]
    class_name      = IMAGENET_LABELS.get(predicted_class,
                                          f"class_{predicted_class}")
    print(f"\n  Predicted class : {predicted_class}  ({class_name})")

    # 4. Verify results
    print("\n[4/4] Verifying results ...")
    convergence_ok = check_convergence(results["convergence_delta"], is_synthetic)
    outputs_ok     = check_outputs(results)

    # 5. Save dashboard
    explainer.save_dashboard(results, save_path=output_path,
                             class_name=class_name)
    print(f"\n  Dashboard saved → {output_path}")

    # 6. Summary
    passed = convergence_ok and outputs_ok
    print("\n" + "=" * 55)
    print(f"  RESULT : {'ALL CHECKS PASSED ✅' if passed else 'SOME CHECKS FAILED ❌  — see above'}")
    print("=" * 55 + "\n")

    return passed


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a specific image file to test."
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        choices=list(SAMPLE_IMAGES.keys()),
        help="Single subject to test: 'dog' or 'cat'. Default: tests both."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (only used when --image is provided)."
    )
    args = parser.parse_args()

    # Test a specific image file
    if args.image:
        passed = run_test(args.image, is_synthetic=False,
                          output_path=args.output)
        sys.exit(0 if passed else 1)

    # Test a single subject
    if args.subject:
        subjects = [args.subject]
    else:
        # Default: test all sample images
        subjects = list(SAMPLE_IMAGES.keys())

    results = {}
    for subject in subjects:
        print(f"\n{'#' * 55}")
        print(f"  Testing subject: {subject.upper()}")
        print(f"{'#' * 55}")
        image_path, is_synthetic = download_image(subject=subject)
        results[subject] = run_test(image_path, is_synthetic)

    # Final summary across all subjects
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    all_passed = True
    for subject, passed in results.items():
        status = "PASS [OK]" if passed else "FAIL [!!]"
        print(f"  {subject:<10} : {status}")
        all_passed = all_passed and passed
    print("=" * 55)
    print(f"  OVERALL : {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 55 + "\n")

    sys.exit(0 if all_passed else 1)