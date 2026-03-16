import torch
from torch import nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from EXACT.explainers.shap_image_explainer import ShapExplainer_Image


def test_shap_image():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Dataset — sklearn digits
    #    No download needed, instant load.
    #    1797 images, 8x8 grayscale, 10 classes (digits 0-9)
    #    Upscaled to 16x16 for a slightly better heatmap resolution.
    # ──────────────────────────────────────────────────────────────────────

    print("Loading digits dataset...")

    digits      = load_digits()
    X           = digits.images.astype(np.float32) / 16.0  # (1797, 8, 8), [0,1]
    y           = digits.target
    class_names = [str(i) for i in range(10)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Upscale 8x8 → 16x16 using bilinear interpolation for better heatmap
    def upscale(images_np, size=16):
        t = torch.tensor(images_np[:, np.newaxis, :, :])   # (N, 1, 8, 8)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.squeeze(1).numpy()                        # (N, 16, 16)

    X_train_up = upscale(X_train)                          # (N, 16, 16)
    X_test_up  = upscale(X_test)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Simple CNN
    #    Input:  (B, 1, 16, 16)
    #    Output: (B, 10)
    # ──────────────────────────────────────────────────────────────────────

    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                        # → (16, 8, 8)
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                        # → (32, 4, 4)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = DigitCNN().to(device)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Train — fast, small dataset, converges in ~10 epochs
    # ──────────────────────────────────────────────────────────────────────

    print("Training...")

    X_train_t = torch.tensor(
        X_train_up[:, np.newaxis, :, :], dtype=torch.float32
    ).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1:2d}/15]  Loss: {loss.item():.4f}")

    model.eval()

    # Quick accuracy check
    with torch.no_grad():
        X_test_t  = torch.tensor(
            X_test_up[:, np.newaxis, :, :], dtype=torch.float32
        ).to(device)
        preds     = model(X_test_t).argmax(1).cpu().numpy()
        accuracy  = (preds == y_test).mean() * 100
    print(f"Test accuracy   : {accuracy:.1f}%")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Pick a test sample
    # ──────────────────────────────────────────────────────────────────────

    sample_index = 0
    image        = X_test_up[sample_index]                 # (16, 16) grayscale
    true_label   = y_test[sample_index]

    input_t = torch.tensor(
        image[np.newaxis, np.newaxis, :, :], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        probs     = torch.softmax(model(input_t), dim=1).cpu().numpy()[0]
        predicted = int(probs.argmax())

    print(f"\nTrue digit      : {true_label}")
    print(f"Predicted digit : {predicted}")
    print(f"Confidence      : {probs[predicted]*100:.1f}%")

    # ──────────────────────────────────────────────────────────────────────
    # 5. SHAP Image Explainer
    #    n_segments=10  — small image needs fewer segments
    #    max_evals=200  — enough for accurate results, much faster
    # ──────────────────────────────────────────────────────────────────────

    shap_image_explainer = ShapExplainer_Image(
        model            = model,
        class_names      = class_names,
        n_segments       = 10,
        background_color = "mean",
        max_evals        = 200,         # reduced from 500 — key speedup
    )

    print("\nComputing SHAP values...")
    explanation = shap_image_explainer.explain(image,
                                               class_index = predicted,
                                               save_png    = True,    
                                               save_dir    = "user_saves",
    )

    # ──────────────────────────────────────────────────────────────────────
    # 6. Visualization
    # ──────────────────────────────────────────────────────────────────────

    shap_image_explainer.visualize(
        explanation  = explanation,
        class_index  = predicted,
        num_segments = 10,
    )

if __name__ == "__main__":
    test_shap_image()