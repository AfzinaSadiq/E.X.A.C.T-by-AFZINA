import torch
import shap
from torch import nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from EXACT.explainers.shap_tabular_explainer import ShapExplainer_Tabular


def test_shap_tabular():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1. Dataset  (Iris — same as LIME test)
    # --------------------------------------------------
    iris          = load_iris()
    X             = iris.data.astype(np.float32)    # (150, 4)
    y             = iris.target                      # (150,)
    feature_names = list(iris.feature_names)
    class_names   = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # 2. Simple PyTorch tabular model  (same as LIME test)
    # --------------------------------------------------
    class IrisModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    model = IrisModel().to(device)

    # --------------------------------------------------
    # 3. Train the model
    # --------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"Training done  —  Final loss: {loss.item():.4f}")

    # --------------------------------------------------
    # 4. Background data
    #    Small representative sample from training set.
    #    SHAP uses this as the baseline reference distribution.
    # --------------------------------------------------
    np.random.seed(42)
    bg_indices      = np.random.choice(len(X_train), size=50, replace=False)
    background_data = X_train[bg_indices]           # (50, 4)

    # --------------------------------------------------
    # 5. Sample to explain  (same index as LIME test)
    # --------------------------------------------------
    sample_index = 0
    instance     = X_test[sample_index]             # (4,)
    true_label   = y_test[sample_index]

    print("\n================ SAMPLE INFORMATION ================")
    print(f"Sample index : {sample_index}")
    for name, value in zip(feature_names, instance):
        print(f"  {name}: {value:.2f}")

    # --------------------------------------------------
    # 6. Model prediction for this sample
    # --------------------------------------------------
    input_t = torch.tensor(instance, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits    = model(input_t)
        probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted = int(probs.argmax())

    print(f"\nTrue class      : {class_names[true_label]}")
    print(f"Predicted class : {class_names[predicted]}")
    print(f"Probabilities   : {[f'{p:.4f}' for p in probs]}")

    # --------------------------------------------------
    # 7. Test DeepExplainer  (default, recommended)
    # --------------------------------------------------
    print("\n\n================ EXPLAINER: DEEP ================")

    shap_explainer = ShapExplainer_Tabular(
        model           = model,
        background_data = background_data,
        feature_names   = feature_names,
        class_names     = class_names,
        explainer_type  = "deep",
    )

    explanation = shap_explainer.explain(instance)

    # Console visualization  (mirrors LIME's visualize output)
    shap_explainer.visualize(
        explanation    = explanation,
        instance_index = 0,
        class_index    = predicted,
        num_features   = 4,
    )

    # --------------------------------------------------
    # 8. Test GradientExplainer
    # --------------------------------------------------
    print("\n\n================ EXPLAINER: GRADIENT ================")

    shap_explainer_grad = ShapExplainer_Tabular(
        model           = model,
        background_data = background_data,
        feature_names   = feature_names,
        class_names     = class_names,
        explainer_type  = "gradient",
    )

    explanation_grad = shap_explainer_grad.explain(instance)

    shap_explainer_grad.visualize(
        explanation    = explanation_grad,
        instance_index = 0,
        class_index    = predicted,
        num_features   = 4,
    )

    # --------------------------------------------------
    # 9. Test KernelExplainer  (slowest, model-agnostic)
    # --------------------------------------------------
    print("\n\n================ EXPLAINER: KERNEL ================")

    shap_explainer_kernel = ShapExplainer_Tabular(
        model           = model,
        background_data = background_data,
        feature_names   = feature_names,
        class_names     = class_names,
        explainer_type  = "kernel",
    )

    explanation_kernel = shap_explainer_kernel.explain(instance, nsamples=200)

    shap_explainer_kernel.visualize(
        explanation    = explanation_kernel,
        instance_index = 0,
        class_index    = predicted,
        num_features   = 4,
    )

    # --------------------------------------------------
    # 10. SHAP plots  (comment out if running headless)
    # --------------------------------------------------
    print("\n\n================ SHAP PLOTS ================")
    print("Generating waterfall plot for DeepExplainer...")
    shap_explainer.waterfall_plot(
        explanation    = explanation,
        data           = instance,
        instance_index = 0,
        class_index    = predicted,
        save_png       = True,
        save_dir       = "user_saves",
    )

    print("Generating summary plot for DeepExplainer...")
    explanation_batch = shap_explainer.explain(X_test)
    shap_explainer.summary_plot(
        explanation = explanation_batch,
        data        = X_test,
        class_index = predicted,
        save_png       = True,
        save_dir       = "user_saves",
    )

    print("Generating force plot  [DEEP]...")
    shap_explainer.force_plot(
        explanation    = explanation,
        data           = instance,
        instance_index = 0,
        class_index    = predicted,
        save_png       = True,
        save_dir       = "user_saves",
    )

    print("Generating bar plot  [DEEP]  on full X_test...")
    shap.summary_plot(
        explanation_batch["shap_values"],
        X_test,
        feature_names = feature_names,
        plot_type     = "bar",
        show          = True,
        save_png       = True,
        save_dir       = "user_saves",
    )

if __name__ == "__main__":
    test_shap_tabular()