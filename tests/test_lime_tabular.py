import torch
from torch import nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from EXACT.utils import predict_proba_fn
from EXACT.explainers.lime_tabular_explainer import LimeExplainer_Tabular


def test_lime_tabular():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1. Load predefined dataset (Iris)
    # --------------------------------------------------
    iris = load_iris()
    X = iris.data                      # shape (150, 4)
    y = iris.target                    # shape (150,)
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # 2. Simple PyTorch Tabular Model
    # --------------------------------------------------
    class IrisModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )

        def forward(self, x):
            return self.net(x)

    model = IrisModel().to(device)
    model.eval()  # training not required for LIME demo

    # --------------------------------------------------
    # 3. Choose sample to explain
    # --------------------------------------------------
    sample_index = 0
    instance = X_test[sample_index]
    true_label = y_test[sample_index]

    print("\n================ SAMPLE INFORMATION ================")
    print("Sample index:", sample_index)

    print("\nFeature values of the flower:")
    for name, value in zip(feature_names, instance):
        print(f"{name}: {value:.2f}")

    # --------------------------------------------------
    # 4. Model prediction for this sample
    # --------------------------------------------------
    probs = predict_proba_fn.predict_proba(
        instance.reshape(1, -1),
        model=model
    )

    predicted_class = probs.argmax(axis=1)[0]

    print("\nTrue class:", class_names[true_label])
    print("Predicted class:", class_names[predicted_class])
    print("Prediction probabilities:", probs)

    # --------------------------------------------------
    # 5. LIME Tabular Explainer
    # --------------------------------------------------
    lime_tabular_explainer = LimeExplainer_Tabular(
        model=model,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )

    explanation = lime_tabular_explainer.explain(
        instance=instance,
        top_labels=1
    )

    # --------------------------------------------------
    # 6. LIME Explanation Output
    # --------------------------------------------------
    lime_tabular_explainer.visualize(
        explanation=explanation,
        num_features=4
    )


if __name__ == "__main__":
    test_lime_tabular()

