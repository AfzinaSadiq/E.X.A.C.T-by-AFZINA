import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from EXACT.explainers.lime_tabular_explainer import LimeExplainer_Tabular
from EXACT.utils import predict_proba_fn


# --------------------------------------------------
# Load Iris Dataset
# --------------------------------------------------
data = load_iris()

X = data.data
y = data.target

feature_names = data.feature_names
class_names = data.target_names


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


# --------------------------------------------------
# PyTorch Tabular Model
# --------------------------------------------------
class TabularNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


model = TabularNet()


# --------------------------------------------------
# Training
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):

    optimizer.zero_grad()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    loss.backward()

    optimizer.step()


# --------------------------------------------------
# Evaluate Model Accuracy
# --------------------------------------------------
with torch.no_grad():

    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    outputs = model(test_tensor)

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions.numpy() == y_test).mean()

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# --------------------------------------------------
# Select Sample to Explain
# --------------------------------------------------
sample_index = 0

instance = X_test[sample_index]

true_label = y_test[sample_index]


# --------------------------------------------------
# Model Prediction
# --------------------------------------------------
probs = predict_proba_fn.predict_proba(
    instance.reshape(1, -1),
    model=model,
    mode="classification"
)

predicted_class = np.argmax(probs)

confidence = probs[0][predicted_class]


print("\nSample Index:", sample_index)

print("\nFeature Values")
print("------------------------------")

for name, value in zip(feature_names, instance):
    print(f"{name:20s}: {value:.4f}")


print("\nTrue Class      :", class_names[true_label])
print("Predicted Class :", class_names[predicted_class])
print("Confidence      :", round(confidence * 100, 2), "%")


print("\nPrediction Probabilities")
print("------------------------------")

for i, cls in enumerate(class_names):
    print(f"{cls:15s}: {probs[0][i]:.4f}")


# --------------------------------------------------
# Initialize LIME Tabular Explainer
# --------------------------------------------------
explainer = LimeExplainer_Tabular(
    model=model,
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification"
)


# --------------------------------------------------
# Generate Explanation
# --------------------------------------------------
explanation = explainer.explain(instance)


# --------------------------------------------------
# Print Text Explanation
# --------------------------------------------------
explainer.visualize(explanation, num_features=4)


# --------------------------------------------------
# Plot Explanation (NEW)
# --------------------------------------------------
explainer.plot_explanation(
    explanation,
    num_features=4,
    title="LIME Feature Contributions",
    save_png=True
)