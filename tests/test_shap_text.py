import torch
from torch import nn
import numpy as np

from EXACT.explainers.shap_text_explainer import ShapExplainer_Text


def test_shap_text():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Sample text input
    # ──────────────────────────────────────────────────────────────────────

    text = "this movie is very good and enjoyable"

    # ──────────────────────────────────────────────────────────────────────
    # 2. Simple vocabulary
    # ──────────────────────────────────────────────────────────────────────

    vocab = {
        "this": 1, "movie": 2, "is": 3, "very": 4,
        "good": 5, "and": 6, "enjoyable": 7, "bad": 8,
        "terrible": 9, "awful": 10, "great": 11, "love": 12,
        "hate": 13, "boring": 14, "excellent": 15,
    }

    vocab_size = len(vocab) + 1
    max_len    = 10

    id2token = {v: k for k, v in vocab.items()}
    id2token[0] = "<PAD>"

    # ──────────────────────────────────────────────────────────────────────
    # 3. Simple tokenizer
    #    Accepts a single string — returns a padded list of token IDs.
    #    (SHAP text explainer calls tokenizer once per text string)
    # ──────────────────────────────────────────────────────────────────────

    def tokenizer(text):
        tokens = text.lower().split()[:max_len]
        ids    = [vocab.get(tok, 0) for tok in tokens]
        ids    = ids + [0] * (max_len - len(ids))
        return ids

    # ──────────────────────────────────────────────────────────────────────
    # 4. Simple text classification model
    # ──────────────────────────────────────────────────────────────────────

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * max_len, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.fc(self.embedding(x))

    model = TextModel().to(device)

    # ──────────────────────────────────────────────────────────────────────
    # 5. Train the model
    #    Small fixed dataset so predictions are meaningful, not random.
    # ──────────────────────────────────────────────────────────────────────

    train_texts  = [
        "this movie is very good and enjoyable",
        "great film love it excellent",
        "very good movie love it",
        "this is terrible and awful",
        "bad movie very boring and hate it",
        "awful film terrible and bad",
    ]
    train_labels = [1, 1, 1, 0, 0, 0]

    X_train = torch.tensor(
        [tokenizer(t) for t in train_texts], dtype=torch.long
    ).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"Training done  —  Final loss: {loss.item():.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Model prediction for the sample text
    # ──────────────────────────────────────────────────────────────────────

    input_t = torch.tensor([tokenizer(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        probs     = torch.softmax(model(input_t), dim=1).cpu().numpy()[0]
        predicted = int(probs.argmax())

    class_names = ["negative", "positive"]

    print(f"\nText            : '{text}'")
    print(f"Predicted class : {class_names[predicted]}")
    print(f"Probabilities   : neg={probs[0]:.4f}  pos={probs[1]:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 7. SHAP Text Explainer
    # ──────────────────────────────────────────────────────────────────────

    shap_text_explainer = ShapExplainer_Text(
        model         = model,
        tokenizer     = tokenizer,
        class_names   = class_names,
        mask_token_id = 0,
        max_seq_len   = max_len,
        nsamples      = 300,
        id2token      = id2token,
    )

    explanation = shap_text_explainer.explain(text=text)

    # ──────────────────────────────────────────────────────────────────────
    # 8. Visualization
    # ──────────────────────────────────────────────────────────────────────

    shap_text_explainer.visualize(
        explanation = explanation,
        class_index = predicted,
        num_tokens  = 7,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 9. Console text plot  (inline token attribution map)
    # ──────────────────────────────────────────────────────────────────────

    shap_text_explainer.text_plot(
        explanation = explanation,
        class_index = predicted,
    )


if __name__ == "__main__":
    test_shap_text()