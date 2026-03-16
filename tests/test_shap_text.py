import torch
from torch import nn
import numpy as np

from EXACT.explainers.shap_text_explainer import ShapExplainer_Text


def test_shap_text():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Sample text inputs
    # ──────────────────────────────────────────────────────────────────────

    text_positive = "this movie is very good and enjoyable"
    text_negative = "this film is terrible and awful boring"

    # ──────────────────────────────────────────────────────────────────────
    # 2. Simple vocabulary
    # ──────────────────────────────────────────────────────────────────────

    vocab = {
        "this": 1, "movie": 2, "film": 3, "is": 4, "very": 5,
        "good": 6, "and": 7, "enjoyable": 8, "bad": 9,
        "terrible": 10, "awful": 11, "great": 12, "love": 13,
        "hate": 14, "boring": 15, "excellent": 16,
    }

    vocab_size = len(vocab) + 1   # +1 for PAD index 0
    max_len    = 10

    id2token = {v: k for k, v in vocab.items()}
    id2token[0] = "<PAD>"

    # ──────────────────────────────────────────────────────────────────────
    # 3. Tokenizer — returns padded list of token IDs
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

    torch.manual_seed(42)
    model = TextModel().to(device)

    # ──────────────────────────────────────────────────────────────────────
    # 5. Train the model on a small fixed dataset
    # ──────────────────────────────────────────────────────────────────────

    train_texts = [
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
    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"Training done  —  Final loss: {loss.item():.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Confirm model predictions before explaining
    # ──────────────────────────────────────────────────────────────────────

    class_names = ["negative", "positive"]

    for text, expected_class in [(text_positive, 1), (text_negative, 0)]:
        input_t = torch.tensor([tokenizer(text)], dtype=torch.long).to(device)
        with torch.no_grad():
            probs     = torch.softmax(model(input_t), dim=1).cpu().numpy()[0]
            predicted = int(probs.argmax())

        print(f"\nText            : '{text}'")
        print(f"Predicted class : {class_names[predicted]}")
        print(f"Probabilities   : neg={probs[0]:.4f}  pos={probs[1]:.4f}")

        assert predicted == expected_class, (
            f"Model prediction sanity check failed: "
            f"expected {class_names[expected_class]}, got {class_names[predicted]}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 7. Build the SHAP Text Explainer
    # ──────────────────────────────────────────────────────────────────────

    explainer = ShapExplainer_Text(
        model         = model,
        tokenizer     = tokenizer,
        class_names   = class_names,
        mask_token_id = 0,
        max_seq_len   = max_len,
        nsamples      = 300,
        id2token      = id2token,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 8. Test explain() output structure
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: explain() output structure")
    print("="*60)

    explanation = explainer.explain(text=text_positive)

    assert "shap_values"    in explanation, "Missing key: shap_values"
    assert "expected_value" in explanation, "Missing key: expected_value"
    assert "tokens"         in explanation, "Missing key: tokens"
    assert "token_ids"      in explanation, "Missing key: token_ids"
    assert "text"           in explanation, "Missing key: text"

    assert explanation["text"] == text_positive
    assert len(explanation["tokens"]) == max_len, (
        f"Expected {max_len} tokens, got {len(explanation['tokens'])}"
    )
    assert explanation["token_ids"].shape == (max_len,), (
        f"Expected token_ids shape ({max_len},), got {explanation['token_ids'].shape}"
    )

    print("✓ explain() keys and shapes are correct")

    # ──────────────────────────────────────────────────────────────────────
    # 9. Test SHAP values shape for both classes
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: SHAP values shape for class_index=0 and class_index=1")
    print("="*60)

    for class_idx in [0, 1]:
        values = explainer._extract_values(explanation["shap_values"], class_idx)
        assert values.shape == (max_len,), (
            f"class_index={class_idx}: expected shape ({max_len},), got {values.shape}"
        )
        assert not np.all(values == 0), (
            f"class_index={class_idx}: all SHAP values are zero — model may not be trained"
        )
        print(f"✓ class_index={class_idx}: shape={values.shape}  "
              f"min={values.min():.4f}  max={values.max():.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 10. Test get_explanation_data() — both classes, PAD filtering
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: get_explanation_data() — PAD filtering and return format")
    print("="*60)

    for class_idx in [0, 1]:
        data = explainer.get_explanation_data(
            explanation, class_index=class_idx, num_tokens=7
        )

        assert isinstance(data, list),      "get_explanation_data must return a list"
        assert len(data) <= 7,              f"Expected ≤7 items, got {len(data)}"
        assert all(isinstance(tok, str) and isinstance(val, float)
                   for tok, val in data),   "Each item must be (str, float)"
        assert all(tok != "<PAD>" for tok, _ in data), \
            "PAD tokens must be filtered from get_explanation_data output"

        print(f"✓ class_index={class_idx}: {len(data)} tokens returned, no PAD tokens")
        for tok, val in data:
            print(f"    {tok:<20s}  {val:+.6f}")

    # ──────────────────────────────────────────────────────────────────────
    # 11. Test visualize() — both classes
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: visualize() — class_index=0 (negative)")
    print("="*60)

    scores_neg = explainer.visualize(explanation, class_index=0, num_tokens=7)
    assert isinstance(scores_neg, list), "visualize() must return a list"
    assert all(tok != "<PAD>" for tok, _ in scores_neg), \
        "PAD tokens must not appear in visualize() output"
    print("✓ visualize() class_index=0 passed")

    print("\n" + "="*60)
    print("TEST: visualize() — class_index=1 (positive)")
    print("="*60)

    scores_pos = explainer.visualize(explanation, class_index=1, num_tokens=7)
    assert isinstance(scores_pos, list), "visualize() must return a list"
    assert all(tok != "<PAD>" for tok, _ in scores_pos), \
        "PAD tokens must not appear in visualize() output"
    print("✓ visualize() class_index=1 passed")

    # ──────────────────────────────────────────────────────────────────────
    # 12. Test text_plot() — does not crash, filters PAD
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: text_plot()")
    print("="*60)

    explainer.text_plot(explanation, class_index=1)
    print("✓ text_plot() ran without errors")

    # ──────────────────────────────────────────────────────────────────────
    # 13. Test explain() on a negative-sentiment text
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: explain() on negative-sentiment text")
    print("="*60)

    explanation_neg = explainer.explain(text=text_negative)
    data_neg = explainer.get_explanation_data(
        explanation_neg, class_index=0, num_tokens=7
    )

    assert len(data_neg) > 0, "Expected non-empty explanation for negative text"
    assert all(tok != "<PAD>" for tok, _ in data_neg), \
        "PAD tokens found in negative text explanation"

    explainer.visualize(explanation_neg, class_index=0, num_tokens=7)
    explainer.text_plot(explanation_neg, class_index=0)
    print("✓ Negative text explanation passed")

    # ──────────────────────────────────────────────────────────────────────
    # 14. Test ValueError on empty input
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("TEST: ValueError on empty tokenizer output")
    print("="*60)

    # Tokenizer returns all zeros for unknown words — but an empty string
    # returns a fully-padded sequence. We test with a truly empty tokenizer
    # by monkey-patching for this one case.
    original_tokenizer = explainer.tokenizer
    explainer.tokenizer = lambda text: []   # force empty output

    try:
        explainer.explain("anything")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"✓ ValueError raised correctly: {e}")
    finally:
        explainer.tokenizer = original_tokenizer   # restore

    # ──────────────────────────────────────────────────────────────────────
    # All tests passed
    # ──────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    test_shap_text()