import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import your explainer
from EXACT.explainers.lime_text_explainer import LimeExplainer_Text


# ---------------- Load pretrained sentiment model ----------------

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ---------------- Tokenizer wrapper ----------------

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )


# ---------------- Initialize explainer ----------------

explainer = LimeExplainer_Text(
    model=model,
    tokenizer=tokenize,
    class_names=["negative", "positive"],
    num_samples=3000
)


# ---------------- Test sentences ----------------

test_texts = [
    "This movie was absolutely fantastic and the acting was brilliant but the time duration was terrible and boring",
    "This product is terrible and completely useless but good and safe for health",
    "The story was good but the acting was horrible"
]


# ---------------- Run tests ----------------

for text in test_texts:

    print("\nInput Text:")
    print(text)

    explanation = explainer.explain(text)

    explainer.visualize(explanation)