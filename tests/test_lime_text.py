import torch
from torch import nn
import numpy as np

def test_lime_text():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------- Sample text input -------------------------
    
    text = "This movie is very good and enjoyable"

    # ------------------------- Simple vocabulary (toy example) -------------------------

    vocab = {
        "this" : 1,
        "movie" : 2,
        "is" : 3,
        "very" : 4,
        "good" : 5,
        "and" : 6,
        "enjoyable" : 7
    }

    vocab_size = len(vocab) + 1 # +1 for unknown tokens

    # ------------------------- Simple Tokenizer --------------------------

    def simple_tokenizer(texts):
        encoded = []
        max_len = 10

        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(tok,0) for tok in tokens]

            # Pad / truncate
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))

            encoded.append(ids)

        return torch.tensor(encoded, dtype = torch.long)
    
    # ------------------------- Simple Text Classification Model -------------------------

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*10, 64),
                nn.ReLU(),
                nn.Linear(64,2)
            )
        
        def forward(self, X):
            X = self.embedding(X)
            return self.fc(X)
    
    model = TextModel().to(device)
    model.eval()

    # ------------------------- LIME Text Explainer  -------------------------
    
    from EXACT.explainers.lime_text_explainer import LimeExplainer_Text

    lime_text_explainer = LimeExplainer_Text(
        model = model,
        tokenizer = simple_tokenizer,
        class_names = ["negative", "positive"],
        num_samples = 2000
    )

    explanation = lime_text_explainer.explain(
        text = text,
        top_labels = 1
    )

    # ------------------------- Visualization -------------------------

    lime_text_explainer.visualize(
        explanation = explanation,
        num_features = 6
    )


if __name__ == '__main__':
    test_lime_text()
