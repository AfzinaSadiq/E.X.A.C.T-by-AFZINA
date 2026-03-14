import numpy as np
import torch
import shap


class ShapExplainer_Text:
    """
    SHAP Text Explainer for EXACT Library  (PyTorch-specific)

    Responsibilities:
        - Generate SHAP explanations for ANY PyTorch text classification model
        - Return per-token SHAP attribution values
        - Provide visualization utilities (text plot, bar plot, summary plot)

    Explainer used:
        KernelExplainer — model-agnostic black-box approach.
        Works with ALL PyTorch text model types:
            - Transformers (BERT, DistilBERT, RoBERTa, etc.)
            - RNN / LSTM / GRU
            - Embedding + Linear (bag-of-words style)
            - Any custom PyTorch nn.Module that accepts token IDs

    How it works for text:
        1. Input text is tokenized into tokens using the provided tokenizer.
        2. A binary masking matrix is built — each row is a coalition where
           some tokens are "present" (original) and some are "absent" (masked).
        3. Absent tokens are replaced with mask_token_id (default: 0 = padding).
        4. The model is evaluated on all coalitions via predict_proba.
        5. KernelExplainer fits a weighted linear model with Shapley-correct
           weights to get per-token SHAP values.

    Notes:
        - Model is automatically set to eval() mode
        - tokenizer must be a callable: text (str) -> token_ids (list[int])
        - max_seq_len controls truncation for long texts
        - All tensor/numpy conversions are handled internally
    """

    def __init__(
        self,
        model,
        tokenizer,
        class_names=None,
        mask_token_id=0,
        max_seq_len=128,
        nsamples=500,
        id2token=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch text classification model.
            Must accept input of shape (batch_size, seq_len) as LongTensor
            (token IDs) and return logits of shape (batch_size, num_classes).

        tokenizer : callable
            A function that converts a raw text string into a list of integer
            token IDs.

            Examples:
                # Hugging Face tokenizer
                tokenizer = lambda text: hf_tokenizer(
                    text,
                    max_length=128,
                    truncation=True,
                    padding='max_length'
                )["input_ids"]

                # Simple whitespace tokenizer with vocab
                tokenizer = lambda text: [vocab[w] for w in text.split()]

            The tokenizer is responsible for padding/truncation if needed.

        class_names : list[str], optional
            Names of output classes (used in visualization only).
            If None, auto-labeled as ['Class_0', 'Class_1', ...]

        mask_token_id : int
            Token ID used to replace "absent" tokens in SHAP coalitions.
            Use 0 for padding-based models.
            Use your tokenizer's [MASK] token ID for BERT-style models
            (typically 103 for BERT, 50264 for RoBERTa).
            Default: 0

        max_seq_len : int
            Maximum number of tokens to explain.
            Longer sequences are truncated to this length.
            Lower values = faster computation. Default: 128.

        nsamples : int
            Number of masked samples KernelExplainer evaluates per explanation.
            Higher -> more accurate but slower.
            Recommended: 200–1000. Default: 500.
        """
        self.model         = model
        self.tokenizer     = tokenizer
        self.class_names   = class_names
        self.mask_token_id = mask_token_id
        self.max_seq_len   = max_seq_len
        self.nsamples      = nsamples
        self.id2token       = id2token

        # Resolve device from model
        self.device = next(model.parameters()).device

        # Always eval mode — critical for correct attributions
        self.model.eval()

    # ──────────────────────────────────────────────────────────────────────
    # Internal: predict proba  (self-contained, text-specific)
    # ──────────────────────────────────────────────────────────────────────

    def _predict_proba(self, input_tensor):
        """
        Run a forward pass on the model and return softmax probabilities.

        This is kept self-contained inside the text explainer because text
        models receive LongTensor (token IDs), not FloatTensor like tabular
        or image models. Using the shared predict_proba_fn from utils would
        incorrectly cast token IDs to float, breaking the embedding lookup.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape: (batch_size, seq_len)  dtype: torch.long
            Already on the correct device.

        Returns
        -------
        np.ndarray   shape: (batch_size, num_classes)  — softmax probabilities
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)             # (batch_size, num_classes)

            if logits.ndim == 2 and logits.shape[-1] == 1:
                # Binary classification with single output neuron
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────
    # Internal: masked prediction function  (used by KernelExplainer)
    # ──────────────────────────────────────────────────────────────────────

    def _make_predict_fn(self, token_ids):
        """
        Build the masked prediction function for a specific input.

        KernelExplainer works by masking subsets of features (tokens here)
        and observing how the model output changes. This function takes a
        binary mask matrix from KernelExplainer and returns model probabilities.

        How masking works for text:
            - mask = 1 → token is PRESENT  (use original token_id)
            - mask = 0 → token is ABSENT   (replace with mask_token_id)

        This is how SHAP evaluates "what happens if this token is missing"
        without actually removing it (which would change sequence length).

        Parameters
        ----------
        token_ids : np.ndarray
            Shape: (seq_len,) — original token IDs for the input text.

        Returns
        -------
        predict_fn : callable
            Accepts binary mask matrix of shape (n_coalitions, seq_len),
            returns probability array of shape (n_coalitions, num_classes).
        """
        def predict_fn(mask_matrix):
            """
            Parameters
            ----------
            mask_matrix : np.ndarray
                Shape: (n_coalitions, seq_len)
                Binary — 1 = keep token, 0 = replace with mask_token_id.

            Returns
            -------
            np.ndarray   shape: (n_coalitions, num_classes)
            """
            n_coalitions = mask_matrix.shape[0]
            seq_len      = len(token_ids)

            # Build masked token ID sequences for all coalitions at once
            # Start with all mask tokens, then fill in present tokens
            masked_inputs = np.full(
                (n_coalitions, seq_len),
                fill_value=self.mask_token_id,
                dtype=np.int64,
            )

            for i in range(n_coalitions):
                present = mask_matrix[i].astype(bool)         # (seq_len,)
                masked_inputs[i, present] = token_ids[present]

            # Convert to LongTensor — must stay as long for embedding lookup
            input_tensor = torch.tensor(
                masked_inputs, dtype=torch.long
            ).to(self.device)                                  # (n_coalitions, seq_len)

            return self._predict_proba(input_tensor)           # (n_coalitions, num_classes)

        return predict_fn

    # ──────────────────────────────────────────────────────────────────────
    # Core explanation logic
    # ──────────────────────────────────────────────────────────────────────

    def explain(self, text):
        """
        Explain a single text input — generate SHAP values per token.

        Each SHAP value represents how much that token pushed the model's
        output above or below the baseline prediction.

        Parameters
        ----------
        text : str
            Raw input text string to explain.

        Returns
        -------
        explanation : dict with keys:
            'shap_values'    : np.ndarray or list[np.ndarray]
                               For multi-class: list of (1, seq_len), one per class.
                               For binary: (1, seq_len).
            'expected_value' : float or list[float]
                               Baseline prediction (model output when all tokens masked).
            'tokens'         : list[str]
                               Token strings corresponding to each SHAP value position.
                               Falls back to token IDs as strings if decode unavailable.
            'token_ids'      : np.ndarray  shape: (seq_len,)
                               Integer token IDs for the input text.
            'text'           : str
                               Original input text.
        """
        # ── Step 1: Tokenize input text ──
        token_ids = self.tokenizer(text)

        # Convert to numpy array and truncate to max_seq_len
        token_ids = np.array(token_ids, dtype=np.int64)[:self.max_seq_len]
        seq_len   = len(token_ids)

        # ── Step 2: Decode tokens to strings for visualization ──
        tokens = self._decode_tokens(token_ids)

        # ── Step 3: Build background for KernelExplainer ──
        # Background = all tokens masked (the "no information" baseline).
        # This represents what the model predicts when it sees nothing.
        background = np.zeros((1, seq_len), dtype=np.float64)  # all tokens absent

        # ── Step 4: Build masked prediction function for this input ──
        predict_fn = self._make_predict_fn(token_ids)

        # ── Step 5: Initialize KernelExplainer and compute SHAP values ──
        # KernelExplainer receives the prediction function and background.
        # It will call predict_fn many times with different binary masks.
        kernel_explainer = shap.KernelExplainer(predict_fn, background)
        shap_values      = kernel_explainer.shap_values(
            np.ones((1, seq_len), dtype=np.float64),   # explain the "all present" state
            nsamples=self.nsamples,
        )

        return {
            "shap_values"    : shap_values,
            "expected_value" : kernel_explainer.expected_value,
            "tokens"         : tokens,
            "token_ids"      : token_ids,
            "text"           : text,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal: token decoding
    # ──────────────────────────────────────────────────────────────────────

    def _decode_tokens(self, token_ids):
        """
        Convert integer token IDs to human-readable token strings.

        Tries three approaches in order:
            1. tokenizer.convert_ids_to_tokens()  — Hugging Face tokenizers
            2. tokenizer.decode() per token       — some custom tokenizers
            3. Fall back to string representation of IDs

        Parameters
        ----------
        token_ids : np.ndarray   shape: (seq_len,)

        Returns
        -------
        list[str]   length: seq_len
        """
       
        # Use reverse vocab lookup if provided
        if self.id2token is not None:
            return [self.id2token.get(int(tid), str(tid)) for tid in token_ids.tolist()]

        # Hugging Face tokenizer
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        # Tokenizer with per-token decode
        if hasattr(self.tokenizer, "decode"):
            return [self.tokenizer.decode([tid]) for tid in token_ids.tolist()]

        # Fallback
        return [str(tid) for tid in token_ids.tolist()]

    # ──────────────────────────────────────────────────────────────────────
    # Raw explanation data  (mirrors LIME's get_explanation_data pattern)
    # ──────────────────────────────────────────────────────────────────────

    def get_explanation_data(self, explanation, class_index=0, num_tokens=10):
        """
        Extract a sorted (token, shap_value) list from explain() output.

        Mirrors the role of LimeExplainer_Tabular.get_explanation_data() and
        ShapExplainer_Tabular.get_explanation_data() so all explainers can be
        used interchangeably downstream.

        Parameters
        ----------
        explanation : dict
            Direct output of explain().

        class_index : int
            Which class to extract SHAP values for. Default: 0.

        num_tokens : int
            How many top tokens to return. Default: 10.

        Returns
        -------
        list of (str, float)
            [(token_string, shap_value), ...] sorted by |shap_value| descending.
        """
        shap_values = explanation["shap_values"]
        tokens      = explanation["tokens"]

        # Normalize to flat 1D array — same logic as tabular explainer
        if isinstance(shap_values, list):
            values = np.array(shap_values[class_index][0]).flatten()
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values = shap_values[0, :, class_index].flatten()
        else:
            values = np.array(shap_values[0]).flatten()

        paired = [(tok, val) for tok, val in zip(tokens, values.tolist())
          if tok != "<PAD>"]
        sorted_ = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        return sorted_[:num_tokens]

    # ──────────────────────────────────────────────────────────────────────
    # Console visualization  (mirrors LIME's visualize pattern)
    # ──────────────────────────────────────────────────────────────────────

    def visualize(self, explanation, class_index=0, num_tokens=10):
        """
        Print SHAP token attributions in readable console format.

        Mirrors the print style of LimeExplainer_Tabular.visualize() and
        ShapExplainer_Tabular.visualize().
        Positive values shown with '+', negative with '-'.

        Parameters
        ----------
        explanation : dict   — direct output of explain()
        class_index : int    — which class to display. Default: 0.
        num_tokens  : int    — how many top tokens to show. Default: 10.

        Returns
        -------
        token_scores : list of (str, float)
            Same as get_explanation_data() — returned for programmatic use.
        """
        token_scores = self.get_explanation_data(
            explanation,
            class_index=class_index,
            num_tokens=num_tokens,
        )

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        print(f"\nSHAP Text Explanation  [Class: {class_label}]")
        print("-" * 42)
        for token, score in token_scores:
            sign = "+" if score >= 0 else "-"
            bar  = "|" * min(int(abs(score) * 80), 20)
            print(f"  {token:<28s}  {sign}  {abs(score):.6f}  {bar}")
        print("-" * 42)

        return token_scores

    # ──────────────────────────────────────────────────────────────────────
    # SHAP plots
    # ──────────────────────────────────────────────────────────────────────

    def text_plot(self, explanation, class_index=0):
        shap_values = explanation["shap_values"]
        tokens      = explanation["tokens"]

        if isinstance(shap_values, list):
            values = np.array(shap_values[class_index][0]).flatten()
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values = shap_values[0, :, class_index].flatten()
        else:
            values = np.array(shap_values[0]).flatten()

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        # Filter out PAD tokens
        pairs   = [(tok, val) for tok, val in zip(tokens, values.tolist())
                if tok != "<PAD>"]
        max_val = max(abs(v) for _, v in pairs) if pairs else 1.0

        print(f"\nSHAP Text Plot  [Class: {class_label}]")
        print("─" * 60)

        token_line = ""
        score_line = ""

        for token, val in pairs:
            bar_len = max(int((abs(val) / max_val) * 4), 0)
            marker  = "+" * bar_len if val > 0.001 else "-" * bar_len if val < -0.001 else ""
            cell    = f"[{marker}{token}{marker}]"
            score   = f" {val:+.3f} "
            width   = max(len(cell), len(score)) + 1
            token_line += cell.center(width)
            score_line += score.center(width)

        print(token_line)
        print(score_line)
        print("─" * 60)
        print("[+++ word +++] = pushes toward this class")
        print("[--- word ---] = pushes away from this class")


    def summary_plot(self, explanation, class_index=0):
        """
        SHAP summary plot — shows SHAP value distribution per token.

        Parameters
        ----------
        explanation : dict   — output of explain()
        class_index : int    — which class to visualize. Default: 0.
        """
        shap_values = explanation["shap_values"]
        tokens      = explanation["tokens"]

        if isinstance(shap_values, list):
            values = np.array(shap_values[class_index])       # (1, seq_len)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values = shap_values[:, :, class_index]           # (1, seq_len)
        else:
            values = np.array(shap_values)                    # (1, seq_len)

        shap.summary_plot(
            values,
            feature_names=tokens,
        )

    def bar_plot(self, explanation, class_index=0):
        """
        SHAP bar plot — shows mean absolute SHAP value per token.

        Parameters
        ----------
        explanation : dict   — output of explain()
        class_index : int    — which class to visualize. Default: 0.
        """
        shap_values = explanation["shap_values"]
        tokens      = explanation["tokens"]

        shap.summary_plot(
            shap_values if not isinstance(shap_values, list)
            else shap_values[class_index],
            feature_names=tokens,
            plot_type="bar",
        )