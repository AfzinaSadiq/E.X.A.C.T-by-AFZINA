import numpy as np
import torch
import shap
import os

class ShapExplainer_Tabular:
    """
    SHAP Tabular Explainer for EXACT Library  (PyTorch-specific)

    Responsibilities:
        - Generate SHAP explanations for PyTorch tabular models
        - Return per-feature SHAP attribution values
        - Provide visualization utilities (summary, bar, waterfall, force)

    Supported explainer types (via shap library):
        'deep'      ->  DeepExplainer      : Best default for PyTorch neural networks.
                                             Uses DeepLIFT + SHAP theory internally.
                                             Fast, backpropagation-based.
                                             Note: may fail on BatchNorm/Dropout layers —
                                             auto-falls back to KernelExplainer if so.

        'gradient'  ->  GradientExplainer  : Gradient x input approach.
                                             Good alternative for deep networks.
                                             More stable than DeepExplainer on some
                                             architectures (e.g. with BatchNorm).

        'kernel'    ->  KernelExplainer    : Model-agnostic black-box fallback.
                                             Works on any model but slowest.
                                             Uses predict_proba wrapper internally.

    Supported model interfaces:
        - Plain tensor output:     model(x) → logits tensor
        - Tuple output:            model(x) → (logits, ...)  first element used
        - HuggingFace-style:       model(x) → output with .logits attribute

    Notes:
        - Model is automatically set to eval() mode.
        - background_data is SHAP's reference distribution (equiv. to LIME's training_data).
        - All tensor/numpy conversions are handled internally.
        - expected_value is computed consistently as raw model output (pre-softmax)
          for deep/kernel, and averaged logits for gradient.

    Limitations:
        - Input must be float-compatible tabular features: shape (batch_size, num_features).
        - CNN/RNN/Transformer-based tabular models may fail with DeepExplainer;
          use explainer_type='kernel' for those.
        - Binary single-output models (1 neuron) always use class_index=0.
    """

    SUPPORTED_EXPLAINERS = ("deep", "gradient", "kernel")

    def __init__(
        self,
        model,
        background_data,
        feature_names=None,
        class_names=None,
        explainer_type="deep",
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch tabular model.
            Expected input shape: (batch_size, num_features) as FloatTensor.
            Output shape: (batch_size, num_classes) logits.

        background_data : np.ndarray or torch.Tensor
            Reference distribution used by SHAP as the baseline.
            Shape: (n_background_samples, num_features).

            Represents the "average" model input — the neutral reference against
            which each feature's contribution is measured.

            Recommended: 50–200 rows from training data.
            Tip: K-Means cluster centers of your training set work well.

        feature_names : list[str], optional
            Names of input features.
            If None, auto-labeled as Feature_0, Feature_1, ...

        class_names : list[str], optional
            Names of output classes (used in visualization only).

        explainer_type : str
            Which SHAP backend to use: 'deep' | 'gradient' | 'kernel'.
            Default: 'deep'.
            Auto-falls back to 'kernel' if the chosen explainer fails to init.
        """
        if explainer_type not in self.SUPPORTED_EXPLAINERS:
            raise ValueError(
                f"[ShapExplainer_Tabular] Invalid explainer_type '{explainer_type}'. "
                f"Choose from: {self.SUPPORTED_EXPLAINERS}"
            )

        self.model          = model
        self.feature_names  = feature_names
        self.class_names    = class_names
        self.explainer_type = explainer_type

        # Resolve device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Always eval mode — critical for correct, deterministic attributions
        self.model.eval()

        # Prepare background in both formats:
        #   background_tensor → used by DeepExplainer and GradientExplainer
        #   background_np     → used by KernelExplainer
        if isinstance(background_data, torch.Tensor):
            self.background_tensor = background_data.float().to(self.device)
            self.background_np     = background_data.detach().cpu().numpy().astype(np.float32)
        else:
            self.background_np     = np.array(background_data, dtype=np.float32)
            self.background_tensor = torch.tensor(
                self.background_np, dtype=torch.float32
            ).to(self.device)

        # Initialize the SHAP explainer
        self.explainer      = self._init_explainer()
        self.expected_value = self._compute_expected_value()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: logit extraction  (handles all model output formats)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_logits(self, output):
        """
        Extract a plain logit tensor from any model output format.

        Handles:
            - Plain torch.Tensor        (most custom PyTorch models)
            - Object with .logits attr  (HuggingFace ModelOutput)
            - Tuple / list              (first element assumed to be logits)

        Parameters
        ----------
        output : any   — raw model output

        Returns
        -------
        torch.Tensor   shape: (batch_size, num_classes)

        Raises
        ------
        RuntimeError   if output format is not recognised.
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "logits"):
            return output.logits

        if isinstance(output, (tuple, list)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                return output[0]

        raise RuntimeError(
            f"Unrecognised model output type: {type(output)}. "
            "Model must return a tensor, a tuple whose first element is a tensor, "
            "or an object with a .logits attribute."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: prediction wrapper  (used only by KernelExplainer)
    # ─────────────────────────────────────────────────────────────────────

    def _predict(self, X):
        """
        Prediction wrapper for KernelExplainer.

        KernelExplainer is model-agnostic — it needs a plain callable that
        accepts numpy input and returns numpy probabilities.

        DeepExplainer and GradientExplainer receive the model directly
        and do NOT use this wrapper — they work via backpropagation.

        Parameters
        ----------
        X : np.ndarray   shape: (n_samples, num_features)

        Returns
        -------
        np.ndarray   shape: (n_samples, num_classes) — softmax probabilities.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Always move to the correct device — fixes crash on GPU models
        X = X.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            logits = self._extract_logits(output)

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            if logits.ndim == 2 and logits.shape[-1] == 1:
                # Binary single-output neuron → sigmoid
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: SHAP explainer initialization
    # ─────────────────────────────────────────────────────────────────────

    def _init_explainer(self):
        """
        Initialize the correct SHAP explainer for the selected type.

        Falls back to KernelExplainer automatically if Deep or Gradient
        initialization fails (e.g. unsupported layer types like LSTM).

        Returns
        -------
        shap explainer object
        """
        try:
            if self.explainer_type == "deep":
                # DeepExplainer: takes the model and background tensor directly.
                # Internally uses DeepLIFT propagation rules.
                return shap.DeepExplainer(self.model, self.background_tensor)

            elif self.explainer_type == "gradient":
                # GradientExplainer: takes the model and background tensor.
                # Uses gradient x input approach, averaged over background samples.
                return shap.GradientExplainer(self.model, self.background_tensor)

            elif self.explainer_type == "kernel":
                # KernelExplainer: takes prediction function and background numpy.
                # Model-agnostic — uses the _predict wrapper above.
                return shap.KernelExplainer(self._predict, self.background_np)

        except Exception as e:
            print(
                f"[ShapExplainer_Tabular] WARNING: '{self.explainer_type}' explainer "
                f"failed to initialize ({e}). Falling back to KernelExplainer."
            )
            self.explainer_type = "kernel"
            return shap.KernelExplainer(self._predict, self.background_np)

    # ─────────────────────────────────────────────────────────────────────
    # Internal: expected value computation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_expected_value(self):
        """
        Compute the baseline expected value for all explainer types.

        DeepExplainer and KernelExplainer expose .expected_value directly.
        GradientExplainer does not — computed manually by running background
        data through the model and averaging the raw logits.

        Using raw logits (not softmax) keeps the baseline consistent with
        what DeepExplainer and KernelExplainer expose internally.

        Returns
        -------
        float or list[float]
            For multi-class: list of per-class baseline values.
            For binary single-output: single float.
        """
        if self.explainer_type in ("deep", "kernel"):
            return self.explainer.expected_value

        # GradientExplainer: compute manually from background data using raw logits
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.background_tensor)      # (N, num_classes)
            logits = self._extract_logits(output)            # (N, num_classes)
            mean   = logits.mean(dim=0).cpu().numpy()        # (num_classes,)

        return mean.tolist()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: SHAP value extraction  (shared by all output methods)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_values(self, shap_values, instance_index, class_index):
        """
        Normalise the shap_values structure to a flat 1D numpy array
        for a given instance and class index.

        KernelExplainer / DeepExplainer / GradientExplainer return
        different structures depending on model output:
            - list of arrays (one per class): multi-class
            - 3D array (n_samples, num_features, n_classes): some SHAP versions
            - 2D array (n_samples, num_features): binary or single-output

        Parameters
        ----------
        shap_values    : list[np.ndarray] or np.ndarray
        instance_index : int
        class_index    : int

        Returns
        -------
        np.ndarray   shape: (num_features,)
        """
        if isinstance(shap_values, list):
            # list[array]: one array per class, each shape (n_samples, num_features)
            arr = np.array(shap_values[class_index][instance_index])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, num_features, n_classes)
            arr = shap_values[instance_index, :, class_index]
        else:
            # (n_samples, num_features) — binary or single-output
            arr = np.array(shap_values[instance_index])

        return arr.flatten()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: expected value extraction
    # ─────────────────────────────────────────────────────────────────────

    def _extract_expected_value(self, expected_value, class_index):
        """
        Extract a scalar expected value for a given class index.

        Parameters
        ----------
        expected_value : float or list[float] or np.ndarray
        class_index    : int

        Returns
        -------
        float
        """
        if isinstance(expected_value, (list, np.ndarray)):
            return float(expected_value[class_index])
        return float(expected_value)

    # ─────────────────────────────────────────────────────────────────────
    # Core explanation logic
    # ─────────────────────────────────────────────────────────────────────

    def explain(self, data, nsamples=500):
        """
        Generate SHAP values for one or more tabular samples.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input sample(s) to explain.
            Shape: (num_features,) for single sample
                   (n_samples, num_features) for batch.

        nsamples : int
            Number of samples for KernelExplainer approximation.
            Ignored for 'deep' and 'gradient' explainers.
            Higher → more accurate but slower. Default: 500.

        Returns
        -------
        explanation : dict with keys:
            'shap_values'    : np.ndarray or list[np.ndarray]
                               For multi-class: list of (n_samples, num_features),
                               one array per class.
                               For binary single-output: (n_samples, num_features).
            'expected_value' : float or list[float]
                               Baseline prediction (model output on background).
                               For multi-class: list, one per class.
            'feature_names'  : list[str] or None
            'explainer_type' : str
        """
        # Normalize input to both tensor and numpy
        if isinstance(data, torch.Tensor):
            data_tensor = data.float().to(self.device)
            data_np     = data.detach().cpu().numpy().astype(np.float32)
        else:
            data_np     = np.array(data, dtype=np.float32)
            data_tensor = torch.tensor(data_np, dtype=torch.float32).to(self.device)

        # Ensure 2D — (1, num_features) for single sample
        if data_np.ndim == 1:
            data_np     = data_np[np.newaxis, :]
            data_tensor = data_tensor.unsqueeze(0)

        # Compute SHAP values using the appropriate explainer
        if self.explainer_type == "kernel":
            shap_values = self.explainer.shap_values(data_np, nsamples=nsamples)
        else:
            # DeepExplainer and GradientExplainer work directly with tensors
            shap_values = self.explainer.shap_values(data_tensor)

        return {
            "shap_values"    : shap_values,
            "expected_value" : self.expected_value,
            "feature_names"  : self.feature_names,
            "explainer_type" : self.explainer_type,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Raw explanation data  (mirrors LIME's get_explanation_data pattern)
    # ─────────────────────────────────────────────────────────────────────

    def get_explanation_data(self, explanation, instance_index=0, class_index=0, num_features=10):
        """
        Extract a sorted (feature_name, shap_value) list from explain() output.

        Mirrors the role of LimeExplainer_Tabular.get_explanation_data() so
        both explainers can be used interchangeably downstream.

        Parameters
        ----------
        explanation    : dict   — direct output of explain()
        instance_index : int    — which sample in the batch to extract. Default: 0.
        class_index    : int    — which class to extract SHAP values for. Default: 0.
        num_features   : int    — how many top features to return. Default: 10.

        Returns
        -------
        list of (str, float)
            [(feature_name, shap_value), ...] sorted by |shap_value| descending.
        """
        values = self._extract_values(
            explanation["shap_values"], instance_index, class_index
        )

        num_total = len(values)
        names     = (
            list(self.feature_names)
            if self.feature_names is not None
            else [f"Feature_{i}" for i in range(num_total)]
        )

        paired  = list(zip(names, values.tolist()))
        sorted_ = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        return sorted_[:num_features]

    # ─────────────────────────────────────────────────────────────────────
    # Console visualization  (mirrors LIME's visualize pattern)
    # ─────────────────────────────────────────────────────────────────────

    def visualize(self, explanation, instance_index=0, class_index=0, num_features=10):
        """
        Print SHAP feature attributions in readable console format.

        Mirrors the print style of LimeExplainer_Tabular.visualize().
        Positive values shown with '+', negative with '-'.

        Parameters
        ----------
        explanation    : dict   — direct output of explain()
        instance_index : int    — which sample to display. Default: 0.
        class_index    : int    — which class to display. Default: 0.
        num_features   : int    — how many top features to show. Default: 10.

        Returns
        -------
        feature_scores : list of (str, float)
            Same as get_explanation_data() — returned for programmatic use.
        """
        feature_scores = self.get_explanation_data(
            explanation,
            instance_index=instance_index,
            class_index=class_index,
            num_features=num_features,
        )

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        print(f"\nSHAP Tabular Explanation  [{self.explainer_type.upper()}  |  Class: {class_label}]")
        print("-" * 42)
        for feature, score in feature_scores:
            sign = "+" if score >= 0 else "-"
            bar  = "|" * min(int(abs(score) * 80), 20)
            print(f"  {feature:<28s}  {sign}  {abs(score):.6f}  {bar}")
        print("-" * 42)

        return feature_scores

    # ─────────────────────────────────────────────────────────────────────
    # SHAP plots
    # ─────────────────────────────────────────────────────────────────────

    def summary_plot(self, explanation, data, class_index=0, save_png=False, save_dir="user_saves"):
        """
        SHAP summary plot — shows feature impact distribution across all samples.
        Each dot = one sample. Color = feature value. X-axis = SHAP value.

        Parameters
        ----------
        explanation : dict                         — output of explain()
        data        : np.ndarray or torch.Tensor   — same input passed to explain()
        class_index : int                          — which class to plot. Default: 0.
        save_png : bool
            If True, saves the plot as a .png file inside save_dir. Default: False.
        save_dir : str
            Directory to save the plot. Created automatically if it does not exist.
            Default: 'user_saves'.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap_values = explanation["shap_values"]

        # Extract the correct class slice for a clean single-class plot
        if isinstance(shap_values, list):
            values_to_plot = np.array(shap_values[class_index])   # (n_samples, num_features)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values_to_plot = shap_values[:, :, class_index]       # (n_samples, num_features)
        else:
            values_to_plot = np.array(shap_values)                # (n_samples, num_features)

        shap.summary_plot(
        values_to_plot,
        data,
        feature_names=self.feature_names,
        show=False,
        )
        if save_png:
            import matplotlib.pyplot as plt
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "shap_summary_plot.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.show()
            print(f"[ShapExplainer_Tabular] Saved to: {save_path}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    def bar_plot(self, explanation, class_index=0, save_png=False, save_dir="user_saves"):
        """
        SHAP bar plot — shows mean absolute SHAP value per feature (global importance).

        Parameters
        ----------
        explanation : dict   — output of explain()
        class_index : int    — which class to plot. Default: 0.
        save_png : bool
            If True, saves the plot as a .png file inside save_dir. Default: False.
        save_dir : str
            Directory to save the plot. Created automatically if it does not exist.
            Default: 'user_saves'.
        """
        shap_values = explanation["shap_values"]

        # Extract the correct class slice — consistent with summary_plot
        if isinstance(shap_values, list):
            values_to_plot = np.array(shap_values[class_index])   # (n_samples, num_features)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            values_to_plot = shap_values[:, :, class_index]       # (n_samples, num_features)
        else:
            values_to_plot = np.array(shap_values)                # (n_samples, num_features)

        # Use summary_plot with plot_type="bar" — more robust than shap.plots.bar()
        # with a manually constructed Explanation object
        
        shap.summary_plot(
            values_to_plot,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )

        if save_png:
            import matplotlib.pyplot as plt
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "shap_bar_plot.png"), bbox_inches="tight", dpi=150)
            plt.show()
            print(f"[ShapExplainer_Tabular] Saved to: {os.path.join(save_dir, 'shap_bar_plot.png')}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    def waterfall_plot(self, explanation, data, instance_index=0, class_index=0, save_png=False, save_dir="user_saves"):
        """
        SHAP waterfall plot — shows how each feature contributed to a single prediction.
        Best for explaining one specific sample step-by-step.

        Parameters
        ----------
        explanation    : dict                         — output of explain()
        data           : np.ndarray or torch.Tensor
        instance_index : int                          — which sample to plot. Default: 0.
        class_index    : int                          — which class to plot. Default: 0.
        save_png : bool
            If True, saves the plot as a .png file inside save_dir. Default: False.
        save_dir : str
            Directory to save the plot. Created automatically if it does not exist.
            Default: 'user_saves'.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap_val     = self._extract_values(
            explanation["shap_values"], instance_index, class_index
        )
        expected_val = self._extract_expected_value(
            explanation["expected_value"], class_index
        )

        explanation_obj = shap.Explanation(
            values        = shap_val,
            base_values   = expected_val,
            data          = data[instance_index],
            feature_names = self.feature_names,
        )

        shap.plots.waterfall(explanation_obj,show=False)

        if save_png:
            import matplotlib.pyplot as plt
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "shap_waterfall_plot.png"), bbox_inches="tight", dpi=150)
            plt.show()
            print(f"[ShapExplainer_Tabular] Saved to: {os.path.join(save_dir, 'shap_waterfall_plot.png')}")
        else:
            import matplotlib.pyplot as plt
            plt.show()
        

    def force_plot(self, explanation, data, instance_index=0, class_index=0, save_png=False, save_dir="user_saves"):
        """
        SHAP force plot — shows how features push prediction from baseline.
        Red features push prediction higher, blue push it lower.

        Parameters
        ----------
        explanation    : dict                         — output of explain()
        data           : np.ndarray or torch.Tensor
        instance_index : int                          — which sample to plot. Default: 0.
        class_index    : int                          — which class to plot. Default: 0.
        save_png : bool
            If True, saves the plot as a .png file inside save_dir. Default: False.
        save_dir : str
            Directory to save the plot. Created automatically if it does not exist.
            Default: 'user_saves'.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap_val     = self._extract_values(
            explanation["shap_values"], instance_index, class_index
        )
        expected_val = self._extract_expected_value(
            explanation["expected_value"], class_index
        )
        
        shap.force_plot(
            expected_val,
            shap_val,
            data[instance_index],
            feature_names = self.feature_names,
            matplotlib    = True,
            show=False,
        )

        if save_png:
            import matplotlib.pyplot as plt
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "shap_force_plot.png"), bbox_inches="tight", dpi=150)
            plt.show()
            print(f"[ShapExplainer_Tabular] Saved to: {os.path.join(save_dir, 'shap_force_plot.png')}")
        else:
            import matplotlib.pyplot as plt
            plt.show()