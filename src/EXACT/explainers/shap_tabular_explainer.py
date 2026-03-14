import numpy as np
import torch
import shap

from ..utils.predict_proba_fn import predict_proba


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

        'gradient'  ->  GradientExplainer  : Gradient x input approach.
                                             Good alternative for deep networks.
                                             More stable than DeepExplainer on some
                                             architectures (e.g. with BatchNorm).

        'kernel'    ->  KernelExplainer    : Model-agnostic black-box fallback.
                                             Works on any model but slowest.
                                             Uses predict_proba wrapper internally.

    Notes:
        - Model is automatically set to eval() mode
        - background_data is SHAP's reference distribution (equiv. to LIME's training_data)
        - All tensor/numpy conversions are handled internally
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
            Expected input shape: (batch_size, num_features).

        background_data : np.ndarray or torch.Tensor
            Reference distribution used by SHAP as the baseline.
            Shape: (n_background_samples, num_features)

            This is SHAP's equivalent of LIME's training_data parameter.
            Represents the "average" model input — the neutral reference against
            which each feature's contribution is measured.

            Recommended: 50–200 rows from training data.
            Tip: K-Means cluster centers of your training set work well.

        feature_names : list[str], optional
            Names of input features.
            If None, shap plots auto-label as Feature 0, Feature 1, ...

        class_names : list[str], optional
            Names of output classes (used in visualization only).

        explainer_type : str
            Which SHAP backend to use: 'deep' | 'gradient' | 'kernel'
            Default: 'deep'
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

        # Resolve device from model
        self.device = next(model.parameters()).device

        # Always eval mode — critical for correct attributions
        self.model.eval()

        # Prepare background in both formats:
        #   background_tensor → used by DeepExplainer and GradientExplainer
        #   background_np     → used by KernelExplainer
        if isinstance(background_data, torch.Tensor):
            self.background_tensor = background_data.float().to(self.device)
            self.background_np     = background_data.detach().cpu().numpy()
        else:
            self.background_np     = np.array(background_data, dtype=np.float32)
            self.background_tensor = torch.tensor(
                self.background_np, dtype=torch.float32
            ).to(self.device)

        # Initialize the SHAP explainer
        self.explainer = self._init_explainer()
        self.expected_value = self._compute_expected_value()
    # ──────────────────────────────────────────────────────────────────────
    # Internal: prediction wrapper  (used only by KernelExplainer)
    # ──────────────────────────────────────────────────────────────────────

    def _predict(self, X):
        """
        Prediction wrapper for KernelExplainer.

        KernelExplainer is model-agnostic — it needs a plain callable that
        accepts numpy input and returns numpy probabilities.
        We reuse the shared predict_proba utility (same as LIME) for this.

        DeepExplainer and GradientExplainer receive the model directly
        and do NOT use this wrapper — they work via backpropagation.

        Parameters
        ----------
        X : np.ndarray   shape: (n_samples, num_features)

        Returns
        -------
        np.ndarray   shape: (n_samples, num_classes)  — softmax probabilities
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32) 
        return predict_proba(X, model=self.model)

    # ──────────────────────────────────────────────────────────────────────
    # Internal: SHAP explainer initialization
    # ──────────────────────────────────────────────────────────────────────

    def _init_explainer(self):
        """
        Initialize the correct SHAP explainer for the selected type.

        Falls back to KernelExplainer automatically if Deep or Gradient
        initialization fails (e.g. unsupported layer types).

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
                # KernelExplainer: takes prediction function and background numpy array.
                # Model-agnostic — uses the _predict wrapper above.
                return shap.KernelExplainer(self._predict, self.background_np)

        except Exception as e:
            print(
                f"[ShapExplainer_Tabular] WARNING: '{self.explainer_type}' explainer "
                f"failed to initialize ({e}). Falling back to KernelExplainer."
            )
            self.explainer_type = "kernel"
            return shap.KernelExplainer(self._predict, self.background_np)


    def _compute_expected_value(self):
        """
        Compute the baseline expected value for all explainer types.

        DeepExplainer and KernelExplainer expose .expected_value directly.
        GradientExplainer does not — so we compute it manually by running
        the background data through the model and averaging the output.

        Returns
        -------
        float or list[float]
            For multi-class: list of per-class baseline predictions.
            For regression/binary: single float.
        """
        if self.explainer_type in ("deep", "kernel"):
            # These explainers already compute and expose expected_value
            return self.explainer.expected_value

        # GradientExplainer: compute manually from background data
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.background_tensor)          # (N, num_classes)
            probs  = torch.softmax(output, dim=1)               # (N, num_classes)
            mean   = probs.mean(dim=0).cpu().numpy()            # (num_classes,)

        # Return as list to match DeepExplainer's format
        return mean.tolist()

    # ──────────────────────────────────────────────────────────────────────
    # Core explanation logic
    # ──────────────────────────────────────────────────────────────────────

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
                               For regression / binary: (n_samples, num_features).
            'expected_value' : float or list[float]
                               Baseline prediction (model output on background).
                               For multi-class: list, one per class.
            'feature_names'  : list[str] or None
            'explainer_type' : str
        """
        # Normalize input to both tensor and numpy
        if isinstance(data, torch.Tensor):
            data_tensor = data.float().to(self.device)
            data_np     = data.detach().cpu().numpy()
        else:
            data_np     = np.array(data, dtype=np.float32)
            data_tensor = torch.tensor(data_np, dtype=torch.float32).to(self.device)

        # Ensure 2D — (1, num_features) for single sample
        if data_np.ndim == 1:
            data_np     = data_np[np.newaxis, :]       # (1, num_features)
            data_tensor = data_tensor.unsqueeze(0)     # (1, num_features)

        # Compute SHAP values using the appropriate explainer
        if self.explainer_type == "kernel":
            # KernelExplainer works with numpy and uses _predict internally
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

    # ──────────────────────────────────────────────────────────────────────
    # Raw explanation data  (mirrors LIME's get_explanation_data pattern)
    # ──────────────────────────────────────────────────────────────────────

    def get_explanation_data(self, explanation, instance_index=0, class_index=0, num_features=10):
        """
        Extract a sorted (feature_name, shap_value) list from explain() output.

        Mirrors the role of LimeExplainer_Tabular.get_explanation_data() so
        both explainers can be used interchangeably downstream.

        Parameters
        ----------
        explanation : dict
            Direct output of explain().

        instance_index : int
            Which sample in the batch to extract. Default: 0.

        class_index : int
            Which class to extract SHAP values for (multi-class models).
            Default: 0.

        num_features : int
            How many top features to return. Default: 10.

        Returns
        -------
        list of (str, float)
            [(feature_name, shap_value), ...] sorted by |shap_value| descending.
        """
        shap_values = explanation["shap_values"]

        # Handle multi-class output: shap_values is a list, one array per class
        if isinstance(shap_values, list):
            values = np.array(shap_values[class_index][instance_index]).flatten()  # (num_features,)

        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array: (n_samples, num_features, num_classes)
            values = shap_values[instance_index, :, class_index].flatten()

        else:
            # 2D array: (n_samples, num_features)
            values = np.array(shap_values[instance_index]).flatten()

    # --- Pair with feature names and sort by absolute importance ---
        num_total = len(values)

        if self.feature_names is not None:
            names = list(self.feature_names)
        else:
            names = [f"Feature_{i}" for i in range(num_total)]

        paired  = list(zip(names, values.tolist()))
        sorted_ = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        return sorted_[:num_features]

    # ──────────────────────────────────────────────────────────────────────
    # Console visualization  (mirrors LIME's visualize pattern)
    # ──────────────────────────────────────────────────────────────────────

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

        print(f"\nSHAP Tabular Explanation  [{self.explainer_type.upper()}]")
        print("-" * 42)
        for feature, score in feature_scores:
            sign = "+" if score >= 0 else "-"
            bar  = "|" * min(int(abs(score) * 80), 20)
            print(f"  {feature:<28s}  {sign}  {abs(score):.6f}  {bar}")
        print("-" * 42)

        return feature_scores

    # ──────────────────────────────────────────────────────────────────────
    # SHAP plots  (uses shap library's built-in plot functions)
    # ──────────────────────────────────────────────────────────────────────

    def summary_plot(self, explanation, data):
        """
        SHAP summary plot — shows feature impact distribution across all samples.
        Each dot = one sample. Color = feature value. X-axis = SHAP value.

        Parameters
        ----------
        explanation : dict   — output of explain()
        data        : np.ndarray or torch.Tensor   — the same input passed to explain()
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap.summary_plot(
            explanation["shap_values"],
            data,
            feature_names=self.feature_names,
        )

    def bar_plot(self, explanation):
        """
        SHAP bar plot — shows mean absolute SHAP value per feature (global importance).

        Parameters
        ----------
        explanation : dict   — output of explain()
        """
        shap_values = explanation["shap_values"]

        if isinstance(shap_values, list):
            values = np.array(shap_values[0]).mean(axis=0)
        else:
            values = np.array(shap_values).mean(axis=0)

        shap.plots.bar(
            shap.Explanation(
                values        = values,
                feature_names = self.feature_names,
            )
        )

    def waterfall_plot(self, explanation, data, instance_index=0, class_index=0):
        """
        SHAP waterfall plot — shows how each feature contributed to a single prediction.
        Best for explaining one specific sample step-by-step.

        Parameters
        ----------
        explanation    : dict   — output of explain()
        data           : np.ndarray or torch.Tensor
        instance_index : int    — which sample to plot. Default: 0.
        class_index    : int    — which class to plot. Default: 0.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap_values   = explanation["shap_values"]
        expected_value = explanation["expected_value"]

        # Extract the right slice based on output format

        # Case 1: multi-class → shap_values is a list, one array per class
        if isinstance(shap_values, list):
            shap_val     = np.array(shap_values[class_index][instance_index]).flatten()   # (num_features,)
            expected_val = (
                expected_value[class_index]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_val= shap_values[instance_index, :, class_index].flatten()
            expected_val = (
            expected_value[class_index]
            if isinstance(expected_value, (list, np.ndarray))
            else expected_value
            )
        else:
            shap_val     = np.array(shap_values[instance_index]).flatten()
            expected_val = (
            expected_value[0]
            if isinstance(expected_value, (list, np.ndarray))
            else expected_value
            )


        explanation_obj = shap.Explanation(
            values        = shap_val,
            base_values   = float(expected_val),
            data          = data[instance_index],
            feature_names = self.feature_names,
        )

        shap.plots.waterfall(explanation_obj)

    def force_plot(self, explanation, data, instance_index=0, class_index=0):
        """
        SHAP force plot — shows how features push prediction from baseline.
        Red features push prediction higher, blue features push it lower.

        Parameters
        ----------
        explanation    : dict   — output of explain()
        data           : np.ndarray or torch.Tensor
        instance_index : int    — which sample to plot. Default: 0.
        class_index    : int    — which class to plot. Default: 0.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[np.newaxis, :]

        shap_values    = explanation["shap_values"]
        expected_value = explanation["expected_value"]

        # Normalize to flat 1D array — same logic as get_explanation_data
        if isinstance(shap_values, list):
            shap_val     = np.array(shap_values[class_index][instance_index]).flatten()
            expected_val = (
                expected_value[class_index]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_val     = shap_values[instance_index, :, class_index].flatten()
            expected_val = (
                expected_value[class_index]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )
        else:
            shap_val     = np.array(shap_values[instance_index]).flatten()
            expected_val = (
                expected_value[0]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )

        shap.force_plot(
            float(expected_val),
            shap_val,
            data[instance_index],
            feature_names = self.feature_names,
            matplotlib    = True,
    )