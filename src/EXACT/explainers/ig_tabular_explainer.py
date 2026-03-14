"""
ig_tabular_explainer.py
========================
Integrated Gradients (IG) explainer for PyTorch tabular classification models.

WHAT THIS FILE DOES IN ONE SENTENCE
-------------------------------------
Given a trained model and one row of tabular data (e.g. a patient record),
this file tells you exactly which features pushed the prediction toward or
away from the predicted class — and by how much.

WHAT IS INTEGRATED GRADIENTS?
-------------------------------
Imagine you have a model that predicts heart disease risk.
You pass in one patient: age=65, cholesterol=280, blood_pressure=145.
The model says "has disease" with high confidence.

The question IG answers:
    "Compared to the average patient, which features caused this prediction?"

The answer comes out as one signed number per feature:
    age            : +0.32   → being 65 (vs average 55) PUSHED toward disease
    cholesterol    : +0.28   → high cholesterol also PUSHED toward disease
    blood_pressure : +0.19   → high BP also PUSHED toward disease
    max_heart_rate : -0.11   → their heart rate actually PUSHED away from disease

HOW IG WORKS (THE INTUITION)
------------------------------
IG walks in a straight line from the baseline (average patient)
to the actual input (this patient), in tiny steps.

At each step along the path it asks:
    "If I nudge this feature by a tiny amount right now, how much does
     the model's prediction score change?"

That answer is the gradient at that step.

After walking all the steps, it averages the gradients for each feature
and multiplies by how far that feature actually moved (input - baseline).

Result: one attribution score per feature that satisfies the
"completeness axiom" — the attributions sum exactly to
F(input) - F(baseline), meaning no credit is lost or invented.

WHAT IS THE BASELINE?
----------------------
For images: the baseline is a black image (all zeros = absence of input).
For tabular data: zeros are often wrong.
    age=0 means nothing. income=0 means nothing.

The correct baseline is the TRAINING SET MEAN — the "average patient".
IG then answers: compared to the average patient, what made this prediction?

Pass training_data=X_train and the mean is computed automatically.

WHAT WILL BREAK (LIMITATIONS)
-------------------------------
1. Regression models:
   Your model must output [batch, num_classes] logits (2D tensor).
   If it outputs a single number [batch] or [batch, 1], the code raises
   a clear error. Use a classification wrapper if needed.

2. Models with Embedding layers for categorical features:
   Embedding layers take integer indices as input. Integers are discrete —
   you cannot take a gradient with respect to an integer.
   For such models the gradient for embedded features will be zero or wrong.
   Use LIME or SHAP KernelExplainer for models with embedding layers.

3. No training_data + unnormalised raw data:
   If you don't pass training_data and your data has values like age=65
   (not normalised), the zero baseline is nonsensical.
   The code will warn you when this looks like it might be happening.

VISUALISATIONS PRODUCED (5 charts)
------------------------------------
1. Bar chart         — signed attributions, sorted by importance
2. Force plot        — shows baseline score → final score with feature arrows
3. Waterfall chart   — shows how attribution accumulates step by step
4. Distribution plot — shows where this sample sits in the training distribution
5. Summary table     — ranked table suitable for reports and presentations
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings        # used to warn users about bad baseline choices

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch           # core PyTorch tensor operations
import torch.nn as nn  # nn.Module base class for type hints

# ── Numerical computing ───────────────────────────────────────────────────────
import numpy as np     # array operations, used for attribution processing

# ── Plotting ──────────────────────────────────────────────────────────────────
# WHY WE DO NOT CALL matplotlib.use("Agg") HERE AT MODULE LEVEL:
#
# matplotlib.use("Agg") must be called BEFORE matplotlib.pyplot is imported
# anywhere in the entire program. If the user has already imported pyplot
# in their Jupyter notebook or script before importing this module, calling
# use("Agg") here either silently does nothing (old matplotlib) or raises a
# warning (new matplotlib). In both cases the backend stays whatever the user
# already set, which may cause a crash on headless servers.
#
# The safe solution: call use("Agg") only if matplotlib has not yet chosen
# a backend. We check this with matplotlib.get_backend() before importing pyplot.
# This way:
#   - In a fresh script:        we set Agg safely before pyplot loads.
#   - In a Jupyter notebook:    pyplot is already loaded with the notebook backend,
#                               we detect this and leave it alone. Saving to file
#                               still works with any backend.
#   - On a headless server:     if nothing loaded pyplot yet, we set Agg correctly.
import matplotlib
if matplotlib.get_backend() == "":
    # No backend chosen yet — safe to set Agg before pyplot is imported.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt    # main plotting interface
import matplotlib.patches as mpatches  # used for legend colour patches

# ── Type hints ────────────────────────────────────────────────────────────────
from typing import Dict, List, Optional


# =============================================================================
# MODULE-LEVEL HELPERS
# These two functions are used by every single plot function.
# Putting them here avoids repeating the same 10 lines in 5 places.
# =============================================================================

def _apply_dark_theme(ax: plt.Axes) -> None:
    """
    Apply a consistent dark visual theme to a matplotlib Axes object.

    Why this exists:
        Every one of our 5 plot functions needs the same dark background,
        white tick labels, hidden top/right spines, and coloured axis labels.
        Without this helper we would copy-paste those 8 lines into every
        plot function — 40 lines of duplication.  One call here does it all.

    What each line does:
        set_facecolor    — dark navy background for the plot area
        tick_params      — white tick labels, font size 8
        label.set_color  — axis labels (xlabel, ylabel) in light purple-white
        title.set_color  — plot title in pure white
        spines visible   — remove the top and right border lines (cleaner look)
        spines color     — make the bottom and left borders dark purple
    """
    ax.set_facecolor("#1a1a2e")                     # dark navy plot background
    ax.tick_params(colors="#ccccee", labelsize=8)   # white-ish tick text
    ax.xaxis.label.set_color("#ccccee")             # x-axis label colour
    ax.yaxis.label.set_color("#ccccee")             # y-axis label colour
    ax.title.set_color("white")                     # subplot title colour
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)         # remove top/right borders
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444466")       # dim purple remaining borders


def _get_top_k_indices(attr: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the k most important features, sorted descending
    by absolute attribution value.

    Why absolute value?
        Both +0.5 and -0.5 are equally important.
        +0.5 strongly supports the prediction.
        -0.5 strongly suppresses the prediction.
        Both are more important than a feature with attribution 0.01.
        So we sort by |value|, not by value.

    Example:
        attr = [0.1, -0.4, 0.3, -0.05, 0.2]
        k = 3
        abs(attr) = [0.1, 0.4, 0.3, 0.05, 0.2]
        argsort ascending = [3, 0, 4, 2, 1]
        reversed (descending) = [1, 2, 4, 0, 3]
        [:3] = [1, 2, 4]  ← indices of top-3
    """
    return np.argsort(np.abs(attr))[::-1][:k]


# =============================================================================
# MAIN CLASS
# =============================================================================

class IGTabularExplainer:
    """
    Integrated Gradients explainer for PyTorch tabular classification models.

    Usage:
        explainer = IGTabularExplainer(model, feature_names=["age", "income"])
        results   = explainer.explain(input_tensor, training_data=X_train)
        explainer.save_dashboard(results, "explanation.png")
    """

    def __init__(
        self,
        model:         nn.Module,
        feature_names: Optional[List[str]] = None,
        device:        Optional[torch.device] = None,
    ):
        """
        Store the model and configuration.  No computation happens here.

        Args:
            model:
                A trained PyTorch model.
                Requirements:
                    - Input shape:  [batch_size, num_features]  (2D float tensor)
                    - Output shape: [batch_size, num_classes]   (2D float tensor)
                The model does NOT need to be in eval() mode before passing in —
                we call model.eval() here automatically.

            feature_names:
                A list of strings naming each input column.
                Example: ["age", "income", "credit_score"]
                If None: features will be named "F0", "F1", "F2", ...
                Must have exactly num_features entries if provided.

            device:
                The torch.device to run inference on ("cpu" or "cuda").
                If None: automatically detected from the model's weight tensors.
                Example: torch.device("cuda") or torch.device("cpu")
        """
        # Put model in evaluation mode.
        # This disables dropout (which would add randomness to gradients)
        # and makes batch norm use stored statistics (not batch statistics).
        # Both are essential for consistent, correct IG attributions.
        self.model = model.eval()

        # Auto-detect device from the model's first parameter tensor.
        self.device = device or next(model.parameters()).device

        # Store feature names for use in visualisations later.
        self.feature_names = feature_names

        # ── Detect Style B ([batch,1] sigmoid) model once at construction ─────
        # _safe_forward() is called hundreds of times inside the Riemann sum
        # loop. If we put the Style B detection + warning there, it would fire
        # 200+ times per explain() call.
        # We probe the model here once at __init__, store a boolean flag,
        # and warn exactly once per explain() call using that flag.
        #
        # Probe: build a 1-sample dummy input and run one forward pass.
        # If output shape is [1, 1] → Style B (sigmoid binary classifier).
        # If output shape is [1, N] where N>=2 → Style A (standard logits).
        try:
            # Find the first 2D parameter — this is always an nn.Linear weight
            # [out_features, in_features]. We skip 1D parameters (biases, BN
            # scales/shifts) which would give the wrong n_in and crash.
            # Example: BN-first model → first param is BN weight [F] (1D) → skip it.
            # We look for ndim==2 so that nn.Sequential(BN, Linear) works correctly.
            n_in = None
            for p in model.parameters():
                if p.ndim == 2:
                    n_in = p.shape[1]
                    break
            if n_in is None:
                raise ValueError("No 2D parameter found — cannot probe model input size.")
            dummy = torch.zeros(1, n_in, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                dummy_out = model(dummy)
            self._is_style_b = (dummy_out.dim() == 2 and dummy_out.shape[1] == 1)
        except Exception:
            # Unusual architecture — assume Style A, let forward pass handle it
            self._is_style_b = False

    # =========================================================================
    # PUBLIC METHOD: explain()
    # =========================================================================

    def explain(
        self,
        input_tensor:  torch.Tensor,
        target_class:  Optional[int]         = None,
        baseline:      Optional[torch.Tensor] = None,
        training_data: Optional[np.ndarray]  = None,
        steps:         int                   = 200,
        batch_size:    int                   = 64,
        top_k:         Optional[int]         = None,
    ) -> Dict:
        """
        Run Integrated Gradients on one tabular sample.

        This is the main method. It:
            1. Validates and prepares inputs
            2. Builds the baseline if not provided
            3. Runs one forward pass to get the predicted class
            4. Runs the IG computation (the Riemann sum loop)
            5. Builds all 5 visualisation charts
            6. Returns everything in a dictionary

        Args:
            input_tensor:
                The single sample to explain.
                Accepted shapes:
                    [num_features]    — a 1D tensor, e.g. torch.tensor([65, 280, 145])
                    [1, num_features] — a 2D tensor with batch dimension
                Both are accepted; [num_features] is auto-expanded to [1, num_features].

            target_class:
                The class index to compute attributions for.
                None → uses the class with the highest predicted score (argmax).
                Example: if your model has 3 classes and predicts class 2 most
                strongly, target_class will be set to 2 automatically.

            baseline:
                The reference point for IG.  Shape: [1, num_features].
                This is what IG compares the input against.
                None → uses training set mean (if training_data provided)
                       or zeros (with a warning if data looks unnormalised).

            training_data:
                numpy array of shape [N, num_features] — the training set.
                Used for two purposes:
                    1. Computing the mean baseline (more meaningful than zeros)
                    2. Drawing distribution histograms in the distribution plot
                Strongly recommended. Without it, the zero baseline may be wrong.

            steps:
                Number of steps in the Riemann sum approximation.
                More steps = more accurate attributions, slower computation.
                200 is accurate for most tabular models (small, simple nets).
                Use 50–100 for fast iteration, 500+ for publication-quality results.

            batch_size:
                How many interpolated samples to process in one forward pass.
                64 is safe for any tabular model.
                Tabular models are tiny compared to image models, so memory is
                not a concern here — 64 is just a sensible default.

            top_k:
                How many features to show in the visualisations.
                None → shows all features.
                Useful when you have many features (e.g. 50+) and only want to
                see the most important ones in the charts.
                All features are still computed — top_k only affects display.

        Returns:
            A dictionary with these keys:

            "target_class"       int
                The class that was explained (either provided or argmax).

            "convergence_delta"  float
                The completeness error: |sum(attributions) - (F(input) - F(baseline))|
                Perfect IG gives delta = 0.
                In practice with 200 steps:  delta < 0.05 is great,
                                             delta < 0.15 is acceptable,
                                             delta > 0.15 means increase steps.

            "feature_names"      List[str]
                Names of all features (either provided or auto-generated).

            "attributions"       np.ndarray of shape [num_features]
                The IG attribution score for each feature.
                Positive = feature pushed prediction TOWARD target_class.
                Negative = feature pushed prediction AWAY from target_class.
                Magnitude = how strongly.

            "input_values"       np.ndarray of shape [num_features]
                The actual feature values of the input sample.

            "baseline_values"    np.ndarray of shape [num_features]
                The baseline feature values that were used.

            "training_data"      np.ndarray or None
                Passed through unchanged (used by distribution plot).

            "top_k"              int
                How many features are shown in the charts.

            "chart_bar"          matplotlib.figure.Figure
            "chart_force"        matplotlib.figure.Figure
            "chart_waterfall"    matplotlib.figure.Figure
            "chart_distribution" matplotlib.figure.Figure
            "chart_summary"      matplotlib.figure.Figure
                The five visualisation charts as Figure objects.
                Save them individually with fig.savefig("path.png")
                or use save_dashboard() to combine them all in one PNG.
        """

        # ── STEP 1: Validate and normalise input tensor shape ─────────────────
        # We accept both [F] and [1, F] so the user doesn't have to worry about
        # adding a batch dimension manually.
        if input_tensor.dim() == 1:
            # [F] → [1, F]: add a batch dimension of size 1
            input_tensor = input_tensor.unsqueeze(0)

        # After the above, we should have exactly a 2D tensor with 1 row.
        if input_tensor.dim() != 2 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be shape [F] or [1, F], "
                f"but got shape {tuple(input_tensor.shape)}. "
                f"Only a single sample (batch size 1) is supported."
            )

        # Guard against too-few steps. steps=1 gives a single-point gradient
        # approximation which violates the Completeness Axiom badly.
        # 50 is the minimum for any meaningful Riemann sum.
        # (Tabular models are small so 200 steps is very fast anyway.)
        if steps < 20:
            raise ValueError(
                f"steps={steps} is too low for a reliable approximation. "
                f"Use at least 50 (recommended: 200 for tabular models)."
            )

        # Cast to float32 and move to the model's device (cpu or cuda).
        # float() ensures we don't accidentally pass int tensors.
        input_tensor = input_tensor.float().to(self.device)

        # Number of features — used for validation and auto-naming.
        F = input_tensor.shape[1]

        # ── CATEGORICAL FEATURE DETECTION ────────────────────────────────────
        # IG computes gradients — it measures "if I change this feature by a
        # tiny continuous amount, how does the output change?"
        # This is meaningful for continuous features (age, income, temperature).
        # It is misleading for categorical features encoded as integers
        # (e.g. gender: 0/1, blood_type: 0/1/2/3) because those values are
        # labels, not magnitudes. Changing gender from 0 to 0.001 has no
        # real-world meaning, so the gradient is technically computed but
        # conceptually wrong.
        #
        # We detect this by checking if ALL values in the input are integers
        # (i.e. value == floor(value) for every element).
        # If so, the user may have forgotten to encode or normalise their data.
        #
        # We WARN rather than crash because:
        #   1. Some users deliberately pass integer-encoded ordinal features
        #      (e.g. severity: 1/2/3/4/5) where IG is still reasonable.
        #   2. Some users normalise after the check and would hit a false alarm.
        #   3. The computation still runs — the user decides whether to trust it.
        vals_np = input_tensor.squeeze(0).cpu().numpy()
        if np.all(np.abs(vals_np - np.round(vals_np)) < 1e-5):
            warnings.warn(
                "All input feature values are integers. "
                "If your data contains categorical features (e.g. gender=0/1, "
                "blood_type=0/1/2/3), IG attributions for those features may be "
                "misleading because IG requires continuous inputs. "
                "Consider: (1) normalising your data first, or (2) using LIME "
                "instead for models with categorical features. "
                "If your features are genuinely ordinal integers (e.g. severity 1-5), "
                "you can safely ignore this warning.",
                UserWarning,
                stacklevel=2,
            )

        # ── STEP 2: Resolve feature names ─────────────────────────────────────
        # If the user provided names at __init__, use them.
        # Otherwise generate: ["F0", "F1", "F2", ...]
        if self.feature_names is not None:
            names = self.feature_names
            if len(names) != F:
                raise ValueError(
                    f"You provided {len(names)} feature names, "
                    f"but the input has {F} features. These must match."
                )
        else:
            names = [f"F{i}" for i in range(F)]

        # ── STEP 3: Build baseline ─────────────────────────────────────────────
        # Priority order:
        #   1. User passed an explicit baseline tensor → use it directly.
        #   2. User passed training_data → compute the mean of each column.
        #   3. Neither → fall back to zeros (and warn if data looks unnormalised).
        if baseline is None:
            if training_data is not None:
                # training_data.mean(axis=0) computes the column-wise mean.
                # axis=0 means "average across rows, keep all columns".
                # Result shape: [F] → we unsqueeze to [1, F] for broadcasting.
                # np.array() guards against pandas DataFrame/Series input:
                # torch.tensor(pandas_series) can raise a warning about
                # non-contiguous data. Converting to numpy first avoids this.
                mean_values = np.array(training_data.mean(axis=0), dtype=np.float32)
                baseline = torch.tensor(mean_values,
                                        dtype=torch.float32).unsqueeze(0)
            else:
                # Zero baseline: only sensible if data is normalised to ~mean 0.
                # If max value is > 10, the data is probably raw (age=65, not 0.2).
                # Warn the user so they know their attributions may be misleading.
                if input_tensor.abs().max().item() > 10:
                    warnings.warn(
                        "No training_data was provided and the input values look "
                        "unnormalised (the largest value is greater than 10). "
                        "Using a zero baseline for raw values like age=65 or "
                        "income=80000 is wrong because zero is not a meaningful "
                        "reference point for those features. "
                        "Please pass training_data=X_train to use the mean baseline.",
                        UserWarning,
                        stacklevel=2,  # points the warning at the caller, not here
                    )
                baseline = torch.zeros_like(input_tensor)

        # Cast and move baseline to the same device as input.
        # Validate shape first — a mismatched baseline gives a confusing
        # low-level PyTorch error inside _compute_ig (shape mismatch on subtraction).
        # We raise a clear ValueError here instead.
        baseline = baseline.float()
        if baseline.shape != input_tensor.shape:
            raise ValueError(
                f"baseline shape {tuple(baseline.shape)} does not match "
                f"input_tensor shape {tuple(input_tensor.shape)}. "
                f"They must be identical (both [1, num_features])."
            )
        baseline = baseline.to(self.device)

        # ── STEP 4: Warn once if this is a Style B ([batch,1]) model ────────
        # self._is_style_b was set at __init__ by probing the model output shape.
        # We warn here — exactly once per explain() call — so the user knows
        # their model's output is being auto-converted from [batch,1] to [batch,2].
        # We do NOT warn inside _safe_forward() because that is called 200+ times
        # per explain() call (once per Riemann sum step), which would spam the user.
        if self._is_style_b:
            warnings.warn(
                "Model output shape is [batch, 1] — detected as a binary classifier "
                "using a single sigmoid output (Style B). "
                "Automatically converting to two-class format: [1-p, p] "
                "where p is the sigmoid probability of class 1. "
                "Both target_class=0 and target_class=1 will work correctly. "
                "To silence this warning, change your model's final layer to "
                "output [batch, 2] logits without sigmoid activation.",
                UserWarning,
                stacklevel=2,   # points warning at the caller of explain(), not here
            )

        # ── STEP 5: Run one forward pass to get predicted class ───────────────
        # We always force eval() mode here, not just at __init__.
        # Reason: the user might have called model.train() between __init__
        # and explain(), which would corrupt attributions via batch norm / dropout.
        self.model.eval()

        with torch.no_grad():
            # Run the model on the input. No gradients needed here.
            # _safe_forward handles both [batch,2] and [batch,1] output styles.
            logits = self._safe_forward(input_tensor)

        # ── OUTPUT SHAPE GUARD ────────────────────────────────────────────────
        # _safe_forward already converts [batch,1] → [batch,2].
        # The only remaining bad case is [batch] (1D) from a regression model.
        if logits.dim() == 1:
            raise ValueError(
                f"Model returned a 1D tensor of shape {tuple(logits.shape)}. "
                f"Expected [batch, num_classes] (2D). "
                f"If this is a binary classifier using a single sigmoid output, "
                f"make sure the output has shape [batch, 1] not [batch]. "
                f"Example: use nn.Linear(32, 1) — not just a scalar output."
            )

        if logits.dim() != 2:
            raise ValueError(
                f"Model returned unexpected shape {tuple(logits.shape)}. "
                f"Expected 2D [batch, num_classes]. "
                f"Regression models are not supported by IG."
            )

        # If target_class not specified, use the class with the highest score.
        # argmax(dim=1) finds the column index of the maximum in each row.
        # .item() converts the 0-dimensional tensor to a plain Python int.
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Cache the model's output score for target_class on the input.
        # This is used later in the convergence check.
        # We cache it here to avoid running the model a second time later.
        # logits[0, target_class] = the score for our target class on sample 0.
        f_input = float(logits[0, target_class].item())

        # ── STEP 6: Compute Integrated Gradients ──────────────────────────────
        # This is where the actual IG algorithm runs.
        # Returns attributions [1, F] and the convergence delta (float).
        attributions, convergence_delta = self._compute_ig(
            input_tensor  = input_tensor,
            baseline      = baseline,
            target_class  = target_class,
            steps         = steps,
            batch_size    = batch_size,
            f_input       = f_input,    # pass cached value to avoid re-running model
        )

        # Squeeze [1, F] → [F] so the user gets a simple 1D array.
        # .cpu() moves from GPU to CPU (safe even if already on CPU).
        # .numpy() converts torch.Tensor to np.ndarray.
        attr_np  = attributions.squeeze(0).cpu().numpy()   # [F]
        input_np = input_tensor.squeeze(0).cpu().numpy()   # [F]
        base_np  = baseline.squeeze(0).cpu().numpy()        # [F]

        # ── STEP 7: Resolve top_k ─────────────────────────────────────────────
        # If top_k not given, show all features.
        # Clamp to F so top_k=100 doesn't crash on a 7-feature dataset.
        k = min(top_k if top_k is not None else F, F)

        # ── STEP 8: Build all visualisations and return ───────────────────────
        return {
            "target_class":       target_class,
            "convergence_delta":  convergence_delta,
            "feature_names":      names,
            "attributions":       attr_np,
            "input_values":       input_np,
            "baseline_values":    base_np,
            "training_data":      training_data,
            "top_k":              k,
            # Each _plot_* method returns a matplotlib Figure object.
            "chart_bar":          self._plot_bar(
                                      attr_np, names, target_class, k),
            "chart_force":        self._plot_force(
                                      attr_np, names, target_class, k,
                                      input_np, f_input, float(attr_np.sum())),
            "chart_waterfall":    self._plot_waterfall(
                                      attr_np, names, target_class, k),
            "chart_distribution": self._plot_distribution(
                                      attr_np, names, target_class, k,
                                      input_np, base_np, training_data),
            "chart_summary":      self._plot_summary(
                                      attr_np, names, target_class, k),
        }

    # =========================================================================
    # PUBLIC METHOD: save_dashboard()
    # =========================================================================

    def save_dashboard(
        self,
        results:    Dict,
        save_path:  str,
        class_name: Optional[str] = None,
        dpi:        int           = 150,
    ) -> None:
        """
        Combine all five charts into one PNG and save it to disk.

        Layout (2 rows):
            Row 1 (2 columns): [ Bar chart (wide) ]  [ Force plot (wide) ]
            Row 2 (3 columns): [ Waterfall ] [ Distribution ] [ Summary table ]

        Args:
            results:    The dictionary returned by explain().
            save_path:  File path for the output PNG, e.g. "explanation.png".
            class_name: Human-readable label for the class, e.g. "has_disease".
                        If None, uses "Class 0", "Class 1", etc.
            dpi:        Dots per inch. 150 = good screen quality.
                        Use 300 for print-quality output.
        """
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        delta  = results["convergence_delta"]
        label  = class_name or f"Class {results['target_class']}"

        # Human-readable quality hint in the title based on convergence delta.
        if delta < 0.05:
            quality = "EXCELLENT"
        elif delta < 0.15:
            quality = "OK"
        else:
            quality = "!! increase steps"

        # ── Create the outer figure ───────────────────────────────────────────
        fig = plt.figure(figsize=(20, 12), facecolor="#111122")
        fig.suptitle(
            f"Integrated Gradients — Tabular Explanation\n"
            f"{label}  |  Convergence Δ = {delta:.4f}  [{quality}]",
            color="white", fontsize=13, fontweight="bold", y=0.99,
        )

        # ── Create a 2-row grid layout ────────────────────────────────────────
        # GridSpec divides the figure into rows and columns.
        # outer[0] = top row (will be split into 2 columns)
        # outer[1] = bottom row (will be split into 3 columns)
        # hspace = vertical space between rows
        outer = GridSpec(2, 1, figure=fig, hspace=0.38, top=0.93, bottom=0.03)

        # Split the top row into 2 equal columns
        top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.25)

        # Split the bottom row into 3 equal columns
        bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.30)

        # Match each grid slot to the chart that goes in it
        slots  = [top[0],    top[1],         bottom[0],         bottom[1],          bottom[2]]
        charts = ["chart_bar", "chart_force", "chart_waterfall", "chart_distribution", "chart_summary"]

        for slot, key in zip(slots, charts):
            # Add an axes at this grid slot in the outer figure
            ax = fig.add_subplot(slot)

            # Get the source Figure for this chart
            src_fig = results[key]

            # Render the source figure to its internal canvas buffer.
            # Without this call the buffer might be empty/stale.
            src_fig.canvas.draw()

            # Read the RGBA pixel buffer from the rendered figure.
            # buffer_rgba() returns raw bytes: R,G,B,A for every pixel.
            # reshape(h, w, 4) turns it into a proper image array.
            w, h = src_fig.canvas.get_width_height()
            img  = np.frombuffer(src_fig.canvas.buffer_rgba(),
                                 dtype=np.uint8).reshape(h, w, 4)

            # Display the rendered image inside our outer axes slot.
            ax.imshow(img)
            ax.axis("off")  # no ticks or borders around the embedded image

        # Save the combined figure to disk and release memory.
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)  # close the outer figure

        # Close all individual chart figures to free memory.
        # If we don't do this, matplotlib accumulates open figures
        # which can cause memory warnings on long-running sessions.
        for key in charts:
            plt.close(results[key])

    # =========================================================================
    # PRIVATE METHOD: _safe_forward()
    # NORMALISES MODEL OUTPUT SHAPE BEFORE ANY INDEXING
    # =========================================================================

    def _safe_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and normalise the output to shape [batch, num_classes].

        WHY THIS EXISTS
        ----------------
        There are two standard ways to write a binary classifier in PyTorch.
        Both are correct. Both are widely used. They produce different shapes:

            Style A — two output neurons (what IG expects natively):
                nn.Linear(32, 2)
                output shape: [batch, 2]
                meaning: [logit_class0, logit_class1] for each sample

            Style B — one output neuron with sigmoid (also very common):
                nn.Linear(32, 1) followed by nn.Sigmoid()
                output shape: [batch, 1]
                meaning: P(class=1) for each sample, a single probability

        Without this helper, every line in this class that does
        output[:, target_class] would crash or give wrong results for Style B.

        This method converts Style B output [batch, 1] to [batch, 2] so that
        target_class=0 and target_class=1 both work correctly throughout.

        The conversion:
            p          = sigmoid output (probability of class 1)
            class 1    = p
            class 0    = 1 - p  (complement probability)
            result     = [[1-p, p], [1-p, p], ...]  shape [batch, 2]

        This conversion is mathematically valid because:
            argmax([1-p, p]) = 1 when p > 0.5  (predicts class 1)
            argmax([1-p, p]) = 0 when p < 0.5  (predicts class 0)
        Which matches the original sigmoid decision rule exactly.

        Args:
            x: Input tensor of shape [batch, num_features].

        Returns:
            Output tensor of shape [batch, num_classes], guaranteed 2D,
            with num_classes >= 2.
        """
        output = self.model(x)

        if output.dim() == 2 and output.shape[1] == 1:
            # Style B detected: [batch, 1] → convert to [batch, 2]
            # p is the sigmoid probability of class 1
            p      = output                        # [batch, 1]
            output = torch.cat([1 - p, p], dim=1) # [batch, 2]

        return output

    # =========================================================================
    # PRIVATE METHOD: _compute_ig()
    # THE CORE ALGORITHM
    # =========================================================================

    def _compute_ig(
        self,
        input_tensor: torch.Tensor,    # shape [1, F]
        baseline:     torch.Tensor,    # shape [1, F]
        target_class: int,
        steps:        int,
        batch_size:   int,
        f_input:      float,           # cached: model(input)[target_class]
    ):
        """
        Compute Integrated Gradients using a Riemann sum approximation.

        THE MATH
        ---------
        IG is defined as:
            IG(x) = (x - baseline) × ∫₀¹ ∂F(baseline + α(x-baseline)) / ∂x  dα

        In English:
            - Walk from baseline to input in tiny steps (controlled by alpha α).
            - At each step, compute how sensitive the model output is to the input
              (that's the gradient ∂F/∂x).
            - Average those gradients over all the steps.
            - Multiply by (input - baseline) to get the final attribution.

        The integral is approximated by a Riemann sum with `steps` equally
        spaced points. More steps = closer to the true integral.

        THE ONLY DIFFERENCE FROM IMAGE IG
        -----------------------------------
        In IGImageExplainer the input is [1, C, H, W] (4D).
        Here the input is [1, F] (2D).
        The only change is the broadcast shape for alphas:
            Image:   batch_alphas.view(-1, 1, 1, 1)   ← spreads over [C, H, W]
            Tabular: batch_alphas.view(-1, 1)          ← spreads over [F]
        Every other line in this function is identical to the image version.

        CONVERGENCE DELTA
        ------------------
        The completeness axiom states:
            sum(attributions) = F(input) - F(baseline)

        Delta measures how much our approximation violates this:
            delta = |sum(attributions) - (F(input) - F(baseline))|

        delta = 0 means perfect.
        delta > 0.15 means the Riemann sum is not fine-grained enough → increase steps.

        Args:
            input_tensor: The sample we are explaining.    Shape [1, F].
            baseline:     The reference point.             Shape [1, F].
            target_class: The class we compute attribution for.
            steps:        Number of Riemann sum steps.
            batch_size:   How many steps to process per forward pass.
            f_input:      Cached model(input)[target_class] score. Passed in so
                          we don't run the model an extra time just for the delta.

        Returns:
            attributions: torch.Tensor of shape [1, F].
            delta:        float, the convergence error.
        """
        # Always enforce eval mode inside the core computation.
        # This protects against the case where user calls model.train()
        # between __init__ and explain().
        self.model.eval()

        # path_diff: the vector from baseline to input.
        # This is the "path" we walk along in the Riemann sum.
        # shape: [1, F]
        path_diff = input_tensor - baseline

        # alphas: evenly spaced from just above 0 to 1.
        # torch.linspace(0, 1, steps+1) gives [0, 0.005, 0.010, ..., 1.0]
        # [1:] drops alpha=0 (which is the baseline itself, adds no gradient info).
        # So alphas = [0.005, 0.010, ..., 1.000]  — exactly `steps` values.
        # shape: [steps]
        alphas = torch.linspace(0, 1, steps + 1)[1:].to(self.device)

        # We accumulate gradients across all batches here.
        # Starts at zero; each batch adds to it.
        # shape: [1, F]
        accum_grad = torch.zeros_like(input_tensor)

        # Process the `steps` alpha values in chunks of size batch_size.
        # e.g. steps=200, batch_size=64 → 4 batches of sizes 64, 64, 64, 8.
        for start in range(0, steps, batch_size):

            # Slice out this batch's alpha values.
            # shape: [b] where b = min(batch_size, remaining steps)
            batch_alphas = alphas[start: start + batch_size]

            # Build one interpolated sample per alpha value.
            # batch_alphas.view(-1, 1) reshapes [b] → [b, 1]
            # so it broadcasts correctly with path_diff [1, F] → result [b, F].
            # interpolated[i] = baseline + alpha[i] * (input - baseline)
            # interpolated[0] ≈ baseline   (alpha near 0, close to average patient)
            # interpolated[-1] = input     (alpha = 1, this exact patient)
            # .clone().detach() creates a new tensor with no connection to any
            # previous computation graph.
            # .requires_grad_(True) tells PyTorch: "I want gradients w.r.t. this."
            interpolated = (
                baseline + batch_alphas.view(-1, 1) * path_diff
            ).clone().detach().requires_grad_(True)  # shape: [b, F]

            # Forward pass: run all b interpolated samples through the model.
            # We call _safe_forward() instead of self.model() directly.
            # _safe_forward normalises the output shape so that:
            #   [b, num_classes] → returned as-is
            #   [b, 1]           → converted to [b, 2] (binary sigmoid case)
            # This ensures output[:, target_class] always works correctly
            # regardless of which output style the user's model uses.
            output = self._safe_forward(interpolated)

            # We want the gradient of the target class score.
            # .sum() across the batch collapses b separate scores into a scalar.
            # This is valid because each interpolated[i] is independent —
            # there's no cross-sample interaction in a standard tabular MLP.
            # (Would break for models with cross-sample ops like train-mode BatchNorm.)
            score = output[:, target_class].sum()   # scalar

            # Compute gradients: ∂score / ∂interpolated
            # This gives the sensitivity of the target class score at each
            # interpolated point, for each feature.
            # grads[i, j] = "how much does class score change if feature j
            #               changes by a tiny amount at interpolation step i?"
            # shape: [b, F]
            grads = torch.autograd.grad(score, interpolated)[0]

            # Accumulate: sum the gradients across the batch dimension.
            # .detach() removes gradients from the accumulator to prevent
            # memory growing across iterations.
            # .sum(dim=0, keepdim=True) reduces [b, F] → [1, F]
            accum_grad += grads.detach().sum(dim=0, keepdim=True)

            # Explicitly delete intermediate tensors to free GPU/CPU memory.
            # Python's garbage collector will eventually do this, but for large
            # models or many steps, explicit deletion prevents memory buildup.
            del interpolated, output, score, grads

        # Final IG formula:
        # (1) Divide accumulated gradients by steps → average gradient along path
        # (2) Multiply by path_diff (input - baseline) → weight by how far each
        #     feature actually moved from baseline to input
        # shape: [1, F]
        attributions = path_diff * (accum_grad / steps)

        # ── Convergence delta ─────────────────────────────────────────────────
        # Compute F(baseline) for the completeness check.
        # We use the cached f_input (model(input)[target_class]) so we only
        # need one extra forward pass here (for the baseline), not two.
        with torch.no_grad():
            # Use _safe_forward so [batch,1] models are handled consistently
            f_baseline = float(self._safe_forward(baseline)[0, target_class].item())

        # delta = |sum of attributions - (model output at input - model output at baseline)|
        # Perfect completeness → delta = 0.
        delta = abs(float(attributions.sum().item()) - (f_input - f_baseline))

        return attributions.detach(), delta

    # =========================================================================
    # PRIVATE: Plot 1 — Bar Chart
    # =========================================================================

    def _plot_bar(
        self,
        attr:         np.ndarray,   # [F] attribution scores
        names:        List[str],    # feature names
        target_class: int,
        k:            int,          # how many features to show
    ) -> plt.Figure:
        """
        Horizontal bar chart of signed attributions.

        This is the primary visualisation — the first chart you should look at.

        How to read it:
            Each row = one feature.
            Bar goes RIGHT (green)  = this feature SUPPORTS the prediction.
            Bar goes LEFT (red)     = this feature SUPPRESSES the prediction.
            Bar LENGTH              = how strongly the feature matters.
            Features are sorted with the most important at the TOP.

        Example reading:
            age       ████████████  +0.32   → high age strongly supports "disease"
            chol      ██████████    +0.28   → high cholesterol also supports it
            heart_rate ████         -0.11   → good heart rate pushes against "disease"
        """
        # Get top-k feature indices sorted by |attribution| descending
        idx = _get_top_k_indices(attr, k)

        # Extract and reverse so the most important feature is at the TOP of
        # the horizontal bar chart (matplotlib draws bottom-to-top by default)
        vals   = attr[idx][::-1]
        labels = [names[i] for i in idx][::-1]

        # Green for positive attributions, red for negative
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]

        # Height scales with number of features so bars don't get squished
        fig, ax = plt.subplots(
            figsize=(7, max(3, k * 0.45 + 1)),
            facecolor="#111122",
        )
        _apply_dark_theme(ax)

        # Draw horizontal bars.
        # vals can be positive or negative — barh handles both correctly.
        bars    = ax.barh(labels, vals, color=colors, edgecolor="#2a2a4a", height=0.65)
        max_abs = float(np.abs(vals).max()) or 1.0   # avoid division by zero

        # Add the numeric value label at the end of each bar
        for bar, v in zip(bars, vals):
            offset = max_abs * 0.025    # small gap between bar end and label
            x_pos  = v + offset if v >= 0 else v - offset
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,  # vertical centre of bar
                f"{v:+.4f}",    # always show + or - sign
                va="center",
                ha="left"  if v >= 0 else "right",
                color="white", fontsize=7.5, fontweight="bold",
            )

        # Vertical line at x=0 to make it easy to see positive vs negative
        ax.axvline(0, color="#aaaacc", lw=0.8, alpha=0.6)
        ax.set_xlabel("Attribution Score")
        ax.set_title(
            f"Feature Attributions (Top {k})  |  Class {target_class}",
            fontsize=10, fontweight="bold", pad=8,
        )

        # Legend patches (coloured squares with labels)
        ax.legend(
            handles=[
                mpatches.Patch(color="#2ecc71", label="Supports prediction"),
                mpatches.Patch(color="#e74c3c", label="Suppresses prediction"),
            ],
            loc="lower right",
            facecolor="#111122", edgecolor="#444466", labelcolor="#ccccee",
            fontsize=7.5,
        )

        plt.tight_layout()
        return fig

    # =========================================================================
    # PRIVATE: Plot 2 — Force Plot
    # =========================================================================

    def _plot_force(
        self,
        attr:           np.ndarray,
        names:          List[str],
        target_class:   int,
        k:              int,
        in_np:          np.ndarray,   # actual feature values of this sample
        f_input:        float,        # model's predicted score for this sample
        full_attr_sum:  float = None, # sum of ALL feature attributions (not just top-k)
    ) -> plt.Figure:
        """
        Force plot — the most intuitive way to explain a single prediction.

        WHAT THIS SHOWS:
            Imagine a horizontal score line from left to right.
            On the left: the baseline score (what the model predicts for an average patient).
            On the right: the final score (what it predicts for THIS patient).

            Each feature is shown as an arrow:
                Green arrows ABOVE the line → push the score UP (toward class)
                Blue arrows BELOW the line  → push the score DOWN (away from class)
                Arrow height                → magnitude of the push

        WHY THIS IS MORE USEFUL THAN THE BAR CHART:
            The bar chart tells you attribution magnitudes.
            The force plot tells you the STORY:
                "I started at 0.45 for the average patient.
                 Then age pushed me to 0.77.
                 Then cholesterol pushed me to 0.95.
                 But heart rate pulled me back to 0.84."
            That narrative is what makes a prediction explainable to a doctor
            or a loan officer.

        This is the visualisation SHAP made famous. It is the single best
        chart for explaining one specific prediction.
        """
        # Get top-k features
        idx     = _get_top_k_indices(attr, k)
        vals    = attr[idx]
        labels  = [names[i] for i in idx]
        invals  = in_np[idx]            # actual values for this sample
        max_abs = float(np.abs(vals).max()) or 1.0

        # Approximate baseline score = final score minus TOTAL attribution sum.
        # Completeness axiom: sum(ALL attributions) = F(input) - F(baseline)
        # Therefore: F(baseline) = F(input) - sum(ALL attributions)
        #
        # IMPORTANT: we must use full_attr_sum (all features), NOT attr[idx].sum()
        # (top-k only). If top_k < total features, using only top-k sum would give
        # a wrong baseline score. Example with 20 features but k=5:
        #   total sum = 0.40, top-5 sum = 0.38, f_input = 0.84
        #   Wrong: 0.84 - 0.38 = 0.46  ← off by 0.02
        #   Right: 0.84 - 0.40 = 0.44  ← correct
        total_sum      = full_attr_sum if full_attr_sum is not None else float(attr.sum())
        baseline_score = f_input - total_sum

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111122")
        _apply_dark_theme(ax)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-2.4, 2.4)
        ax.axis("off")    # no axis ticks — this is a custom diagram, not a standard plot
        ax.set_title(
            f"Force Plot  |  Class {target_class}  |  "
            f"Baseline = {baseline_score:.3f}   →   Prediction = {f_input:.3f}",
            fontsize=11, fontweight="bold", pad=10,
        )

        # ── Central horizontal spine (the score number line) ──────────────────
        # This arrow goes from x=0.07 to x=0.93 at y=0, representing the score axis.
        ax.annotate(
            "",                         # no text label on the arrow itself
            xy=(0.93, 0.0),             # arrowhead at x=0.93
            xytext=(0.07, 0.0),         # arrow starts at x=0.07
            arrowprops=dict(
                arrowstyle="-|>",       # line with solid arrowhead
                color="#555577",
                lw=2.5,
                mutation_scale=16,      # size of the arrowhead
            ),
        )

        # ── Baseline score box (left anchor) ──────────────────────────────────
        ax.text(
            0.04, 0.0,
            f"Baseline\n{baseline_score:.3f}",
            ha="center", va="center",
            color="#ccccee", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555577", lw=1.5),
        )

        # ── Prediction score box (right anchor) ───────────────────────────────
        ax.text(
            0.96, 0.0,
            f"Score\n{f_input:.3f}",
            ha="center", va="center",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0d2b1a", ec="#2ecc71", lw=2.0),
        )

        # ── Feature arrows ────────────────────────────────────────────────────
        # Spread features evenly between x=0.13 and x=0.87.
        xs = np.linspace(0.13, 0.87, max(len(vals), 1))

        for i, (v, label, iv) in enumerate(zip(vals, labels, invals)):
            pos   = v >= 0
            color = "#2ecc71" if pos else "#5588ff"   # green above, blue below
            tc    = "#2ecc71" if pos else "#7aadff"   # text colour
            sign  = 1 if pos else -1                   # +1 = above, -1 = below

            # Stem height: minimum 0.15, maximum ~1.0, proportional to |v|
            stem = sign * (0.15 + abs(v) / max_abs * 0.85)
            x    = xs[i]

            # Draw vertical stem from spine (y=0.10) to the arrow level
            ax.plot([x, x], [sign * 0.10, stem], color=color, lw=1.8)

            # Draw horizontal arrowhead pointing in direction of push
            ax.annotate(
                "",
                xy     = (x + (0.03 if pos else -0.03), stem),
                xytext = (x, stem),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=9),
            )

            # Feature label box above/below the arrow
            ax.text(
                x,
                stem + sign * 0.22,
                f"{label}\n={iv:.2f}\n{v:+.3f}",    # name, actual value, attribution
                ha="center",
                va="bottom" if pos else "top",
                color=tc, fontsize=6.5, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="#0a2a0a" if pos else "#0a0a2a",
                    ec=color, lw=1.0,
                ),
            )

        # ── Footer legend ──────────────────────────────────────────────────────
        ax.text(
            0.5, -2.2,
            "▲ Green (above) pushes score UP   ▼ Blue (below) pushes score DOWN",
            ha="center", color="#aaaacc", fontsize=8, style="italic",
        )

        plt.tight_layout()
        return fig

    # =========================================================================
    # PRIVATE: Plot 3 — Waterfall Chart
    # =========================================================================

    def _plot_waterfall(
        self,
        attr:         np.ndarray,
        names:        List[str],
        target_class: int,
        k:            int,
    ) -> plt.Figure:
        """
        Waterfall chart — shows how attribution accumulates step by step.

        HOW TO READ IT:
            The chart starts at 0 (no attribution yet).
            Each bar adds one feature's attribution to the running total.
            After all bars, the chart ends at the total attribution sum.

            Green bar = this feature's contribution is positive (score goes up)
            Red bar   = this feature's contribution is negative (score goes down)
            Dashed connectors show where each bar picks up from the last.

        "Others" bar:
            If top_k < total features, the remaining features are bundled into
            one "Others" bar so the chart always ends at the true total.
            This ensures the waterfall is honest — nothing is hidden.

        DIFFERENCE FROM BAR CHART:
            Bar chart: "age contributes +0.32 and blood_pressure contributes +0.19"
            Waterfall: "after age we're at +0.32, then after blood_pressure we're
                        at +0.51, then after heart_rate we drop to +0.40..."
            The waterfall shows the cumulative story, not just individual values.
        """
        idx    = _get_top_k_indices(attr, k)
        vals   = list(attr[idx])             # convert to list for .append()
        labels = [names[i] for i in idx]

        # Add "Others" bar if we're not showing all features.
        # Its value = total attribution - sum of what we're already showing.
        if k < len(attr):
            vals.append(float(attr.sum() - sum(vals)))
            labels.append("Others")

        # Compute where each bar starts (its "bottom" in matplotlib terms).
        # running[0] = 0 (chart starts at zero)
        # running[1] = vals[0]
        # running[2] = vals[0] + vals[1]  ... etc.
        running = np.zeros(len(vals) + 1)
        for i, v in enumerate(vals):
            running[i + 1] = running[i] + v

        # Each bar's bottom = the running total BEFORE adding this bar
        bottoms = running[:-1]
        colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
        xs      = np.arange(len(vals))      # x positions: 0, 1, 2, ...

        fig, ax = plt.subplots(
            figsize=(max(5, len(vals) * 0.85 + 1), 5),
            facecolor="#111122",
        )
        _apply_dark_theme(ax)

        # Draw bars. Each bar starts at bottoms[i] and has height vals[i].
        # For negative vals[i], matplotlib automatically draws the bar downward.
        ax.bar(xs, vals, bottom=bottoms, color=colors, edgecolor="#2a2a4a", width=0.62)

        # Dashed connector lines between the top of each bar and the bottom of the next.
        # This visually shows the flow from one step to the next.
        for i in range(len(vals) - 1):
            y_top = bottoms[i] + vals[i]   # where this bar ends
            ax.plot(
                [xs[i] + 0.31, xs[i + 1] - 0.31],  # horizontal connector
                [y_top, y_top],
                color="#7777aa", lw=0.9, ls="--", alpha=0.7,
            )

        # Value labels above each bar
        span = float(np.abs(running).max()) or 1.0   # for proportional offset
        for i, v in enumerate(vals):
            ax.text(
                xs[i],
                bottoms[i] + v + span * 0.03,   # slightly above the bar top
                f"{v:+.3f}",
                ha="center", va="bottom",
                color="white", fontsize=7.5, fontweight="bold",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.axhline(0, color="#aaaacc", lw=0.9, alpha=0.6)  # zero reference line
        ax.set_ylabel("Cumulative Attribution")
        ax.set_title(
            f"Waterfall — Attribution Build-Up  |  Class {target_class}",
            fontsize=10, fontweight="bold", pad=8,
        )
        plt.tight_layout()
        return fig

    # =========================================================================
    # PRIVATE: Plot 4 — Distribution Plot
    # =========================================================================

    def _plot_distribution(
        self,
        attr:          np.ndarray,
        names:         List[str],
        target_class:  int,
        k:             int,
        in_np:         np.ndarray,              # this sample's feature values
        base_np:       np.ndarray,              # baseline feature values
        training_data: Optional[np.ndarray],    # full training set [N, F]
    ) -> plt.Figure:
        """
        Distribution plot — answers WHY a feature has a high attribution.

        THE PROBLEM WITH OTHER CHARTS ALONE:
            They tell you THAT age has attribution +0.32.
            But they don't tell you WHY.
            Is it because age=65 is unusual in your dataset?
            Or because the model is just very sensitive to age even at normal values?
            Without context, you cannot know.

        WHAT THIS CHART ADDS:
            One subplot per top-k feature.
            Each subplot shows:
                Gray histogram  = how this feature is distributed across all training samples
                Orange dashed   = the baseline value (training mean)
                Colored dot     = THIS sample's value (green if +attr, red if -attr)
                Colored line    = vertical line at this sample's value

        READING EXAMPLE:
            age subplot:
                Histogram centered around 55 (most patients are ~55).
                Orange dashed line at 55 (baseline = mean age).
                Green dot at 72 (far right of histogram, in top 5%).
                → This explains WHY age has high attribution: age=72 is unusual.
                  The model rarely sees patients this old, so it has a strong effect.

        WITHOUT training_data (fallback):
            Shows a simple side-by-side bar chart: baseline value vs sample value.
            Less informative but still useful.
        """
        idx   = _get_top_k_indices(attr, k)
        n     = len(idx)

        # Arrange subplots in a grid with at most 4 columns.
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.2, nrows * 2.8 + 0.6),
            facecolor="#111122",
            squeeze=False,   # always return 2D array of axes, even for 1 row/col
        )
        fig.suptitle(
            f"Feature Value vs Training Distribution (Top {k})  |  Class {target_class}\n"
            f"Orange dashed = baseline   Coloured dot = this sample",
            color="white", fontsize=10, fontweight="bold", y=1.02,
        )

        for plot_i, feat_i in enumerate(idx):
            row = plot_i // ncols
            col = plot_i % ncols
            ax  = axes[row][col]
            _apply_dark_theme(ax)

            v        = attr[feat_i]
            dot_color = "#2ecc71" if v >= 0 else "#e74c3c"  # green=supports, red=suppresses

            if training_data is not None:
                # Draw histogram of this feature's training distribution
                ax.hist(
                    training_data[:, feat_i],   # all training values for this feature
                    bins=25,
                    color="#445588", alpha=0.75, edgecolor="#2a2a4a", lw=0.5,
                )
                # IMPORTANT: call get_ylim() AFTER ax.hist(), not before.
                # The histogram sets the y-axis range when it is drawn.
                # Calling get_ylim() before hist() returns the default (0,1)
                # and the dot ends up at y=0.06 regardless of histogram scale.
                y_max = ax.get_ylim()[1]
                # Baseline vertical line (orange dashed)
                ax.axvline(base_np[feat_i],  color="#f39c12", lw=1.8, ls="--")
                # This sample's vertical line
                ax.axvline(in_np[feat_i],    color=dot_color, lw=1.4)
                # Dot at 6% of histogram height (visually at the base of the line)
                ax.scatter(
                    [in_np[feat_i]],
                    [y_max * 0.06],
                    color=dot_color, s=80, zorder=5,
                )
                ax.set_ylabel("Count")
            else:
                # Fallback: simple side-by-side bars
                ax.bar(
                    ["Baseline", "Sample"],
                    [base_np[feat_i], in_np[feat_i]],
                    color=["#f39c12", dot_color],
                    edgecolor="#2a2a4a", width=0.5,
                )
                ax.set_ylabel("Value")

            ax.set_title(
                f"{names[feat_i]}\nattr = {v:+.3f}",
                fontsize=8, fontweight="bold", pad=4,
            )

        # Hide any unused subplot panels
        # (happens when n is not a multiple of ncols)
        for plot_i in range(n, nrows * ncols):
            axes[plot_i // ncols][plot_i % ncols].set_visible(False)

        plt.tight_layout()
        return fig

    # =========================================================================
    # PRIVATE: Plot 5 — Summary Table
    # =========================================================================

    def _plot_summary(
        self,
        attr:         np.ndarray,
        names:        List[str],
        target_class: int,
        k:            int,
    ) -> plt.Figure:
        """
        Summary table — ranked features in report-ready format.

        COLUMNS:
            Rank        : 1 = most important (by absolute attribution)
            Feature     : the feature name
            Attribution : signed score (+0.32 = supports, -0.11 = suppresses)
            Direction   : "▲ Supports" or "▼ Suppresses" (colour coded)
            Importance  : a visual bar made of block characters showing magnitude

        WHEN TO USE THIS INSTEAD OF THE BAR CHART:
            - Writing a report or slide deck (table text is copy-pasteable)
            - Sharing with non-technical stakeholders who find charts confusing
            - When you need exact numbers AND visual ranking in the same place
            - For audit logs that need to record exact attribution values

        ROW COLOURS:
            Dark green background = positive attribution (supports prediction)
            Dark red background   = negative attribution (suppresses prediction)
        """
        idx     = _get_top_k_indices(attr, k)
        vals    = attr[idx]
        labels  = [names[i] for i in idx]
        abs_max = float(np.abs(vals).max()) or 1.0  # for normalising importance bar

        rows        = []
        cell_colors = []

        for rank, (name, v) in enumerate(zip(labels, vals), start=1):
            direction   = "▲ Supports"  if v >= 0 else "▼ Suppresses"

            # Importance bar: scale the count of █ characters by relative magnitude.
            # max(1, ...) ensures at least one block even for tiny values.
            block_count = max(1, int(abs(v) / abs_max * 12))
            importance  = "█" * block_count

            rows.append([str(rank), name, f"{v:+.4f}", direction, importance])

            # Row background: dark green for positive, dark red for negative
            row_bg = "#0d2b1a" if v >= 0 else "#2b0d0d"
            cell_colors.append([row_bg] * 5)

        # Figure height scales with number of rows
        fig, ax = plt.subplots(
            figsize=(8, max(2.8, k * 0.42 + 1.5)),
            facecolor="#111122",
        )
        ax.set_facecolor("#111122")
        ax.axis("off")   # table is drawn as a matplotlib table object, not on axes

        # Prepend header row to the data
        header      = [["Rank", "Feature", "Attribution", "Direction", "Importance"]]
        header_bg   = [["#2c2c4e"] * 5]     # dark blue header background

        table = ax.table(
            cellText    = header + rows,
            cellColours = header_bg + cell_colors,
            cellLoc     = "center",
            loc         = "center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)   # make rows taller (1.45x default height)

        # Style the header row (row index 0)
        for col in range(5):
            table[0, col].set_text_props(color="white", fontweight="bold")
            table[0, col].set_edgecolor("#555577")

        # Style each data row (rows 1 to k)
        for row in range(1, len(rows) + 1):
            is_positive = rows[row - 1][2].startswith("+")  # check attribution sign
            for col in range(5):
                cell = table[row, col]
                cell.set_edgecolor("#333355")
                # Colour Direction and Importance columns green/red
                if col in (3, 4):
                    cell.set_text_props(
                        color="#2ecc71" if is_positive else "#e74c3c",
                        fontweight="bold" if col == 3 else "normal",
                    )
                else:
                    cell.set_text_props(color="white")

        ax.set_title(
            f"Feature Importance Summary (Top {k})  |  Class {target_class}",
            color="white", fontsize=10, fontweight="bold", pad=10,
        )
        plt.tight_layout()
        return fig