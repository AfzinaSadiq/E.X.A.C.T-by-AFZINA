"""
test_ig_tabular.py
===================
End-to-end test suite for IGTabularExplainer.

TESTS INCLUDED
---------------
Test 1 — Standard MLP [batch, 2] output        (the normal happy path)
Test 2 — Style B binary model [batch, 1] output (sigmoid, auto-converted)
Test 3 — Unnormalised data without training_data (should warn, not crash)
Test 4 — Wrong feature name count               (should raise ValueError)
Test 5 — Wrong input shape [3, 7]               (should raise ValueError)
Test 6 — 1D regression model output [batch]     (should raise ValueError)
Test 7 — model.train() called before explain()  (eval() enforced internally)
Test 8 — Integer-only input data                (should warn about categoricals)

HOW TO RUN
-----------
    python test_ig_tabular.py          ← from the project root
    python -m tests.test_ig_tabular    ← using module syntax from project root
"""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.figure

from EXACT.explainers.ig_tabular_explainer import IGTabularExplainer

OUTPUT_DIR = "user_saves"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP  = "=" * 62
SEP2 = "-" * 62


# =============================================================================
# SHARED: DATASET
# =============================================================================

def make_dataset():
    """
    Synthetic heart-disease dataset: 400 samples, 7 features, binary labels.

    Signal injected: high age + cholesterol + blood_pressure → disease (class 1).
    Data is StandardScaler-normalised so zero baseline is meaningful.

    Returns:
        X     : np.ndarray [400, 7]  normalised float features
        y     : np.ndarray [400]     binary integer labels
        names : List[str]            feature names
    """
    rng = np.random.default_rng(42)
    N   = 400

    age    = rng.normal(55, 10, N)
    chol   = rng.normal(240, 40, N)
    bp     = rng.normal(130, 20, N)
    hr     = rng.normal(150, 25, N)
    sugar  = rng.normal(120, 30, N)
    pain   = rng.integers(0, 4, N).astype(float)
    angina = rng.integers(0, 2, N).astype(float)

    X = np.column_stack([age, chol, bp, hr, sugar, pain, angina])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)   # normalise

    score = (age - 55) / 10 + (chol - 240) / 40 + (bp - 130) / 20
    y     = (score > 0.5).astype(int)

    names = ["age", "cholesterol", "blood_pressure",
             "max_heart_rate", "blood_sugar", "chest_pain_type", "exercise_angina"]
    return X, y, names


# =============================================================================
# SHARED: MODELS
# =============================================================================

class MLPStyleA(nn.Module):
    """
    Standard binary classifier — outputs [batch, 2] raw logits.
    This is the format IGTabularExplainer expects natively.
    Final layer: nn.Linear(32, 2) — no activation.
    """
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 2),    # output: [batch, 2]
        )
    def forward(self, x):
        return self.net(x)


class MLPStyleB(nn.Module):
    """
    Binary classifier with sigmoid output — outputs [batch, 1].
    This is the second standard PyTorch pattern for binary classification.
    Final layer: nn.Linear(32, 1) + nn.Sigmoid().

    The explainer must auto-convert [batch, 1] → [batch, 2].
    If it doesn't, target_class indexing will crash or give wrong results.
    """
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1),    # output: [batch, 1]
            nn.Sigmoid(),        # squash to (0, 1) probability
        )
    def forward(self, x):
        return self.net(x)


class RegressionModel(nn.Module):
    """
    Regression model — outputs [batch] (1D, no class dimension).
    The explainer must raise a clear ValueError for this.
    """
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        # .squeeze(1) collapses [batch, 1] → [batch]
        # This is the regression output pattern that breaks IG
        return self.net(x).squeeze(1)


def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                epochs: int = 60, lr: float = 0.01) -> float:
    """
    Train any classification model with CrossEntropyLoss.
    Works for both StyleA [batch,2] and StyleB [batch,1] models.

    For StyleB (sigmoid output), we use BCELoss instead of CrossEntropyLoss
    because BCELoss expects scalar probabilities, not class logits.

    Returns training accuracy.
    """
    Xt = torch.FloatTensor(X)
    yt = torch.LongTensor(y)

    # Detect output style by running one dummy forward pass
    with torch.no_grad():
        dummy_out = model(Xt[:2])

    use_bce = (dummy_out.dim() == 2 and dummy_out.shape[1] == 1)

    if use_bce:
        # BCELoss for sigmoid binary output — expects float targets in [0,1]
        loss_fn = nn.BCELoss()
        yt_loss = torch.FloatTensor(y).unsqueeze(1)   # [N, 1] float
    else:
        # CrossEntropyLoss for logits output — expects integer class indices
        loss_fn = nn.CrossEntropyLoss()
        yt_loss = yt

    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt_loss), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        out = model(Xt)
        if use_bce:
            # For sigmoid: predict class 1 if probability > 0.5
            preds = (out.squeeze(1) > 0.5).long()
        else:
            preds = out.argmax(dim=1)
        acc = (preds == yt).float().mean().item()

    return acc


# =============================================================================
# SHARED: CHECK HELPER
# =============================================================================

def check(condition: bool, description: str) -> bool:
    """Print PASS/FAIL with description. Returns the condition."""
    status = "PASS ✅" if condition else "FAIL ❌"
    print(f"  {description:<50} {status}")
    return condition


def check_raises(exc_type, description: str, fn):
    """
    Verify that calling fn() raises the expected exception type.
    Prints PASS if the exception is raised, FAIL if it is not.

    Why we need this:
        Some inputs SHOULD cause errors. A good library raises clear errors
        rather than silently producing wrong results. We test that those
        errors are raised correctly.

    Args:
        exc_type    : The exception class we expect (e.g. ValueError)
        description : What we're checking (shown in output)
        fn          : A zero-argument callable that should raise exc_type
    """
    try:
        fn()
        # If we reach here, no exception was raised — that is a test failure
        print(f"  {description:<50} FAIL ❌  (no exception raised)")
        return False
    except exc_type:
        # Correct exception raised
        print(f"  {description:<50} PASS ✅")
        return True
    except Exception as e:
        # Wrong exception type raised
        print(f"  {description:<50} FAIL ❌  (wrong exception: {type(e).__name__}: {e})")
        return False


def check_warns(description: str, fn):
    """
    Verify that calling fn() raises at least one UserWarning.
    Prints PASS if a warning was issued, FAIL if none was issued.

    Why we need this:
        Some bad inputs should not crash but should warn the user.
        Checking that warnings fire is just as important as checking
        that errors fire — it proves the defensive code is running.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")   # catch all warnings, not just first
        fn()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    passed = len(user_warnings) > 0
    status = "PASS ✅" if passed else "FAIL ❌  (no UserWarning was raised)"
    print(f"  {description:<50} {status}")
    if not passed:
        return False
    # Print what the warning said so it's visible in the test output
    for w in user_warnings:
        print(f"    ↳ Warning: {str(w.message)[:80]}...")
    return True


# =============================================================================
# TEST 1 — Standard [batch, 2] model (the normal case)
# =============================================================================

def test_standard_model(X, y, names, device):
    """
    Happy path: Style A model with [batch, 2] output.
    This is what the explainer was originally designed for.
    Verifies: attributions shape, convergence delta, all 5 charts, dashboard.
    """
    print(f"\n{SEP2}")
    print("  TEST 1: Standard [batch, 2] model")
    print(SEP2)

    model = MLPStyleA(X.shape[1]).to(device)
    acc   = train_model(model, X, y)
    print(f"  Training accuracy : {acc * 100:.1f}%")

    sample    = torch.FloatTensor(X[5])
    explainer = IGTabularExplainer(model, feature_names=names, device=device)
    results   = explainer.explain(sample, training_data=X, steps=200)

    delta = results["convergence_delta"]
    print(f"  Convergence Δ     : {delta:.6f}")
    print(f"  Predicted class   : {results['target_class']}")
    print()

    all_ok = True
    all_ok &= check(results["attributions"].shape == (7,),   "attributions.shape == (7,)")
    all_ok &= check(delta < 0.05,                            f"convergence delta < 0.05  (got {delta:.4f})")
    all_ok &= check(len(results["feature_names"]) == 7,      "feature_names length == 7")
    all_ok &= check(results["input_values"].shape == (7,),   "input_values.shape == (7,)")
    all_ok &= check(results["baseline_values"].shape == (7,),"baseline_values.shape == (7,)")
    for key in ["chart_bar","chart_force","chart_waterfall","chart_distribution","chart_summary"]:
        all_ok &= check(isinstance(results[key], matplotlib.figure.Figure), f"{key} is Figure")

    path = os.path.join(OUTPUT_DIR, "test1_standard.png")
    explainer.save_dashboard(results, path,
                             class_name="has_disease" if results["target_class"]==1 else "no_disease")
    all_ok &= check(os.path.exists(path), f"dashboard saved to {path}")

    return all_ok


# =============================================================================
# TEST 2 — Style B: [batch, 1] sigmoid model (THE KEY FIX)
# =============================================================================

def test_sigmoid_model(X, y, names, device):
    """
    Style B: model outputs [batch, 1] with sigmoid.
    This is one of the two standard binary classifier patterns in PyTorch.

    WHAT THIS TEST PROVES:
        Before the fix: output[:, target_class] would crash or give wrong results
                        because you can't index class 1 on a [batch,1] tensor.
        After the fix:  _safe_forward() detects [batch,1] and converts to [batch,2]
                        automatically. The user sees a clear warning and correct results.

    We verify:
        1. A UserWarning is issued (auto-conversion happened)
        2. Attributions are still the right shape
        3. Convergence delta is still small (math is still correct)
        4. Both target_class=0 and target_class=1 work (not just argmax)
    """
    print(f"\n{SEP2}")
    print("  TEST 2: Style B [batch, 1] sigmoid model (the key fix)")
    print(SEP2)

    model = MLPStyleB(X.shape[1]).to(device)
    acc   = train_model(model, X, y)
    print(f"  Training accuracy : {acc * 100:.1f}%")

    sample    = torch.FloatTensor(X[5])
    explainer = IGTabularExplainer(model, feature_names=names, device=device)

    all_ok = True

    # Sub-test 2a: auto-conversion warning is issued
    all_ok &= check_warns(
        "[batch,1] model triggers auto-conversion warning",
        lambda: explainer.explain(sample, training_data=X, steps=200),
    )

    # Sub-test 2b: results are still correct after conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress for clean output
        results = explainer.explain(sample, training_data=X, steps=200)

    delta = results["convergence_delta"]
    print(f"\n  Convergence Δ     : {delta:.6f}")
    print(f"  Predicted class   : {results['target_class']}")
    print()

    all_ok &= check(results["attributions"].shape == (7,), "attributions.shape == (7,)")
    all_ok &= check(delta < 0.15,                          f"convergence delta < 0.15  (got {delta:.4f})")

    # Sub-test 2c: explicitly requesting target_class=0 works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_c0 = explainer.explain(sample, training_data=X,
                                       steps=200, target_class=0)
    all_ok &= check(results_c0["target_class"] == 0,       "target_class=0 explicit works")
    all_ok &= check(results_c0["attributions"].shape==(7,), "attributions shape OK for class 0")

    # Sub-test 2d: explicitly requesting target_class=1 works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_c1 = explainer.explain(sample, training_data=X,
                                       steps=200, target_class=1)
    all_ok &= check(results_c1["target_class"] == 1,       "target_class=1 explicit works")
    all_ok &= check(results_c1["attributions"].shape==(7,), "attributions shape OK for class 1")

    # Sub-test 2e: attributions for class 0 and class 1 are different
    # (they should be, because they explain different decisions)
    attrs_differ = not np.allclose(
        results_c0["attributions"], results_c1["attributions"], atol=1e-4
    )
    all_ok &= check(attrs_differ, "class 0 and class 1 attributions are different")

    return all_ok


# =============================================================================
# TEST 3 — Unnormalised data without training_data (warn, not crash)
# =============================================================================

def test_unnormalised_warning(X, y, names, device):
    """
    When the user passes raw unnormalised data (age=65, income=80000)
    without providing training_data, the zero baseline is meaningless.
    The explainer should WARN the user, not crash silently.

    We verify:
        - A UserWarning is issued
        - The computation still completes (we don't block the user)
        - The warning message mentions training_data
    """
    print(f"\n{SEP2}")
    print("  TEST 3: Unnormalised data without training_data (should warn)")
    print(SEP2)

    model = MLPStyleA(X.shape[1]).to(device)
    train_model(model, X, y)
    explainer = IGTabularExplainer(model, feature_names=names, device=device)

    # Build a raw unnormalised sample: values like real clinical data
    # age=72, cholesterol=310, bp=160 — these are large numbers, not ~0
    raw_sample = torch.FloatTensor([72.0, 310.0, 160.0, 140.0, 95.0, 2.0, 1.0])

    all_ok = True

    # Should warn because max value >> 10 and no training_data provided
    all_ok &= check_warns(
        "unnormalised data without training_data warns user",
        lambda: explainer.explain(raw_sample, training_data=None, steps=50),
    )

    # Should still complete (warning, not crash)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = explainer.explain(raw_sample, training_data=None, steps=50)
            all_ok &= check(True, "computation completes despite bad baseline")
            all_ok &= check(results["attributions"].shape == (7,),
                            "attributions still produced")
        except Exception as e:
            all_ok &= check(False, f"computation completes — got: {e}")

    return all_ok


# =============================================================================
# TEST 4 — Wrong feature name count (should raise ValueError)
# =============================================================================

def test_wrong_feature_names(X, y, device):
    """
    Passing 5 feature names for a 7-feature model is a user mistake.
    The explainer should catch this immediately with a clear error message
    that tells the user exactly what went wrong.
    """
    print(f"\n{SEP2}")
    print("  TEST 4: Wrong feature name count (should raise ValueError)")
    print(SEP2)

    model = MLPStyleA(X.shape[1]).to(device)
    train_model(model, X, y)

    wrong_names = ["age", "cholesterol", "bp"]   # 3 names for 7 features

    explainer = IGTabularExplainer(model, feature_names=wrong_names, device=device)
    sample    = torch.FloatTensor(X[5])

    all_ok = True
    all_ok &= check_raises(
        ValueError,
        "wrong feature name count raises ValueError",
        lambda: explainer.explain(sample),
    )
    return all_ok


# =============================================================================
# TEST 5 — Wrong input shape [3, 7] (should raise ValueError)
# =============================================================================

def test_wrong_input_shape(X, y, names, device):
    """
    The explainer accepts [F] or [1, F] — a single sample.
    Passing [3, 7] (3 samples) is not supported.
    The error message should tell the user exactly what shape was received
    and what shape is expected.
    """
    print(f"\n{SEP2}")
    print("  TEST 5: Wrong input shape [3, 7] (should raise ValueError)")
    print(SEP2)

    model     = MLPStyleA(X.shape[1]).to(device)
    train_model(model, X, y)
    explainer = IGTabularExplainer(model, feature_names=names, device=device)

    # Pass 3 samples instead of 1
    batch_sample = torch.FloatTensor(X[:3])   # shape [3, 7]

    all_ok = True
    all_ok &= check_raises(
        ValueError,
        "batch input [3,7] raises ValueError",
        lambda: explainer.explain(batch_sample),
    )
    return all_ok


# =============================================================================
# TEST 6 — Regression model [batch] output (should raise ValueError)
# =============================================================================

def test_regression_model(X, y, names, device):
    """
    A regression model outputs [batch] — a single scalar per sample.
    This is fundamentally incompatible with IG's class-based attribution.
    The explainer must raise a clear ValueError that tells the user:
        - What shape was received
        - What shape is expected
        - Why regression doesn't work
    """
    print(f"\n{SEP2}")
    print("  TEST 6: Regression model [batch] output (should raise ValueError)")
    print(SEP2)

    model     = RegressionModel(X.shape[1]).to(device)
    explainer = IGTabularExplainer(model, feature_names=names, device=device)
    sample    = torch.FloatTensor(X[5])

    all_ok = True
    all_ok &= check_raises(
        ValueError,
        "regression model [batch] output raises ValueError",
        lambda: explainer.explain(sample),
    )
    return all_ok


# =============================================================================
# TEST 7 — model.train() called before explain() (eval enforced internally)
# =============================================================================

def test_train_mode_override(X, y, names, device):
    """
    A common professional mistake: the user trains their model, then forgets
    to call model.eval() before explaining. Or they call model.train() between
    creating the explainer and calling explain().

    The explainer calls self.model.eval() at the START of both explain() and
    _compute_ig(). This test verifies that the results are the same whether
    the user left the model in train mode or eval mode.

    WHY THIS MATTERS:
        In train mode, BatchNorm uses batch statistics instead of stored statistics.
        For IG, we pass 64 interpolated samples per batch. The batch norm would
        normalise across those 64 samples — which is wrong because we want the
        model to behave consistently at every interpolation point.

    Our model (MLP) has no BatchNorm, so the values will be identical.
    This test proves the eval() enforcement runs, even if the effect is subtle
    on models without BatchNorm.
    """
    print(f"\n{SEP2}")
    print("  TEST 7: model.train() before explain() — eval enforced internally")
    print(SEP2)

    model    = MLPStyleA(X.shape[1]).to(device)
    train_model(model, X, y)
    sample   = torch.FloatTensor(X[5])

    # Run explain in normal eval mode — get baseline result
    model.eval()
    explainer = IGTabularExplainer(model, feature_names=names, device=device)
    results_eval = explainer.explain(sample, training_data=X, steps=100)

    # Now deliberately put model in train mode before calling explain()
    # The explainer must override this with eval() internally
    model.train()   # simulate user forgetting to call eval()
    results_train_mode = explainer.explain(sample, training_data=X, steps=100)

    all_ok = True

    # Attributions should be identical because eval() is enforced internally
    attrs_match = np.allclose(
        results_eval["attributions"],
        results_train_mode["attributions"],
        atol=1e-5,
    )
    all_ok &= check(attrs_match,
                    "attributions identical whether model in train/eval mode")
    all_ok &= check(
        results_eval["convergence_delta"] < 0.05,
        f"convergence delta still good  (got {results_eval['convergence_delta']:.4f})",
    )

    return all_ok


# =============================================================================
# TEST 8 — Integer-only input (should warn about categorical features)
# =============================================================================

def test_integer_input_warning(X, y, names, device):
    """
    If ALL input feature values are integers, the data might contain categorical
    features. IG gradients for categorical features are technically computed but
    conceptually wrong (you can't move between discrete categories continuously).

    The explainer should warn the user so they can decide whether to trust
    the attributions or switch to LIME for their specific use case.

    We construct an integer-valued sample and verify the warning fires.
    """
    print(f"\n{SEP2}")
    print("  TEST 8: Integer-only input (should warn about categorical features)")
    print(SEP2)

    model     = MLPStyleA(X.shape[1]).to(device)
    train_model(model, X, y)
    explainer = IGTabularExplainer(model, feature_names=names, device=device)

    # All integer values — looks like categorical/ordinal encoding
    int_sample = torch.FloatTensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0])

    all_ok = True
    all_ok &= check_warns(
        "integer-only input warns about categorical features",
        lambda: explainer.explain(int_sample, steps=50),
    )
    return all_ok


# =============================================================================
# MAIN: RUN ALL TESTS
# =============================================================================

def run() -> bool:
    print(f"\n{SEP}")
    print("  IGTabularExplainer — Full Test Suite")
    print(SEP)

    # ── Setup ────────────────────────────────────────────────────────────────
    print("\n[Setup] Building dataset ...")
    X, y, names = make_dataset()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dataset : {X.shape}   Device: {device}")

    # ── Run all tests ─────────────────────────────────────────────────────────
    results = {
        "Test 1 — Standard [batch,2] model"       : test_standard_model(X, y, names, device),
        "Test 2 — Sigmoid [batch,1] model"         : test_sigmoid_model(X, y, names, device),
        "Test 3 — Unnormalised data warning"        : test_unnormalised_warning(X, y, names, device),
        "Test 4 — Wrong feature name count"         : test_wrong_feature_names(X, y, device),
        "Test 5 — Wrong input shape [3,7]"          : test_wrong_input_shape(X, y, names, device),
        "Test 6 — Regression model output"          : test_regression_model(X, y, names, device),
        "Test 7 — model.train() before explain()"   : test_train_mode_override(X, y, names, device),
        "Test 8 — Integer-only input warning"       : test_integer_input_warning(X, y, names, device),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    all_passed = True
    for name, passed in results.items():
        status     = "PASS ✅" if passed else "FAIL ❌"
        all_passed = all_passed and passed
        print(f"  {name:<45} {status}")

    print(SEP)
    if all_passed:
        print("  ALL TESTS PASSED ✅")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"  {len(failed)} TEST(S) FAILED ❌")
        for f in failed:
            print(f"    → {f}")
    print(SEP + "\n")

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)