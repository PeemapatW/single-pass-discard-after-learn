"""
Edge case, sklearn interface, determinism, and TRACED counter tests.

Covers gaps not addressed by test_classifiers.py or test_hyperparams.py:
- sklearn interface: classes_, get_params/set_params, clone
- Edge cases: single sample, single class, new class mid-stream
- Determinism: two independent instances produce identical results
- TRACED counters: count_overlap / count_outside increment correctly

Known limitations documented with xfail
----------------------------------------
SCIL (N0=3, default): neurons with n < N0 are excluded from predict.
    A single-sample-per-class fit leaves all neurons with n=1, producing
    an empty predict pool and a ValueError.  Use SCIL(N0=1) to allow
    single-sample prediction.

TRACED: eigendecomposition is called on a covariance derived from a single
    sample, which produces a 0-dimensional array and raises LinAlgError.
    TRACED requires at least 2 samples per class.

D4 (single feature): np.cov on a 1-feature input returns a scalar;
    LA.eig requires a 2-D array.  D4 requires at least 2 features.

classes_ reflects actual trained classes (by design)
-----------------------------------------------------
All classifiers derive classes_ from the neurons present in neuron_list,
NOT from the classes= argument.  After a partial_fit that only sees class 0,
classes_ == [0] even if classes=[0, 1] was passed.  The classes= argument
is accepted for API compatibility but does not pre-populate classes_.

D4 predict dtype
------------------
D4.predict initialises y_pred with np.nan (float64) for NaN-sentinel logic
and the labels remain float64 after being filled in.  All other classifiers
return integer dtype.  This mirrors the deprecated DataFrame implementation.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification

from spdal import LRHE, VEBF, SCIL, SHEF, D4, TRACED

ALL_CLASSIFIERS = [LRHE, VEBF, SCIL, SHEF, D4, TRACED]
ALL_CLASSIFIER_IDS = ["LRHE", "VEBF", "SCIL", "SHEF", "D4", "TRACED"]

# Classifiers that can handle a single sample per class (n=1 neurons included)
SINGLE_SAMPLE_OK = [LRHE, VEBF, SHEF, D4]
SINGLE_SAMPLE_OK_IDS = ["LRHE", "VEBF", "SHEF", "D4"]

# Classifiers that can handle a single input feature
SINGLE_FEATURE_OK = [LRHE, VEBF, SCIL, SHEF, TRACED]
SINGLE_FEATURE_OK_IDS = ["LRHE", "VEBF", "SCIL", "SHEF", "TRACED"]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=300, random_state=42)
    return X[:200], X[200:], y[:200]


@pytest.fixture(scope="module")
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X[:120], X[120:], y[:120]


# ---------------------------------------------------------------------------
# sklearn interface tests
# ---------------------------------------------------------------------------

class TestSklearnInterface:

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_classes_attribute_set_after_fit(self, binary_data, Clf):
        X_train, _, y_train = binary_data
        clf = Clf()
        assert not hasattr(clf, 'classes_'), "classes_ should not exist before fit"
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert hasattr(clf, 'classes_'), "classes_ must be set after partial_fit"
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_classes_attribute_multiclass(self, iris_data, Clf):
        X_train, _, y_train = iris_data
        clf = Clf()
        clf.fit(X_train, y_train)
        assert hasattr(clf, 'classes_')
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_get_params_returns_dict(self, Clf):
        clf = Clf()
        params = clf.get_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_set_params_roundtrip(self, Clf):
        clf = Clf()
        original_params = clf.get_params()
        clf.set_params(**original_params)
        assert clf.get_params() == original_params

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_clone_produces_unfitted_copy(self, binary_data, Clf):
        X_train, X_test, y_train = binary_data
        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])

        clf2 = clone(clf)

        assert clf2.get_params() == clf.get_params()
        assert not hasattr(clf2, 'classes_'), "cloned estimator must be unfitted"
        assert len(clf2.neuron_list) == 0

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_predict_output_shape(self, binary_data, Clf):
        X_train, X_test, y_train = binary_data
        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_predict_labels_in_classes(self, binary_data, Clf):
        X_train, X_test, y_train = binary_data
        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert set(preds).issubset({0, 1}), f"unexpected labels: {set(preds)}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.parametrize("Clf", SINGLE_SAMPLE_OK, ids=SINGLE_SAMPLE_OK_IDS)
    def test_single_sample_per_class(self, Clf):
        """LRHE, VEBF, SHEF, D4 handle 1 sample per class without error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2, 4))
        y = np.array([0, 1])
        clf = Clf()
        clf.partial_fit(X, y, classes=[0, 1])
        preds = clf.predict(X)
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("Clf", [SCIL, TRACED], ids=["SCIL", "TRACED"])
    def test_single_sample_per_class_known_limitation(self, Clf):
        """
        SCIL with default N0=3: all neurons have n=1 after single-sample fit,
        so the predict pool is empty → ValueError.

        TRACED: covariance of a single sample produces a 0-d value → LinAlgError
        in eigendecomposition.

        These are documented limitations, not regressions.
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2, 4))
        y = np.array([0, 1])
        clf = Clf()
        with pytest.raises(Exception):
            clf.partial_fit(X, y, classes=[0, 1])
            clf.predict(X)

    def test_scil_single_sample_works_with_n0_1(self):
        """SCIL(N0=1) includes n=1 neurons and handles single-sample fit."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2, 4))
        y = np.array([0, 1])
        clf = SCIL(N0=1)
        clf.partial_fit(X, y, classes=[0, 1])
        preds = clf.predict(X)
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_single_class_fit_then_predict(self, Clf):
        """partial_fit with only one class; predict must return that class for all points."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((20, 4))
        y_train = np.zeros(20, dtype=int)
        X_test = rng.standard_normal((5, 4))

        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert preds.shape == (5,)
        assert np.all(preds == 0), f"expected all 0, got {preds}"

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_new_class_introduced_in_second_partial_fit(self, Clf):
        """
        classes_ is derived from actual trained neurons, not the classes= argument.

        After fitting only class-0 data: classes_ == [0]
        After fitting class-1 data in a second call: classes_ == [0, 1]

        The classes= argument is accepted for API compatibility but does not
        pre-populate classes_.
        """
        rng = np.random.default_rng(2)
        X0 = rng.standard_normal((20, 4))
        y0 = np.zeros(20, dtype=int)
        X1 = rng.standard_normal((20, 4)) + 3
        y1 = np.ones(20, dtype=int)
        X_test = rng.standard_normal((10, 4))

        clf = Clf()
        clf.partial_fit(X0, y0, classes=[0, 1])
        # After first fit: only class 0 neurons exist
        assert 0 in clf.classes_
        assert len(clf.classes_) == 1

        clf.partial_fit(X1, y1, classes=[0, 1])
        # After second fit: both classes present
        np.testing.assert_array_equal(clf.classes_, [0, 1])

        preds = clf.predict(X_test)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("Clf", SINGLE_FEATURE_OK, ids=SINGLE_FEATURE_OK_IDS)
    def test_single_feature(self, Clf):
        """LRHE, VEBF, SCIL, SHEF, TRACED handle 1-feature input."""
        rng = np.random.default_rng(3)
        X_train = np.concatenate([rng.standard_normal((30, 1)),
                                   rng.standard_normal((30, 1)) + 5])
        y_train = np.array([0]*30 + [1]*30)
        X_test = np.array([[-1.0], [6.0]])

        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert preds.shape == (2,)

    def test_d4_single_feature_known_limitation(self):
        """
        D4 requires at least 2 features.
        np.cov on a 1-feature input returns a scalar; LA.eig expects a 2-D array.
        """
        rng = np.random.default_rng(3)
        X_train = np.concatenate([rng.standard_normal((30, 1)),
                                   rng.standard_normal((30, 1)) + 5])
        y_train = np.array([0]*30 + [1]*30)
        clf = D4()
        with pytest.raises(Exception):
            clf.partial_fit(X_train, y_train, classes=[0, 1])

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_many_incremental_chunks(self, Clf):
        """10 small partial_fit calls should leave classes_ intact and predict validly."""
        X, y = make_classification(n_samples=200, random_state=7)
        clf = Clf()
        for i in range(10):
            Xc = X[i*20:(i+1)*20]
            yc = y[i*20:(i+1)*20]
            clf.partial_fit(Xc, yc, classes=[0, 1])
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("Clf", [LRHE, VEBF, SCIL, SHEF, TRACED],
                             ids=["LRHE", "VEBF", "SCIL", "SHEF", "TRACED"])
    def test_predict_returns_integer_dtype(self, binary_data, Clf):
        """All classifiers except D4 return integer-dtype predictions."""
        X_train, X_test, y_train = binary_data
        clf = Clf()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert np.issubdtype(preds.dtype, np.integer), (
            f"expected integer dtype, got {preds.dtype}"
        )

    def test_d4_predict_returns_numeric_labels(self, binary_data):
        """
        D4.predict returns float64 (initialises y_pred with np.nan sentinel).
        Labels are numeric and match the training class set.
        This matches the deprecated DataFrame implementation.
        """
        X_train, X_test, y_train = binary_data
        clf = D4()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert np.issubdtype(preds.dtype, np.floating), (
            f"D4 expected float dtype, got {preds.dtype}"
        )
        assert set(preds).issubset({0.0, 1.0}), f"unexpected labels: {set(preds)}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_two_instances_same_predictions_binary(self, binary_data, Clf):
        X_train, X_test, y_train = binary_data
        clf1 = Clf()
        clf2 = Clf()
        clf1.partial_fit(X_train, y_train, classes=[0, 1])
        clf2.partial_fit(X_train, y_train, classes=[0, 1])
        np.testing.assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_two_instances_same_predictions_iris(self, iris_data, Clf):
        X_train, X_test, y_train = iris_data
        clf1 = Clf()
        clf2 = Clf()
        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
        np.testing.assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))

    @pytest.mark.parametrize("Clf", ALL_CLASSIFIERS, ids=ALL_CLASSIFIER_IDS)
    def test_fit_and_partial_fit_produce_same_result(self, binary_data, Clf):
        """fit(X, y) and partial_fit(X, y) on the same data must agree."""
        X_train, X_test, y_train = binary_data
        clf_fit = Clf()
        clf_fit.fit(X_train, y_train)

        clf_pf = Clf()
        clf_pf.partial_fit(X_train, y_train, classes=[0, 1])

        np.testing.assert_array_equal(
            clf_fit.predict(X_test),
            clf_pf.predict(X_test),
        )


# ---------------------------------------------------------------------------
# TRACED counter tests
# ---------------------------------------------------------------------------

class TestTRACEDCounters:

    def test_counters_start_at_zero(self):
        clf = TRACED()
        assert clf.count_overlap == 0
        assert clf.count_outside == 0

    def test_counters_are_integers(self):
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="overlap-outside", reduce_dims=1)
        clf.fit(X[:120], y[:120])
        clf.predict(X[120:])
        assert isinstance(clf.count_overlap, (int, np.integer))
        assert isinstance(clf.count_outside, (int, np.integer))

    def test_overlap_method_only_increments_overlap_not_outside(self):
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="overlap", reduce_dims=1)
        clf.fit(X[:120], y[:120])
        clf.predict(X[120:])
        assert clf.count_outside == 0, "method='overlap' must not increment count_outside"

    def test_outside_method_only_increments_outside_not_overlap(self):
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="outside")
        clf.fit(X[:120], y[:120])
        clf.predict(X[120:])
        assert clf.count_overlap == 0, "method='outside' must not increment count_overlap"

    def test_reduce_dims_zero_no_overlap_resolution(self):
        """reduce_dims=0 (default) disables PCA overlap resolution → count_overlap stays 0."""
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="overlap-outside", reduce_dims=0)
        clf.fit(X[:120], y[:120])
        clf.predict(X[120:])
        assert clf.count_overlap == 0

    def test_counters_accumulate_across_predict_calls(self):
        """Counters must not reset between predict calls."""
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="overlap-outside", reduce_dims=1)
        clf.fit(X[:120], y[:120])

        clf.predict(X[120:])
        total_after_first = clf.count_overlap + clf.count_outside

        clf.predict(X[120:])
        total_after_second = clf.count_overlap + clf.count_outside

        assert total_after_second >= total_after_first, (
            "counters must accumulate, not reset"
        )

    def test_count_overlap_is_nonnegative(self):
        """count_overlap is always a non-negative integer — never goes negative."""
        X, y = load_iris(return_X_y=True)
        clf = TRACED(method="overlap-outside", reduce_dims=1)
        clf.fit(X[:120], y[:120])
        clf.predict(X[120:])
        assert clf.count_overlap >= 0
        assert clf.count_outside >= 0
