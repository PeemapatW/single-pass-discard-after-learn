from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as LA
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

_SQRT_2PI = np.sqrt(2 * np.pi)


class HyperellipsoidBaseClassifier(BaseEstimator, ABC):
    """Root base class for all hyperellipsoid classifiers.

    Inherits sklearn's BaseEstimator (provides get_params/set_params).
    Subclasses must define self.epsilon in __init__ and implement
    check_neuron_class_exist() and set_classes().
    """

    # Eigensolver for compute_sorted_eigencomponent. Default 'eigh' (symmetric
    # solver — the correct/robust choice; see method docstring). Set to 'eig'
    # to reproduce the original paper code (deprecated/spdal.py) bit-for-bit —
    # used only by the refactor-faithfulness regression tests. Not a tuned
    # hyperparameter; not exposed in __init__/get_params.
    eig_solver = 'eigh'

    def compute_sorted_eigencomponent(self, cov : np.ndarray):
        """
        Computes the sorted eigenvectors and eigenvalues of a covariance matrix.

        Args:
            cov: A numpy array representing the covariance matrix.

        Returns:
            A tuple containing:
                - eig_c: A numpy array of sorted eigenvectors (principal components),
                  stored one eigenvector PER ROW — see the convention note below.
                - pca_var: A numpy array of sorted eigenvalues (variances).

        Eigenvector storage convention — ROWS, not columns
        --------------------------------------------------
        The solvers return eigenvectors as COLUMNS; this method transposes so that
        ``eig_c[i]`` is the i-th eigenvector (descending eigenvalue). 

        Covariance matrices are symmetric positive semi-definite by construction
        (pooled second moment / parallel-axis merge / np.cov), so the default
        solver is the symmetric ``eigh``: real eigenvalues, orthonormal
        eigenvectors, ascending order, always converges. The general ``LA.eig``
        returns complex dtype, non-orthonormal eigenvectors and can fail to
        converge on ill-conditioned input; it is
        kept behind ``eig_solver='eig'`` only to reproduce the paper code.
        """
        if self.eig_solver == 'eig':
            # Legacy path — bit-identical to deprecated/spdal.py (paper code).
            eig_value, eig_vector = LA.eig(cov)
            sort_idx = eig_value.argsort()[::-1]
            pca_var = eig_value[sort_idx]          # fancy index -> copy, safe to clamp
            eig_c = eig_vector[:, sort_idx].T
        else:
            # Default: symmetric solver. Symmetrize first to clean any
            # floating-point asymmetry (~1e-15) and make the result independent
            # of which triangle eigh reads.
            cov = (cov + cov.T) / 2.0
            eig_value, eig_vector = LA.eigh(cov)
            # eigh returns eigenvalues ascending; reverse to descending.
            pca_var = eig_value[::-1].copy()       # copy: the clamp below mutates it
            eig_c = eig_vector[:, ::-1].T

        # Find the first non-positive eigenvalue
        first_zero_index =  np.searchsorted(-pca_var,0)

        # Replace non-positive eigenvalues with the lower bound
        lowest_lambda = min(pca_var[first_zero_index-1]/2,self.epsilon)
        pca_var[first_zero_index:] = lowest_lambda

        return eig_c.real, pca_var.real

    def _merge_covariance(self, n_a, cov_a, cen_a, n_b, cov_b, cen_b):
        """Merges covariances of two neuron populations using the parallel-axis theorem.

        Combines n_a samples (cov_a, cen_a) with n_b samples (cov_b, cen_b)
        into a single pooled covariance for the merged neuron.
        """
        n_c = n_a + n_b
        return (1 / n_c) * (
            n_a * cov_a
            + n_b * cov_b
            + (n_a * n_b) / n_c * np.outer(cen_a - cen_b, cen_a - cen_b)
        )

    @abstractmethod
    def check_neuron_class_exist(self, y):
        """Returns True if a neuron with class label 'y' exists in neuron_list."""

    @abstractmethod
    def set_classes(self):
        """Sets self.classes_ from neuron_list."""


class ListNeuronMixin:
    """Mixin that overrides neuron_list base methods for list-of-dicts storage.

    Must appear first in the MRO (before VersatileEllipticBaseClassifier or
    ScalableHyperelipsoidBaseClassifier) so its implementations take priority.
    """

    def check_neuron_class_exist(self, y):
        """Returns True if at least one neuron with class label y exists."""
        return any(n['y'] == y for n in self.neuron_list)

    def set_classes(self):
        """Sets self.classes_ to a sorted array of all unique class labels in neuron_list."""
        self.classes_ = np.unique([n['y'] for n in self.neuron_list])


class VersatileEllipticBaseClassifier(HyperellipsoidBaseClassifier):
    """Base class for classifiers that use hyperellipsoid neurons with pairwise-distance width init.

    Provides: width initialisation, the hyperellipsoidal decision function,
    neuron creation, and neuron merging logic. Used by LRHE, VEBF, SCIL, D4, TRACED.
    """

    def width_init(self, X, y):
        """
        Initializes width for new classes using per-class average pairwise distance.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]
            self.init_width[y_] = self.average_pairwise_distance(Xy)

    def average_pairwise_distance(self, X):
        """Returns a d-dimensional width vector whose value is delta * (mean pairwise distance).

        Used as the initial width when a class is first seen.
        The same scalar is broadcast to all d dimensions.
        """
        n, d = X.shape
        return np.array([self.delta/(n**2)*sum(sp.spatial.distance.pdist(X))*2]*d)

    def hyperellipsoidal_fn(self, x, center, eig_c, width):
        """Evaluates the hyperellipsoidal membership function for a single point x.

        Returns (||P(x-center) / (width+epsilon)||² - 1), where P is the eigenvector
        projection matrix. A value <= 0 means x lies inside the ellipsoid.
        """
        x_centered = x - center
        Margin = width + self.epsilon
        P_d_x = x_centered @ eig_c.T
        return (LA.norm(P_d_x/Margin,ord=2))**2-1

    def create_new_neuron(self, x_i, y_i):
        """Creates a new neuron centred at x_i with identity covariance/eigenvectors and n=1.

        Initial width is taken from self.init_width[y_i] (set during width_init).
        """
        cen = x_i
        cov = np.identity(len(x_i))
        width = self.init_width[y_i]
        n = 1
        eig_c = np.identity(len(x_i))
        neuron = {'y':y_i,'cov':cov,'center':cen,'eig_component':eig_c,'width':width,'n':n}
        return neuron

    def merge_neuron(self, alpha, y):
        """Attempts to merge neuron alpha with any overlapping neuron of the same class y.

        For each other neuron beta of class y, checks if either center lies inside the
        other's ellipsoid (psi <= theta). If so, merges the two into beta using the pooled
        covariance, removes alpha, and stops. SCIL overrides this method with a different
        width formula.
        """
        neurons_y_idx = [i for i, n in enumerate(self.neuron_list) if n['y'] == y]
        if len(neurons_y_idx) > 1:
            n_alpha = self.neuron_list[alpha]['n']
            cov_alpha = self.neuron_list[alpha]['cov']
            cen_alpha = self.neuron_list[alpha]['center']
            width_alpha = self.neuron_list[alpha]['width']
            eig_c_alpha = self.neuron_list[alpha]['eig_component']
            for beta in neurons_y_idx:
                if alpha != beta:
                    cov_beta = self.neuron_list[beta]['cov']
                    cen_beta = self.neuron_list[beta]['center']
                    n_beta = self.neuron_list[beta]['n']
                    width_beta = self.neuron_list[beta]['width']
                    eig_c_beta = self.neuron_list[beta]['eig_component']
                    psi_alpha = self.hyperellipsoidal_fn(cen_alpha, cen_beta, eig_c_beta, width_beta)
                    psi_beta = self.hyperellipsoidal_fn(cen_beta, cen_alpha, eig_c_alpha, width_alpha)
                    if psi_alpha <= self.theta or psi_beta <= self.theta:
                        n_gamma = n_alpha + n_beta
                        cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                        cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)
                        eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)
                        width_gamma = np.array([_SQRT_2PI*np.sqrt(np.abs(pca_var_gamma[d])) for d in range(len(pca_var_gamma))])
                        self.neuron_list[beta]['n'] = n_gamma
                        self.neuron_list[beta]['center'] = cen_gamma
                        self.neuron_list[beta]['cov'] = cov_gamma
                        self.neuron_list[beta]['eig_component'] = eig_c_gamma
                        self.neuron_list[beta]['width'] = width_gamma
                        self.neuron_list.pop(alpha)
                        break


class PrincipleProjectionBaseClassifier(BaseEstimator):
    """Base class for classifiers that resolve overlapping-class regions via eigenvector projection.

    Used by D4 and TRACED. When two closest neurons belong to different classes,
    prediction is broken by projecting each point onto paired principal axes and
    comparing distances in that reduced subspace.
    """

    def angle_between_unit_vectors(self, v1, v2):
        """Computes pairwise acute angles (degrees) between rows of two unit-vector arrays.

        Returns an (m x n) matrix where entry [i,j] is the acute angle between v1[i] and v2[j].
        """
        dot_product = np.einsum('ij,kj->ik', v1, v2)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        return np.minimum(angle, np.pi - angle) * 180 / np.pi

    def find_index_pairs(self, arr):
        """Selects (rows - reduce_dims) non-conflicting (row, col) pairs from an angle matrix.

        Greedy: picks pairs with the smallest angle first, ensuring each row and column
        is used at most once. Fills remaining slots with leftover indices if needed.
        TRACED overrides this to return the transposed shape (2, k) instead of (k, 2).
        """
        rows, cols = arr.shape
        n_pairs = max(1, rows - self.reduce_dims)

        # Create a list of (value, row, col) tuples for elements below threshold
        flat_list = [(val, i // cols, i % cols) for i, val in enumerate(arr.ravel()) if val <= self.threshold]

        flat_list.sort()

        used_rows = set()
        used_cols = set()
        result = []

        for val, row, col in flat_list:
            if row not in used_rows and col not in used_cols:
                result.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
                if len(result) == n_pairs:
                    break
        if len(result) < n_pairs:
            all_rows = set(range(rows))
            all_cols = set(range(cols))
            remaining_rows = sorted(all_rows - used_rows, reverse=True)
            remaining_cols = sorted(all_cols - used_cols, reverse=True)
            remaining_pairs = list(zip(remaining_rows, remaining_cols))
            result.extend(remaining_pairs[:n_pairs - len(result)])

        return np.array(result)

    def calculate_proj_dist(self, x_centered, P, M, norm):
        """
            Calculates the projected distance between a data point and a neuron

        Args:
            x_centered: A numpy array of shape (n_samples, n_features) representing the centered data points.
            P: A numpy array representing the projection matrix.
            M: A numpy array representing the margin or width of hyperellipsoids.
            norm: The order of the norm to use for distance calculation.

        Returns:
        A numpy array of distances between the data points and the hyperellipsoids.
        """

        P_d_x = np.tensordot(x_centered, P, axes=(1, 1))  # Project data points
        return (LA.norm(P_d_x / M, ord=norm, axis=1)) - 1  # D4 Eq. 7

    def predict_with_eigen_proj(self, neuron_list_test, X, argsort_dist):
        """Predicts class labels using eigenprojection for cross-class neuron pairs.

        For each point:
        - If the two closest neurons share the same class, assign that class directly.
        - Otherwise, pair up their principal axes by smallest angle, project the point,
          and assign the class of whichever neuron's projected distance is smaller.
        Wraps the inner logic of predict() for D4 and TRACED.

        Args:
            neuron_list_test: A list of neuron dicts.
            X: A numpy array of shape (n_samples, n_features) representing the data.
            argsort_dist: A numpy array of shape (n_samples, 2) containing the indices of the two
                            closest neurons for each data point.

        Returns:
            A numpy array of predicted class labels.
        """

        first_index, second_index = argsort_dist[:, 0], argsort_dist[:, 1]
        top_two_index = np.array([sorted(idx) for idx in zip(first_index, second_index)])
        unique_top_two = list(set(map(tuple, top_two_index)))
        y_pred = np.full(len(X), np.nan)

        for unique_pair in unique_top_two:
            mask = (top_two_index == unique_pair).all(axis=1)
            if neuron_list_test[unique_pair[0]]['y'] == neuron_list_test[unique_pair[1]]['y']:
                y_pred[mask] = neuron_list_test[unique_pair[0]]['y']
            else:
                x_unique_pair = X[mask]

                centers, eig_cs, widths = zip(*[self.get_neuron_data(neuron_list_test, idx) for idx in unique_pair])

                eig_angle = self.angle_between_unit_vectors(eig_cs[0], eig_cs[1])
                top_index = self.find_index_pairs(eig_angle)

                Ps = [eig_c[top_index[:, i]] for i, eig_c in enumerate(eig_cs)]
                ws = [width[top_index[:, i]] for i, width in enumerate(widths)]
                Ms = [w + self.epsilon for w in ws]

                x_centereds = [x_unique_pair - center for center in centers]

                x_proj_dist = {
                    y: self.calculate_proj_dist(x_centered, P, M, self.norm)
                    for y, x_centered, P, M in zip(unique_pair, x_centereds, Ps, Ms)
                }

                dist_matrix = np.column_stack([x_proj_dist[unique_pair[0]], x_proj_dist[unique_pair[1]]])
                winner_col = np.argmin(dist_matrix, axis=1)
                winner_idx = np.array(unique_pair)[winner_col]
                y_pred[mask] = np.array([neuron_list_test[i]['y'] for i in winner_idx])

        return y_pred


class ScalableHyperelipsoidBaseClassifier(HyperellipsoidBaseClassifier):
    """Base class for classifiers that use nearest-neighbour distance thresholds.

    Instead of pairwise distances, uses median/mean NN distance per class
    as the creation threshold. Used by SHEF and TRACED.
    """

    def distance_init(self, X, y):
        """
        Initializes distance thresholds for new classes using median nearest-neighbor distance.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.dist_ths.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]
            self.dist_ths[y_] = self.find_median_dist_to_neighbor(Xy)

    def find_median_dist_to_neighbor(self, X):
        """
        Calculates the median distance between each point in X and its nearest neighbor.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.

        Returns:
            The median distance to the nearest neighbor.
        """

        if len(X) == 1:
            return np.sqrt(np.e)
        else:
            neigh1 = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X).kneighbors()[0]
            median_dist = np.median(neigh1)
            if median_dist > self.epsilon:
                return median_dist
            else:
                return self.epsilon

    def find_mean_dist_to_neighbor(self, X):
        """
        Calculates the mean distance between each point in X and its nearest neighbor.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.

        Returns:
            The mean distance to the nearest neighbor.
        """

        if len(X) == 1:
            return np.sqrt(np.e)
        else:
            neigh1 = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X).kneighbors()[0]
            return np.mean(neigh1)

    @abstractmethod
    def dist_ths_y_update(self, y):
        """Updates the distance threshold for class label 'y'."""
