from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as LA
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

_SQRT_2PI = np.sqrt(2 * np.pi)


class HyperellipsoidBaseClassifier(BaseEstimator, ABC):
    def compute_sorted_eigencomponent(self, cov : np.ndarray):
        """
        Computes the sorted eigenvectors and eigenvalues of a covariance matrix.

        Args:
            cov: A numpy array representing the covariance matrix.

        Returns:
            A tuple containing:
                - eig_c: A numpy array of sorted eigenvectors (principal components).
                - pca_var: A numpy array of sorted eigenvalues (variances).
        """

        eig_value, eig_vector = LA.eig(cov)
        sort_idx = eig_value.argsort()[::-1]
        pca_var = eig_value[sort_idx]
        eig_c = eig_vector[:,sort_idx].T

        # Find the first non-positive eigenvalue
        first_zero_index =  np.searchsorted(-pca_var,0)

        # Replace non-positive eigenvalues with the lower bound
        lowest_lambda = min(pca_var[first_zero_index-1]/2,self.epsilon)
        pca_var[first_zero_index:] = lowest_lambda

        return eig_c.real, pca_var.real

    def _merge_covariance(self, n_a, cov_a, cen_a, n_b, cov_b, cen_b):
        """Computes the merged covariance of two neuron populations."""
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
    """Mixin that overrides neuron_list base methods for list-of-dicts storage."""

    def check_neuron_class_exist(self, y):
        return any(n['y'] == y for n in self.neuron_list)

    def set_classes(self):
        self.classes_ = np.unique([n['y'] for n in self.neuron_list])


class VersatileEllipticBaseClassifier(HyperellipsoidBaseClassifier):

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
        n, d = X.shape
        return np.array([self.delta/(n**2)*sum(sp.spatial.distance.pdist(X))*2]*d)

    def hyperellipsoidal_fn(self, x, center, eig_c, width):
        x_centered = x - center
        Margin = width + self.epsilon
        P_d_x = x_centered @ eig_c.T
        return (LA.norm(P_d_x/Margin,ord=2))**2-1

    def create_new_neuron(self, x_i, y_i):
        cen = x_i
        cov = np.identity(len(x_i))
        width = self.init_width[y_i]
        n = 1
        eig_c = np.identity(len(x_i))
        neuron = {'y':y_i,'cov':cov,'center':cen,'eig_component':eig_c,'width':width,'n':n}
        return neuron

    def merge_neuron(self, alpha, y):
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
    def angle_between_unit_vectors(self, v1, v2):
        dot_product = np.einsum('ij,kj->ik', v1, v2)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        return np.minimum(angle, np.pi - angle) * 180 / np.pi

    def find_index_pairs(self, arr):
        rows, cols = arr.shape

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
                if len(result) == self.max_d:
                    break
        if len(result) < self.max_d:
            all_rows = set(range(rows))
            all_cols = set(range(cols))
            remaining_rows = sorted(all_rows - used_rows, reverse=True)
            remaining_cols = sorted(all_cols - used_cols, reverse=True)
            remaining_pairs = list(zip(remaining_rows, remaining_cols))
            result.extend(remaining_pairs[:self.max_d - len(result)])

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
        return (LA.norm(P_d_x / M, ord=norm, axis=1)) - self.r  # Calculate distance

    def predict_with_eigen_proj(self, neuron_list_test, X, argsort_dist):
        """
        Predicts class labels for data points using eigenprojection and distance calculations.

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
