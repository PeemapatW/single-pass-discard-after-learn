import numpy as np
import numpy.linalg as LA

from ._base import ScalableHyperelipsoidBaseClassifier


class SHEF(ScalableHyperelipsoidBaseClassifier):
    """Scalable Hyperellipsoidal Ensemble Fusion (SHEF).

    Uses median nearest-neighbour distance as a creation threshold: a new neuron
    is spawned only when x_i is farther than dist_ths[y] from all existing neurons.
    Neurons store only center and covariance (no eigenvectors/widths). Prediction
    uses a projection distance and a two-stage tie-breaking rule:
      1. If both top neurons belong to the same class → assign directly.
      2. If only the closest neuron is inside its ellipsoid → assign to it.
      3. Otherwise compute a discriminant vector w = inv(S1+S2)(c2-c1) and assign
         to the neuron with smaller projection ratio distance.

    Parameters
    ----------
    M : int
        Minimum sample count before a neuron is eligible for merging.
    r : float
        Ellipsoid radius scaling factor used in projection-distance calculations.
    epsilon : float
        Regularisation of Eq. 17 (S + epsilon I), the initial threshold sqrt(epsilon) of a class first seen
        as one sample or after the first chunk, and the initial covariance epsilon I of a new neuron.
        Default 1e-4, the value used in the SHEF paper.
    """

    def __init__(self, M = 3, r = 1.5, epsilon=1e-4):
        self.neuron_list = []
        self.dist_ths = {}
        self.M = M
        self.r = r
        self.epsilon = epsilon

    def check_neuron_class_exist(self, y):
        """Returns True if at least one neuron with class label y exists."""
        return any(n['y'] == y for n in self.neuron_list)

    def set_classes(self):
        """Sets self.classes_ from neuron_list (ListNeuronMixin equivalent for SHEF)."""
        self.classes_ = np.unique([n['y'] for n in self.neuron_list])

    def dist_ths_y_update(self, y):
        """Doubles dist_ths[y] if more than half the neurons for class y have n < M.

        Called after adding a new neuron to adaptively widen the creation threshold
        when the model is still young (many small neurons exist).
        """
        ths = self.dist_ths[y]
        neurons_y = [n for n in self.neuron_list if n['y'] == y]
        m = sum(1 for n in neurons_y if n['n'] < self.M)
        if m > len(neurons_y) / 2:
            self.dist_ths[y] = ths * 2

    def create_new_neuron(self, x_i, y_i):
        """Creates a neuron with only center, cov (epsilon*I), and n=1 — no eig/width fields."""
        cen = x_i
        cov = np.zeros([len(x_i), len(x_i)]) + np.identity(len(x_i)) * self.epsilon
        n = 1
        neuron = {'y': y_i, 'cov': cov, 'center': cen, 'n': n}
        return neuron

    def distance_init(self, X, y):
        """Classes in the first chunk start at the median nearest-neighbour distance of their own data
        (Eq. 25). A class first seen in a later chunk starts at sqrt(epsilon) (p. 6: "the initial width of all
        new classes appearing after the first chunk will be set to sqrt(eps)")."""
        if not self.dist_ths:
            return super().distance_init(X, y)
        for y_ in set(np.unique(y)) - set(self.dist_ths.keys()):
            self.dist_ths[y_] = np.sqrt(self.epsilon)

    def find_median_dist_to_neighbor(self, X):
        """A one-sample class starts at sqrt(epsilon) (Eq. 25, Lemma 1); otherwise the median nearest-
        neighbour distance."""
        if len(X) == 1:
            return np.sqrt(self.epsilon)
        return super().find_median_dist_to_neighbor(X)

    def _reg_inv(self, S):
        """(S + epsilon I)^-1, the regularised inverse of Eq. 17. Works on one matrix or a stack."""
        return LA.inv(S + self.epsilon * np.eye(S.shape[-1]))

    def merge_neuron(self, alpha, y):
        """Merges neuron alpha with its nearest same-class neighbour if their ellipsoids overlap.

        Overlap is determined via eigenvalues of a block matrix P constructed from the two
        covariances scaled by r². A merge occurs when P has complex eigenvalues, repeated
        eigenvalues, or any negative eigenvalue — indicating geometric intersection.
        Only triggers when alpha has grown past M samples.
        """
        neurons_y_idx = [i for i, n in enumerate(self.neuron_list) if n['y'] == y]
        n_alpha = self.neuron_list[alpha]['n']
        cen_alpha = self.neuron_list[alpha]['center']
        cov_alpha = self.neuron_list[alpha]['cov']
        if len(neurons_y_idx) > 1 and n_alpha >= self.M:          # Alg. 1 step 15
            distances = [LA.norm(cen_alpha - self.neuron_list[i]['center']) for i in neurons_y_idx]
            beta = neurons_y_idx[np.argsort(distances)[1]]
            cen_beta = self.neuron_list[beta]['center']
            cov_beta = self.neuron_list[beta]['cov']

            # Precompute some matrices for efficiency
            # Eq. 37: the r-scaled ellipsoid has shape r^2 S, inverse S^-1 / r^2 (Theorem 2 prints S / r^2).
            cov_tilde_inv = self._reg_inv(cov_beta) / (self.r ** 2)
            cov_tilde = cov_alpha * (self.r ** 2)
            F = (-cen_alpha + cen_beta) @ cov_tilde_inv
            D = cov_tilde @ cov_tilde_inv + np.outer(cen_alpha, F)

            # Construct P matrix using more efficient methods
            P_upper = np.hstack([D, -D @ cen_beta[:, np.newaxis] + cen_alpha[:, np.newaxis]])
            P_lower = np.hstack([F, -F @ cen_beta[:, np.newaxis] + 1])
            P = np.vstack([P_upper, P_lower])

            eig_P = LA.eig(P)[0]

            # Check merge conditions
            if all(np.isreal(eig_P)) and len(np.unique(eig_P)) == len(eig_P) and any(np.less(eig_P, 0)):
                pass  # Conditions not met, continue to next neuron
            else:
                n_beta = self.neuron_list[beta]['n']
                n_gamma = n_alpha + n_beta
                cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)

                # Update beta in-place, then remove alpha
                self.neuron_list[beta]['n'] = n_gamma
                self.neuron_list[beta]['center'] = cen_gamma
                self.neuron_list[beta]['cov'] = cov_gamma
                self.neuron_list.pop(alpha)

    def fit(self, X, y, classes=None, _reset=True):
        """Train on X, y. Resets all state first unless _reset=False (used by partial_fit)."""
        if _reset:
            self.neuron_list = []
            self.dist_ths = {}
        self.distance_init(X, y)
        for x_i, y_i in zip(X, y):
            if self.check_neuron_class_exist(y_i):
                neurons_y = [(i, n) for i, n in enumerate(self.neuron_list) if n['y'] == y_i]
                alpha, neuron_alpha = min(neurons_y, key=lambda t: LA.norm(x_i - t[1]['center']))
                if LA.norm(x_i - neuron_alpha['center']) > self.dist_ths[y_i]:
                    # Create new neuron
                    self.neuron_list.append(self.create_new_neuron(x_i, y_i))
                    alpha = len(self.neuron_list) - 1
                    self.dist_ths_y_update(y_i)
                else:
                    n_a = neuron_alpha['n']
                    cen_a = neuron_alpha['center']
                    cov_a = neuron_alpha['cov']
                    new_cen = (n_a * cen_a + x_i) / (n_a + 1)
                    new_cov = n_a / (n_a + 1) * (cov_a + np.outer(cen_a - x_i, cen_a - x_i) / (n_a + 1))
                    self.neuron_list[alpha]['cov'] = new_cov
                    self.neuron_list[alpha]['center'] = new_cen
                    self.neuron_list[alpha]['n'] = n_a + 1
                self.merge_neuron(alpha, y_i)
            else:
                # Create new neuron
                self.neuron_list.append(self.create_new_neuron(x_i, y_i))
                self.dist_ths_y_update(y_i)

        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        """Incrementally train on X, y — preserves existing neurons."""
        self.fit(X, y, _reset=False)

    def _vectorized_discriminant_vector(self, c1, S1, c2, S2):
        """Calculates the discriminant vector between two SHEFs for a batch of neurons."""
        S_sum = S1 + S2
        S_sum_inv = self._reg_inv(S_sum)
        c_diff = c2 - c1
        w = np.einsum('...ij,...j->...i', S_sum_inv, c_diff)
        return w

    def _vectorized_projection_distance(self, x, c, S, w):
        """Calculates the projection ratio distance for a batch of points."""
        x_centered = x - c
        numerator = np.abs(np.einsum('...i,...i->...', x_centered, w))
        wSw = np.einsum('...i,...ij,...j->...', w, S, w)
        denominator = self.r * np.sqrt(wSw)
        denominator[denominator < self.epsilon] = self.epsilon
        return numerator / denominator

    def predict(self, X):
        """Predicts class labels using Mahalanobis projection distance with three-stage tie-breaking."""
        if len(self.neuron_list) == 1:
            return np.array([self.neuron_list[0]['y']] * len(X))

        dist = np.empty((len(X), len(self.neuron_list)))
        for idx, neuron in enumerate(self.neuron_list):
            center = neuron['center']
            cov = neuron['cov']
            cov_inv = self._reg_inv(cov)
            x_centered = X - center
            wp = np.tensordot(x_centered, cov_inv, axes=(1, 1))
            wp_norm = LA.norm(wp, axis=1)[:, None]
            wp_norm[wp_norm < self.epsilon] = self.epsilon
            wp = wp / wp_norm
            r_wSw = self.r * np.sqrt(np.einsum('ij,ij->i', np.tensordot(wp, cov, axes=(1, 1)), wp))
            r_wSw[r_wSw < self.epsilon] = self.epsilon
            dist[:, idx] = np.abs(np.einsum('ij,ij->i', x_centered, wp)) / r_wSw

        sort_idx = np.argsort(dist, axis=1)[:, :2]
        sort_dist = np.take_along_axis(dist, sort_idx, axis=1)

        y_values = np.array([n['y'] for n in self.neuron_list])
        class1 = y_values[sort_idx[:, 0]]
        class2 = y_values[sort_idx[:, 1]]

        y_predict = np.empty(X.shape[0], dtype=class1.dtype)

        mask_same_class = (class1 == class2)
        y_predict[mask_same_class] = class1[mask_same_class]

        mask_clear_win = (sort_dist[:, 0] <= 1) & (sort_dist[:, 1] > 1)
        y_predict[mask_clear_win & ~mask_same_class] = class1[mask_clear_win & ~mask_same_class]

        mask_ambiguous = ~mask_same_class & ~mask_clear_win

        if np.any(mask_ambiguous):
            X_amb = X[mask_ambiguous]
            n1_idx = sort_idx[mask_ambiguous, 0]
            n2_idx = sort_idx[mask_ambiguous, 1]

            centers = np.stack([n['center'] for n in self.neuron_list])
            covs = np.stack([n['cov'] for n in self.neuron_list])

            c1, S1 = centers[n1_idx], covs[n1_idx]
            c2, S2 = centers[n2_idx], covs[n2_idx]

            w = self._vectorized_discriminant_vector(c1, S1, c2, S2)
            d1 = self._vectorized_projection_distance(X_amb, c1, S1, w)
            d2 = self._vectorized_projection_distance(X_amb, c2, S2, w)

            y_predict[mask_ambiguous] = np.where(d1 <= d2, class1[mask_ambiguous], class2[mask_ambiguous])

        return y_predict
