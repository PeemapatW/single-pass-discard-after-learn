import numpy as np
import numpy.linalg as LA

from spdal._base import VersatileEllipticBaseClassifier, ListNeuronMixin


class LRHE(ListNeuronMixin, VersatileEllipticBaseClassifier):
    """Learning Rule for Hyperellipsoid Expansion (LRHE).

    Grows hyperellipsoid neurons one sample at a time. For each incoming
    sample of class y, finds the nearest same-class neuron and checks whether
    a temporary updated center still covers the sample. If yes, updates the
    neuron; otherwise creates a new one. Competing neurons from other classes
    that cover the sample are shifted and shrunk away. After each update or
    creation, overlapping same-class neurons are merged.

    Parameters
    ----------
    alpha : float
        Minimum-width factor used in shift_and_shrink_neuron (width floor = alpha * current_width).
        Paper default 0.99 ("gradual shrinking", LRHE paper p.11; alpha=1 disables
        shrinking, smaller alpha = more aggressive). Default changed 0.5 -> 0.99 on
        2026-06-11 to match the paper; benchmarks tune alpha per dataset, so bench
        results are unaffected by the default.
    theta : float
        Overlap threshold for merge_neuron (merge when psi <= theta).
    delta : float
        Scaling factor for initial pairwise-distance width.
    epsilon : float
        Numerical floor added to widths and eigenvalues.
    """

    def __init__(self, alpha=0.99, theta=0, delta=1, epsilon=1e-10):
        self.neuron_list = []
        self.init_width = {}
        self.delta = delta
        self.alpha = alpha
        self.theta = theta
        self.epsilon = epsilon

    def fit(self, X, y, classes=None, _reset=True):
        """Train on X, y. Resets all state first unless _reset=False (used by partial_fit)."""
        if _reset:
            self.neuron_list = []
            self.init_width = {}
        self.width_init(X, y)
        for x_i, y_i in zip(X, y):
            if self.check_neuron_class_exist(y_i):
                neurons_y = [(i, n) for i, n in enumerate(self.neuron_list) if n['y'] == y_i]
                xi, neuron_xi = min(neurons_y, key=lambda t: LA.norm(x_i - t[1]['center']))
                cov_xi = neuron_xi['cov']
                cen_xi = neuron_xi['center']
                n_xi = neuron_xi['n']
                width_xi = neuron_xi['width']
                eig_c_xi = neuron_xi['eig_component']
                psi_xi = self.hyperellipsoidal_fn(x_i, cen_xi, eig_c_xi, width_xi)

                self.shift_and_shrink_neuron(x_i, y_i, psi_xi)

                cen_temp = (n_xi * cen_xi + x_i) / (n_xi + 1)
                psi_temp = self.hyperellipsoidal_fn(x_i, cen_temp, eig_c_xi, width_xi)

                if psi_temp > 0:
                    # Create new neuron
                    self.neuron_list.append(self.create_new_neuron(x_i, y_i))
                    alpha = len(self.neuron_list) - 1
                else:
                    # Update
                    n_temp = n_xi + 1
                    cov_temp = (n_xi*cov_xi + np.matmul(np.array([x_i]).T, [x_i]) - np.matmul(np.array([cen_xi]).T, [cen_xi])) / (n_xi+1) - np.matmul(np.array([cen_temp]).T, [cen_temp]) + np.matmul(np.array([cen_xi]).T, [cen_xi])
                    eig_c_temp, _ = self.compute_sorted_eigencomponent(cov_temp)
                    width_temp = np.array([width_xi[d] + np.abs(np.matmul(cen_temp - cen_xi, eig_c_temp[d].T)) for d in range(len(width_xi))])
                    self.neuron_list[xi]['cov'] = cov_temp
                    self.neuron_list[xi]['width'] = width_temp
                    self.neuron_list[xi]['center'] = cen_temp
                    self.neuron_list[xi]['n'] = n_temp
                    self.neuron_list[xi]['eig_component'] = eig_c_temp
                    alpha = xi

                self.merge_neuron(alpha, y_i)

            else:
                self.neuron_list.append(self.create_new_neuron(x_i, y_i))

        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        """Incrementally train on X, y — preserves existing neurons."""
        self.fit(X, y, _reset=False)

    def predict(self, X):
        """Predicts class labels by finding the nearest neuron (min hyperellipsoidal distance)."""
        dist = np.empty((len(X), len(self.neuron_list)))
        for idx, neuron in enumerate(self.neuron_list):
            center = neuron['center']
            eig_c = neuron['eig_component'].real
            width = np.array(neuron['width']).reshape(len(center))
            x_centered = X - center
            P_d_x = np.tensordot(x_centered, eig_c, axes=(1, 1))
            dist[:, idx] = LA.norm(P_d_x / width, ord=2, axis=1) ** 2 - 1
        y_pred = np.array([n['y'] for n in self.neuron_list])[np.argmin(dist, axis=1)]
        return y_pred

    def shift_and_shrink_neuron(self, x_i, y_i, psi_xi):
        """Shrinks and shifts competing neurons (different class) that cover x_i.

        For each neuron of a different class that contains x_i (psi_idx <= 0) and
        overlaps more than the nearest same-class neuron (psi_idx <= psi_xi), updates
        its width (shrink, floored at alpha * current_width) and shifts its center away.
        """
        for idx, neuron in enumerate(self.neuron_list):
            if neuron['y'] != y_i:
                cen_idx = neuron['center']
                n_idx = neuron['n']
                width_idx = neuron['width']
                eig_c_idx = neuron['eig_component']
                psi_idx = self.hyperellipsoidal_fn(x_i, cen_idx, eig_c_idx, width_idx)
                if psi_idx <= 0 and psi_idx <= psi_xi:
                    new_width = np.array([max([(n_idx*width_idx[d] + np.matmul(x_i - cen_idx, eig_c_idx[d].T)) / (n_idx+1), self.alpha*width_idx[d]]) for d in range(len(width_idx))])
                    new_cen = cen_idx - 1/n_idx * (x_i - cen_idx)
                    self.neuron_list[idx]['width'] = new_width
                    self.neuron_list[idx]['center'] = new_cen
