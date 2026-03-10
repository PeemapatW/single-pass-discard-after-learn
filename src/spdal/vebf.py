import numpy as np
import numpy.linalg as LA

from ._base import VersatileEllipticBaseClassifier, ListNeuronMixin


class VEBF(ListNeuronMixin, VersatileEllipticBaseClassifier):
    """Versatile Elliptic Basis Function (VEBF) classifier.

    Simplified variant of LRHE: no shift-and-shrink step. A single global
    average pairwise distance (computed over all classes together) is used as
    the initial width for every new class, rather than a per-class distance.
    Otherwise follows the same create-or-update-then-merge pattern per sample.

    Parameters
    ----------
    theta : float
        Overlap threshold for merge_neuron.
    delta : float
        Scaling factor for initial pairwise-distance width.
    epsilon : float
        Numerical floor added to widths and eigenvalues.
    """

    def __init__(self, theta=0, delta=1, epsilon=1e-10):
        self.neuron_list = []
        self.init_width = {}
        self.delta = delta
        self.theta = theta
        self.epsilon = epsilon

    def width_init(self, X, y):
        """Initialises width for new classes using a single global average pairwise distance.

        Unlike the base class (which computes per-class distance), VEBF uses one distance
        value computed over the entire batch X, then assigns it to all new classes.
        """
        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class
        if len(new_class) > 0:
            average_distance = self.average_pairwise_distance(X)
            for y_ in new_class:
                self.init_width[y_] = average_distance

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
