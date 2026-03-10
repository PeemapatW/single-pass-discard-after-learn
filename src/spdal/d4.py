import numpy as np
import numpy.linalg as LA

from ._base import ListNeuronMixin, VersatileEllipticBaseClassifier, PrincipleProjectionBaseClassifier, _SQRT_2PI


class D4(ListNeuronMixin, VersatileEllipticBaseClassifier, PrincipleProjectionBaseClassifier):
    """Dimensionality-Driven Dual-Distance Discriminant (D4) classifier.

    Maintains exactly one neuron per class. Each fit() call (or partial_fit chunk)
    merges all new samples of a class into its single neuron. 
    Width is a combination of the Gaussian width (sqrt(2π|λ|)) and the
    displacement-corrected previous width, controlled by width_parameter.

    During predict(), when the two closest neurons belong to different classes,
    principal axes are paired by smallest angle and the point is assigned to the
    neuron with smaller projected distance in that reduced subspace.

    Parameters
    ----------
    norm : int
        Norm order for distance calculation (default 2 = Euclidean).
    delta : float
        Scaling factor for initial pairwise-distance width.
    width_parameter : float
        Blending weight: 1.0 = pure Gaussian width, 0.0 = pure displacement-based width.
    reduce_dims : int
        Number of principal axes to drop in eigenprojection (0 = use all axes).
    epsilon : float
        Numerical floor for widths and distances.
    r : float
        Radius scaling factor in calculate_proj_dist.
    threshold : float
        Angle threshold (degrees) for find_index_pairs.
    """

    def __init__(self, norm=2, delta=1, width_parameter=1, reduce_dims=0, epsilon=1e-10, r=1.5, threshold=15):
        self.neuron_list = []
        self.init_width = {}
        self.norm = norm
        self.delta = delta
        self.width_parameter = width_parameter
        self.reduce_dims = reduce_dims
        self.epsilon = epsilon
        self.threshold = threshold
        self.r = r

    def create_new_neuron(self, X, y):
        """Creates a neuron from a full batch X (one class). Uses covariance of X when n > 1.

        Overrides base class: accepts a batch instead of a single point.
        Width is a blend of Gaussian width and initial pairwise-distance width.
        """
        n, d = X.shape
        if n > 1:
            cen = X.mean(axis=0)
            cov = np.cov(X.T)
            eig_c, pca_var = self.compute_sorted_eigencomponent(cov)
            initial_width = self.init_width[y]
            width = _SQRT_2PI * np.sqrt(np.abs(pca_var)) * self.width_parameter + (initial_width) * (1 - self.width_parameter)
        else:
            cen = X[0]
            cov = np.identity(d)
            eig_c = np.identity(d)
            pca_var = np.ones(d)
            width = self.init_width[y]
        neuron = {'y': y, 'cov': cov, 'center': cen, 'eig_component': eig_c, 'width': width, 'variance': pca_var, 'n': n}
        return neuron

    def fit(self, X, y, classes=None, _reset=True):
        """Train on X, y — one neuron per class, merged from the full batch. Resets unless _reset=False."""
        if _reset:
            self.neuron_list = []
            self.init_width = {}
        all_class = np.unique(y)
        self.width_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]
            if not self.check_neuron_class_exist(y_):
                neuron = self.create_new_neuron(Xy, y_)
                self.neuron_list.append(neuron)
            else:
                idx = next(i for i, n in enumerate(self.neuron_list) if n['y'] == y_)
                new_n, d = Xy.shape
                if new_n > 1:
                    new_cen = Xy.mean(axis=0)
                    new_cov = np.cov(Xy.T)
                else:
                    new_cov = np.identity(d)
                    new_cen = Xy[0]
                old_cov = self.neuron_list[idx]['cov']
                old_cen = self.neuron_list[idx]['center']
                old_n = self.neuron_list[idx]['n']
                old_w = self.neuron_list[idx]['width']
                merge_n = new_n + old_n
                merge_cen = (new_n * new_cen + old_n * old_cen) / merge_n
                merge_cov = self._merge_covariance(new_n, new_cov, new_cen, old_n, old_cov, old_cen)
                eig_c, pca_var = self.compute_sorted_eigencomponent(merge_cov)
                new_w = _SQRT_2PI * np.sqrt(np.abs(pca_var)) * self.width_parameter + (old_w + np.abs(np.matmul(eig_c, merge_cen - old_cen))) * (1 - self.width_parameter)

                self.neuron_list[idx]['n'] = merge_n
                self.neuron_list[idx]['center'] = merge_cen
                self.neuron_list[idx]['cov'] = merge_cov
                self.neuron_list[idx]['eig_component'] = eig_c
                self.neuron_list[idx]['variance'] = pca_var
                self.neuron_list[idx]['width'] = new_w

        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        """Incrementally train on X, y — merges into existing per-class neurons."""
        self.fit(X, y, _reset=False)

    def get_neuron_data(self, neuron_list, idx):
        """Returns (center, eig_component, width) for neuron at idx — used by predict_with_eigen_proj."""
        neuron = neuron_list[idx]
        return neuron['center'], neuron['eig_component'].real, neuron['width']

    def predict(self, X):
        """Predicts class labels using eigenprojection distance with principal-axis pairing."""
        dist = np.empty((len(X), len(self.neuron_list)))
        for idx, neuron in enumerate(self.neuron_list):
            center, eig_c, w = self.get_neuron_data(self.neuron_list, idx)
            x_centered = X - center
            Margin = w + self.epsilon
            dist[:, idx] = self.calculate_proj_dist(x_centered, eig_c, Margin, self.norm)

        argsort_dist = np.argsort(dist, axis=1)
        if argsort_dist.shape[1] > 1:
            y_pred = self.predict_with_eigen_proj(self.neuron_list, X, argsort_dist)
        else:
            y_pred = np.array([self.neuron_list[i]['y'] for i in np.argmin(dist, axis=1)])
        return np.array(y_pred)
