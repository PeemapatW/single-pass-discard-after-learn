import numpy as np
import numpy.linalg as LA

from ._base import ListNeuronMixin, VersatileEllipticBaseClassifier, PrincipleProjectionBaseClassifier, _SQRT_2PI


class D4(ListNeuronMixin, VersatileEllipticBaseClassifier, PrincipleProjectionBaseClassifier):
    def __init__(self, norm=2, delta=1, alpha=1, max_d=None, epsilon=1e-10, r=1.5, threshold=15):
        self.neuron_list = []
        self.init_width = {}
        self.norm = norm
        self.delta = delta
        self.alpha = alpha
        self.max_d = max_d
        self.epsilon = epsilon
        self.threshold = threshold
        self.r = r

    def create_new_neuron(self, X, y):
        n, d = X.shape
        if n > 1:
            cen = X.mean(axis=0)
            cov = np.cov(X.T)
            eig_c, pca_var = self.compute_sorted_eigencomponent(cov)
            initial_width = self.init_width[y]
            width = _SQRT_2PI * np.sqrt(np.abs(pca_var)) * self.alpha + (initial_width) * (1 - self.alpha)
        else:
            cen = X[0]
            cov = np.identity(d)
            eig_c = np.identity(d)
            pca_var = np.ones(d)
            width = self.init_width[y]
        neuron = {'y': y, 'cov': cov, 'center': cen, 'eig_component': eig_c, 'width': width, 'variance': pca_var, 'n': n}
        return neuron

    def fit(self, X, y, classes=None, _reset=True):
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
                new_w = _SQRT_2PI * np.sqrt(np.abs(pca_var)) * self.alpha + (old_w + np.abs(np.matmul(eig_c, merge_cen - old_cen))) * (1 - self.alpha)

                self.neuron_list[idx]['n'] = merge_n
                self.neuron_list[idx]['center'] = merge_cen
                self.neuron_list[idx]['cov'] = merge_cov
                self.neuron_list[idx]['eig_component'] = eig_c
                self.neuron_list[idx]['variance'] = pca_var
                self.neuron_list[idx]['width'] = new_w

        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        self.fit(X, y, _reset=False)

    def get_neuron_data(self, neuron_list, idx):
        neuron = neuron_list[idx]
        return neuron['center'], neuron['eig_component'].real, neuron['width']

    def predict(self, X):
        if self.max_d is None:
            self.max_d = len(X[0])

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
