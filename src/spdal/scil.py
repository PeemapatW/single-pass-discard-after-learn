import numpy as np
import numpy.linalg as LA

from ._base import VersatileEllipticBaseClassifier, ListNeuronMixin, _SQRT_2PI


class SCIL(ListNeuronMixin, VersatileEllipticBaseClassifier):
    def __init__(self, N0=3, eta=2, delta=1, epsilon=1e-10, theta=0):
        self.neuron_list = []
        self.init_width = {}
        self.delta = delta
        self.N0 = N0
        self.eta = eta
        self.theta = theta
        self.epsilon = epsilon

    def width_init(self, X, y):
        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class
        if len(new_class) > 0:
            average_distance = self.average_pairwise_distance(X)
            for y_ in new_class:
                self.init_width[y_] = average_distance

    def create_new_neuron(self, X, y):
        select_index = 0
        cen = X[select_index, :]
        cov = np.zeros([len(cen)]*2)
        eig_c = np.identity(len(cen))
        pca_var = np.ones(len(cen))
        width = self.init_width[y]
        n = 1
        neuron = {'y': y, 'cov': cov, 'center': cen, 'eig_component': eig_c, 'variance': pca_var, 'width': width, 'n': n}
        X_ = np.delete(X, select_index, axis=0)
        return X_, neuron

    def select_update_data(self, X, neuron):
        center = neuron["center"]
        eig_c = neuron["eig_component"]
        width = neuron['width'] + self.epsilon
        n = neuron["n"]

        center_temp = (n * center + X) / (n + 1)
        x_centered = X - center_temp
        P_d_x = np.tensordot(x_centered, eig_c, axes=(1, 1))
        Psi = LA.norm(P_d_x / width, ord=2, axis=1) ** 2 - 1

        Y_index = np.where(Psi <= 0)[0]
        Y = X[Y_index]

        return Y, Y_index

    def update_parameter(self, neuron, alpha, X, Y, Y_index):
        cen_alpha = neuron["center"]
        cov_alpha = neuron["cov"]
        n_alpha = neuron["n"]
        width_alpha = neuron["width"]
        n_Y = len(Y)
        cen_Y = np.mean(Y, axis=0)
        n_new = n_alpha + n_Y
        cen_new = (n_alpha * cen_alpha + n_Y * cen_Y) / n_new

        Y_sum = np.sum(Y[:, :, np.newaxis] * Y[:, np.newaxis, :], axis=0)
        cov_new = (
            n_alpha * (cov_alpha + np.outer(cen_alpha, cen_alpha)) / n_new
            + Y_sum / n_new
            - np.outer(cen_new, cen_new)
        )

        eig_c_new, pca_var_new = self.compute_sorted_eigencomponent(cov_new)
        width_new = np.array([width_alpha[d] + np.abs(np.matmul(cen_new - cen_alpha, eig_c_new[d].T)) for d in range(len(width_alpha))])
        max_psi = np.max([self.hyperellipsoidal_fn(y, cen_new, eig_c_new, width_new) for y in Y])
        if max_psi > 0:
            width_new = np.sqrt(1 + self.eta * max_psi) * width_new
        X_new = np.delete(X, Y_index, axis=0)

        self.neuron_list[alpha]['center'] = cen_new
        self.neuron_list[alpha]['eig_component'] = eig_c_new
        self.neuron_list[alpha]['n'] = n_new
        self.neuron_list[alpha]['cov'] = cov_new
        self.neuron_list[alpha]['variance'] = pca_var_new
        self.neuron_list[alpha]['width'] = width_new

        return X_new

    def create_and_update(self, X, y):
        X, neuron = self.create_new_neuron(X, y)
        self.neuron_list.append(neuron)
        alpha = len(self.neuron_list) - 1

        Y, Y_index = self.select_update_data(X, neuron)
        if len(Y) != 0:
            X = self.update_parameter(neuron, alpha, X, Y, Y_index)

        return X

    def find_and_update(self, X, y):
        while len(X) != 0:
            neurons_y = [(i, n) for i, n in enumerate(self.neuron_list) if n['y'] == y]
            x_mean = np.mean(X, axis=0)
            alpha, neuron = min(neurons_y, key=lambda t: LA.norm(x_mean - t[1]['center']))

            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)
                self.merge_neuron(alpha, y)
            else:
                break

        return X

    def find_and_capture(self, X, y):
        while len(X) != 0:
            X, neuron = self.create_new_neuron(X, y)
            self.neuron_list.append(neuron)
            alpha = len(self.neuron_list) - 1

            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)

        return X

    def merge_neuron(self, alpha, y):
        neurons_y_idx = [i for i, n in enumerate(self.neuron_list) if n['y'] == y]
        if len(neurons_y_idx) > 1:
            cov_alpha = self.neuron_list[alpha]['cov']
            cen_alpha = self.neuron_list[alpha]['center']
            n_alpha = self.neuron_list[alpha]['n']
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

                        width_gamma = np.array([1.96*np.sqrt(np.abs(pca_var_gamma[d])/n_gamma) for d in range(len(pca_var_gamma))])  # 1.96 = z-score for 95% CI
                        self.neuron_list[beta]['n'] = n_gamma
                        self.neuron_list[beta]['center'] = cen_gamma
                        self.neuron_list[beta]['cov'] = cov_gamma
                        self.neuron_list[beta]['eig_component'] = eig_c_gamma
                        self.neuron_list[beta]['width'] = width_gamma
                        self.neuron_list[beta]['variance'] = pca_var_gamma

                        self.neuron_list.pop(alpha)
                        break

    def fit(self, X, y, classes=None):
        all_class = np.unique(y)
        self.width_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]
            if not self.check_neuron_class_exist(y_):
                Xy = self.create_and_update(Xy, y_)
            Xy = self.find_and_update(Xy, y_)
            Xy = self.find_and_capture(Xy, y_)
        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        self.fit(X, y)

    def predict(self, X):
        neurons_test = [n for n in self.neuron_list if n['n'] >= self.N0]
        dist = np.empty((len(X), len(neurons_test)))
        for idx, neuron in enumerate(neurons_test):
            center = neuron['center']
            eig_c = neuron['eig_component'].real
            width = _SQRT_2PI * np.sqrt(neuron['variance']).reshape(len(center))
            x_centered = X - center
            P_d_x = np.tensordot(x_centered, eig_c, axes=(1, 1))
            dist[:, idx] = LA.norm(P_d_x / width, ord=2, axis=1) ** 2 - 1
        y_pred = np.array([n['y'] for n in neurons_test])[np.argmin(dist, axis=1)]
        return y_pred
