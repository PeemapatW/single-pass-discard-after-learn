import numpy as np
import numpy.linalg as LA

from ._base import ListNeuronMixin, ScalableHyperelipsoidBaseClassifier, PrincipleProjectionBaseClassifier, _SQRT_2PI


class TRACED(ListNeuronMixin, ScalableHyperelipsoidBaseClassifier, PrincipleProjectionBaseClassifier):

    def __init__(self, norm=2, epsilon=1e-10, method="overlap-outside", r=_SQRT_2PI, N0=3, delta=2,
                alpha=0, beta=0, distance_metric='boundary', width_parameter=1, reduce_dims=0,
                threshold=15) -> None:

        # Attributes
        self.neuron_list = []
        self.dist_ths = {}
        self.epsilon = epsilon

        # Hyper-parameters
        self.norm = norm
        self.r = r
        self.N0 = N0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.method = method
        self.distance_metric = distance_metric
        self.width_parameter = width_parameter
        self.reduce_dims = reduce_dims
        self.threshold = threshold

        self.count_overlap = 0
        self.count_outside = 0

    def distance_init(self, X, y):
        all_class = np.unique(y)
        exist_class = set(self.dist_ths.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]
            initial_dist = self.find_mean_dist_to_neighbor(Xy) * self.delta
            self.dist_ths[y_] = initial_dist

    def create_new_neuron(self, X, y):
        select_index = 0
        cen = X[select_index, :]
        cov = np.zeros([len(cen)]*2) + np.identity(len(cen))*self.epsilon
        eig_c = np.identity(len(cen))
        pca_var = np.ones(len(cen))
        displacement = np.zeros(len(cen))
        expansion = np.ones(len(cen))
        width = np.zeros(len(cen)) + self.epsilon
        n = 1
        neuron = {'y': y, 'cov': cov, 'center': cen, 'eig_component': eig_c, 'variance': pca_var,
                  'width': width, 'displacement': displacement, 'expansion': expansion, 'n': n}
        X_ = np.delete(X, select_index, axis=0)
        return X_, neuron

    def select_update_data(self, X, neuron):
        center = neuron["center"]
        y = neuron["y"]
        distances = np.linalg.norm(X - center, axis=1)
        indices = np.where(distances <= self.dist_ths[y])[0]
        return X[indices], indices

    def update_parameter(self, neuron, alpha, X, Y, Y_index):
        cen_alpha = neuron["center"]
        eig_c_alpha = neuron["eig_component"]
        width_alpha = neuron["width"]
        cov_alpha = neuron["cov"]
        n_alpha = neuron["n"]
        displacement_alpha = neuron["displacement"]
        expansion_alpha = neuron["expansion"]
        n_Y = len(Y)
        center_Y = np.mean(Y, axis=0)
        n_new = n_alpha + n_Y
        cen_new = (n_alpha * cen_alpha + n_Y * center_Y) / n_new

        Y_sum = np.sum(Y[:, :, np.newaxis] * Y[:, np.newaxis, :], axis=0)
        cov_new = (
            n_alpha * (cov_alpha + np.outer(cen_alpha, cen_alpha)) / n_new
            + Y_sum / n_new
            - np.outer(cen_new, cen_new)
        )

        eig_c_new, pca_var_new = self.compute_sorted_eigencomponent(cov_new)

        width_new = (self.r * np.sqrt(np.abs(pca_var_new)) * self.width_parameter
                     + (width_alpha + np.abs(np.matmul(eig_c_new, cen_new - cen_alpha))) * (1 - self.width_parameter))

        if n_alpha != 1:
            displacement_new = cen_new - cen_alpha
            cov_alpha_reconstructed = eig_c_alpha @ np.diag(width_alpha**2) @ eig_c_alpha.T
            old_projected = np.array([
                np.sqrt(eig_c_new[i].T @ cov_alpha_reconstructed @ eig_c_new[i])
                for i in range(len(width_new))
            ])
            expansion_new = width_new / (old_projected + self.epsilon)
        else:
            displacement_new = np.zeros(len(cen_new))
            expansion_new = np.ones(len(cen_new))

        X_new = np.delete(X, Y_index, axis=0)

        self.neuron_list[alpha]['center'] = cen_new
        self.neuron_list[alpha]['eig_component'] = eig_c_new
        self.neuron_list[alpha]['n'] = n_new
        self.neuron_list[alpha]['cov'] = cov_new
        self.neuron_list[alpha]['variance'] = pca_var_new
        self.neuron_list[alpha]['width'] = width_new
        self.neuron_list[alpha]['displacement'] = displacement_new * self.alpha + displacement_alpha * (1 - self.alpha)
        self.neuron_list[alpha]['expansion'] = expansion_new * self.beta + expansion_alpha * (1 - self.beta)

        return X_new

    def merge_neuron(self, alpha, y):
        neurons_y = [(i, n) for i, n in enumerate(self.neuron_list) if n['y'] == y]
        if len(neurons_y) <= 1:
            return False, alpha

        cen_alpha = self.neuron_list[alpha]['center']
        eig_alpha = self.neuron_list[alpha]['eig_component']
        width_alpha = self.neuron_list[alpha]['width']
        cov_alpha = self.neuron_list[alpha]['cov']

        distances = [LA.norm(cen_alpha - n['center']) for _, n in neurons_y]
        sorted_y_idxs = [neurons_y[i][0] for i in np.argsort(distances)]

        for beta in sorted_y_idxs:
            if beta == alpha:
                continue
            nb = self.neuron_list[beta]
            cov_beta = nb['cov']
            cen_beta = nb['center']
            eig_beta = nb['eig_component']
            width_beta = nb['width']

            cov_tilde_alpha = eig_alpha @ np.diag(width_alpha**2) @ eig_alpha.T
            cov_tilde_inv_beta = eig_beta @ np.diag(1 / (width_beta**2)) @ eig_beta.T

            F = (-cen_alpha + cen_beta) @ cov_tilde_inv_beta
            D = cov_tilde_alpha @ cov_tilde_inv_beta + np.outer(cen_alpha, F)
            P_upper = np.hstack([D, -D @ cen_beta[:, np.newaxis] + cen_alpha[:, np.newaxis]])
            P_lower = np.hstack([F, -F @ cen_beta[:, np.newaxis] + 1])
            P = np.vstack([P_upper, P_lower])
            eig_P = LA.eig(P)[0]

            if all(np.isreal(eig_P)) and len(np.unique(eig_P)) == len(eig_P) and any(np.less(eig_P, 0)):
                pass
            else:
                n_alpha = self.neuron_list[alpha]['n']
                n_beta = nb['n']
                n_gamma = n_alpha + n_beta
                cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)
                eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)
                width_gamma = self.r * np.sqrt(np.abs(pca_var_gamma))

                self.neuron_list[alpha]['n'] = n_gamma
                self.neuron_list[alpha]['center'] = cen_gamma
                self.neuron_list[alpha]['cov'] = cov_gamma
                self.neuron_list[alpha]['eig_component'] = eig_c_gamma
                self.neuron_list[alpha]['variance'] = pca_var_gamma
                self.neuron_list[alpha]['width'] = width_gamma
                self.neuron_list[alpha]['displacement'] = np.zeros(len(cen_gamma))
                self.neuron_list[alpha]['expansion'] = np.ones(len(cen_gamma))
                self.neuron_list.pop(beta)
                if beta < alpha:
                    alpha -= 1
                return True, alpha

        return False, alpha

    def create_and_update(self, X, y):
        X, neuron = self.create_new_neuron(X, y)
        self.neuron_list.append(neuron)
        self.dist_ths_y_update(y)
        alpha = len(self.neuron_list) - 1
        Y, Y_index = self.select_update_data(X, neuron)
        if len(Y) != 0:
            X = self.update_parameter(neuron, alpha, X, Y, Y_index)
        return X

    def find_and_update(self, X, y):
        while len(X) != 0:
            neurons_y = [(i, n) for i, n in enumerate(self.neuron_list) if n['y'] == y]
            x_mean = np.mean(X, axis=0)
            distances = [LA.norm(x_mean - n['center']) for _, n in neurons_y]
            alpha = neurons_y[np.argmin(distances)][0]
            neuron = self.neuron_list[alpha]
            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)
                merged, alpha = self.merge_neuron(alpha, y)
                while merged:
                    merged, alpha = self.merge_neuron(alpha, y)
            else:
                break
        return X

    def find_and_capture(self, X, y):
        while len(X) != 0:
            X, neuron = self.create_new_neuron(X, y)
            self.neuron_list.append(neuron)
            self.dist_ths_y_update(y)
            alpha = len(self.neuron_list) - 1
            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)
            merged, alpha = self.merge_neuron(alpha, y)
            while merged:
                merged, alpha = self.merge_neuron(alpha, y)
        return X

    def fit(self, X, y, classes=None, _reset=True):
        if _reset:
            self.neuron_list = []
            self.dist_ths = {}
            self.count_overlap = 0
            self.count_outside = 0
        all_class = np.unique(y)
        self.distance_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]
            if not self.check_neuron_class_exist(y_):
                Xy = self.create_and_update(Xy, y_)
            Xy = self.find_and_update(Xy, y_)
            Xy = self.find_and_capture(Xy, y_)
        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        self.fit(X, y, _reset=False)

    def dist_ths_y_update(self, y):
        ths = self.dist_ths[y]
        neurons_y = [n for n in self.neuron_list if n['y'] == y]
        m = sum(1 for n in neurons_y if n['n'] < self.N0)
        if m > len(neurons_y) / 2:
            self.dist_ths[y] = ths * 2

    def _calculate_boundary_distance(self, x_centered, P, M):
        distances = self._calculate_center_distance(x_centered, P, M)
        geo_distance = LA.norm(x_centered, axis=1, ord=self.norm) * (1 - 1 / distances)
        return geo_distance

    def _calculate_center_distance(self, x_centered, P, M):
        normalized_projections = (x_centered @ P.T) / M
        distances = np.sum(np.abs(normalized_projections)**self.norm, axis=1) ** (1 / self.norm)
        return distances

    def _compute_distance(self, x_centered, P, M):
        if self.distance_metric.lower() == 'boundary':
            return self._calculate_boundary_distance(x_centered, P, M)
        elif self.distance_metric.lower() == 'center':
            return self._calculate_center_distance(x_centered, P, M)**self.norm - 1
        else:
            raise ValueError(f"distance_metric should be 'boundary' or 'center'")

    def find_index_pairs(self, arr):
        rows, cols = arr.shape
        n_pairs = max(1, rows - self.reduce_dims)
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

        return np.array(result).T

    def predict(self, X):
        neuron_list_test = [n for n in self.neuron_list if n['n'] >= self.N0]
        k = X.shape[1]

        dist = np.empty((len(X), len(neuron_list_test)))
        for idx, neuron in enumerate(neuron_list_test):
            center = neuron['center']
            eig_c = neuron['eig_component']
            width = neuron['width'] + self.epsilon
            dist[:, idx] = self._compute_distance(X - center, eig_c, width)

        y_values = np.array([n['y'] for n in neuron_list_test])
        argsorted_indices = np.argsort(dist, axis=1)

        if argsorted_indices.shape[1] == 1:
            return y_values[argsorted_indices[:, 0]]

        first_index, second_index = argsorted_indices[:, 0], argsorted_indices[:, 1]
        first_values = dist[np.arange(len(X)), first_index]
        second_values = dist[np.arange(len(X)), second_index]
        top_two_index = np.array([sorted(idx) for idx in zip(first_index, second_index)])
        unique_top_two = list(set(map(tuple, top_two_index)))
        y_pred = y_values[first_index].copy()

        for unique_pair in unique_top_two:
            mask = (top_two_index == unique_pair).all(axis=1) & (first_values <= 0) & (second_values <= 0)
            if np.any(mask) and 'overlap' in self.method.split("-") and self.reduce_dims > 0:
                if neuron_list_test[unique_pair[0]]['y'] != neuron_list_test[unique_pair[1]]['y']:
                    self.count_overlap += sum(mask)
                    x_unique_pair = X[mask]
                    centers = [neuron_list_test[idx]['center'] for idx in unique_pair]
                    eig_cs_k = [neuron_list_test[idx]['eig_component'][-k:] for idx in unique_pair]
                    widths_k = [neuron_list_test[idx]['width'][-k:] for idx in unique_pair]

                    eig_angle = self.angle_between_unit_vectors(eig_cs_k[0], eig_cs_k[1])
                    non_overlap_index = self.find_index_pairs(eig_angle)
                    Ps = [eig_c[non_overlap_index[i]] for i, eig_c in enumerate(eig_cs_k)]
                    Ms = [width[non_overlap_index[i]] for i, width in enumerate(widths_k)]
                    x_centereds = [x_unique_pair - center for center in centers]
                    x_proj_dist = np.column_stack([
                        self._compute_distance(x_centered, P, M)
                        for x_centered, P, M in zip(x_centereds, Ps, Ms)
                    ])
                    winner_col = np.argmin(x_proj_dist, axis=1)
                    winner_neuron_idx = np.array(unique_pair)[winner_col]
                    y_pred[mask] = np.array([neuron_list_test[i]['y'] for i in winner_neuron_idx])

            mask = (top_two_index == unique_pair).all(axis=1) & (first_values > 0) & (second_values > 0)
            if np.any(mask) and 'outside' in self.method.split("-"):
                if neuron_list_test[unique_pair[0]]['y'] != neuron_list_test[unique_pair[1]]['y']:
                    self.count_outside += sum(mask)
                    x_unique_pair = X[mask]
                    centers = [neuron_list_test[idx]['center'] + neuron_list_test[idx]['displacement'] for idx in unique_pair]
                    eig_cs_k = [neuron_list_test[idx]['eig_component'][-k:] for idx in unique_pair]
                    widths_k = [(neuron_list_test[idx]['width'] * neuron_list_test[idx]['expansion'])[-k:] for idx in unique_pair]
                    x_centereds = [x_unique_pair - center for center in centers]
                    x_proj_dist = np.column_stack([
                        self._compute_distance(x_centered, P, M)
                        for x_centered, P, M in zip(x_centereds, eig_cs_k, widths_k)
                    ])
                    winner_col = np.argmin(x_proj_dist, axis=1)
                    winner_neuron_idx = np.array(unique_pair)[winner_col]
                    y_pred[mask] = np.array([neuron_list_test[i]['y'] for i in winner_neuron_idx])

        return y_pred
