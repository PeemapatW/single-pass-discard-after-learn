import numpy as np
import numpy.linalg as LA
import pandas as pd

from ._base import ScalableHyperelipsoidBaseClassifier, PrincipleProjectionBaseClassifier, _SQRT_2PI
from ._utils import get_axis_sample_points


class TRACED(ScalableHyperelipsoidBaseClassifier, PrincipleProjectionBaseClassifier):

    def __init__(self, norm=2, epsilon=1e-10, method="overlap-outside", r=_SQRT_2PI, N0=3, delta=2,
                alpha=0, beta=0, variance_threshold = 1, components = None,
                pca_strategy = 'bottom', distance_metric = 'boundary', overlap_selection = "parallel",
                width_parameter = 1, reduce_dims = 0, threshold = 15, min_dims=None, threshold_percentile = 100) -> None:

        # Attributes
        self.neuron_list = pd.DataFrame([])
        self.dist_ths = {}
        self.epsilon = epsilon

        # Hyper-parameter
        self.norm = norm
        self.r = r
        self.N0 = N0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.method = method
        self.variance_threshold = variance_threshold
        self.components = components
        self.pca_strategy = pca_strategy
        self.distance_metric = distance_metric
        self.width_parameter = width_parameter
        self.reduce_dims = reduce_dims
        self.threshold = threshold
        self.min_dims = min_dims
        self.overlap_selection = overlap_selection
        self.num_sample = 5
        self.threshold_percentile = threshold_percentile

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
        neuron = {'y':y,'cov':cov,'center':cen,'eig_component':eig_c,'variance':pca_var,'width':width,'displacement':displacement,'expansion':expansion,'n':n}
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
        variance_alpha = neuron["variance"]
        width_alpha = neuron["width"]
        cov_alpha = neuron["cov"]
        n_alpha = neuron["n"]
        displacement_alpha = neuron["displacement"]
        expansion_alpha = neuron["expansion"]
        n_Y = len(Y)
        center_Y = np.mean(Y, axis=0)
        n_new = n_alpha + n_Y
        cen_new = (n_alpha * cen_alpha + n_Y * center_Y) / n_new

        # Optimize covariance calculation
        Y_sum = np.sum(Y[:, :, np.newaxis] * Y[:, np.newaxis, :], axis=0)  # Vectorized sum of outer products
        cov_new = (
            n_alpha * (cov_alpha + np.outer(cen_alpha, cen_alpha)) / n_new
            + Y_sum / n_new
            - np.outer(cen_new, cen_new)
        )

        eig_c_new, pca_var_new = self.compute_sorted_eigencomponent(cov_new)

        width_new = self.r*np.sqrt(np.abs(pca_var_new))*self.width_parameter + (width_alpha + np.abs(np.matmul(eig_c_new,cen_new-cen_alpha)))*(1-self.width_parameter)

        # Compute Displacement and Expansion
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

        # Efficiently remove assigned data points from X
        X_new = np.delete(X, Y_index, axis=0)

        # Update neuron parameters in the neuron list
        self.neuron_list.at[alpha, 'center'] = cen_new
        self.neuron_list.at[alpha, 'eig_component'] = eig_c_new
        self.neuron_list.at[alpha, 'n'] = n_new
        self.neuron_list.at[alpha, 'cov'] = cov_new
        self.neuron_list.at[alpha, 'variance'] = pca_var_new
        self.neuron_list.at[alpha, 'width'] = width_new
        self.neuron_list.at[alpha, 'displacement'] = displacement_new*self.alpha + displacement_alpha*(1-self.alpha)
        self.neuron_list.at[alpha, 'expansion'] = expansion_new*self.beta + expansion_alpha*(1-self.beta)

        return X_new

    def merge_neuron(self, alpha, y):

        neuron_list_y = self.neuron_list.query(f'y == {y}')
        merged = False
        beta = 0
        if len(neuron_list_y) > 1:
            cen_alpha = self.neuron_list.at[alpha, "center"]
            cov_alpha = self.neuron_list.at[alpha, "cov"]
            eig_alpha = self.neuron_list.at[alpha, 'eig_component']
            width_alpha = self.neuron_list.at[alpha, "width"]
            distances = [LA.norm(cen_alpha-neuron_list_y.at[idx,'center']) for idx in neuron_list_y.index]
            sorted_neuron_idx = neuron_list_y.index[np.argsort(distances)]
            for beta in sorted_neuron_idx:
                if alpha != beta:
                    cov_beta = self.neuron_list.at[beta, 'cov']
                    cen_beta = self.neuron_list.at[beta, 'center']
                    eig_beta = self.neuron_list.at[beta, 'eig_component']
                    width_beta = self.neuron_list.at[beta, 'width']

                    # Precompute some matrices for efficiency
                    cov_tilde_alpha = eig_alpha @ np.diag(width_alpha**2) @ eig_alpha.T
                    cov_tilde_inv_beta = eig_beta @ np.diag(1 / (width_beta**2)) @ eig_beta.T

                    F = (-cen_alpha + cen_beta) @ cov_tilde_inv_beta
                    D = cov_tilde_alpha  @ cov_tilde_inv_beta + np.outer(cen_alpha, F)

                    # Construct P matrix using more efficient methods
                    P_upper = np.hstack([D, -D @ cen_beta[:, np.newaxis] + cen_alpha[:, np.newaxis]])
                    P_lower = np.hstack([F, -F @ cen_beta[:, np.newaxis] + 1])
                    P = np.vstack([P_upper, P_lower])

                    eig_P = LA.eig(P)[0]

                    # Check merge conditions
                    if all(np.isreal(eig_P)) and len(np.unique(eig_P)) == len(eig_P) and any(np.less(eig_P, 0)):
                        pass  # Conditions not met, continue to next neuron
                    else:
                        # Gather neuron parameters
                        n_alpha = self.neuron_list.at[alpha, 'n']
                        n_beta = self.neuron_list.at[beta, 'n']

                        # Calculate merged parameters
                        n_gamma = n_alpha + n_beta
                        cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                        cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)

                        eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)
                        # width_gamma = self.r*np.sqrt(np.abs(pca_var_gamma))*self.width_parameter + (width_alpha + np.abs(np.matmul(eig_c_gamma,cen_gamma-cen_alpha)))*(1-self.width_parameter)
                        width_gamma = self.r*np.sqrt(np.abs(pca_var_gamma))

                        # Update neuron beta with merged parameters
                        self.neuron_list.at[alpha, 'n'] = n_gamma
                        self.neuron_list.at[alpha, 'center'] = cen_gamma
                        self.neuron_list.at[alpha, 'cov'] = cov_gamma
                        self.neuron_list.at[alpha, 'eig_component'] = eig_c_gamma
                        self.neuron_list.at[alpha, 'variance'] = pca_var_gamma
                        self.neuron_list.at[alpha, 'width'] = width_gamma
                        self.neuron_list.at[alpha, 'displacement'] = np.zeros(len(cen_gamma))
                        self.neuron_list.at[alpha, 'expansion'] = np.ones(len(cen_gamma))
                        # Remove neuron alpha
                        self.neuron_list = self.neuron_list.drop(beta, axis=0)
                        merged = True
                        break  # Stop after merging
        return merged

    def create_and_update(self, X, y):

        # Create a new neuron and add it to the neuron list
        X, neuron = self.create_new_neuron(X, y)
        self.neuron_list = pd.concat([self.neuron_list, pd.DataFrame([neuron])], ignore_index=True)

        # Update distance threshold for the class
        self.dist_ths_y_update(y)

        alpha = self.neuron_list.index[-1]

        # Select data points and update neuron parameters
        Y, Y_index = self.select_update_data(X, neuron)
        if len(Y) != 0:
            X = self.update_parameter(neuron, alpha, X, Y, Y_index)

        return X

    def find_and_update(self, X, y):

        while len(X) != 0:
            neuron_list_y = self.neuron_list.query(f'y == {y}')
            x_mean = np.mean(X, axis=0)

            # Finding the closest neuron
            distances = [LA.norm(x_mean-neuron_list_y.at[idx,'center']) for idx in neuron_list_y.index]
            alpha = neuron_list_y.index[np.argmin(distances)]

            neuron = self.neuron_list.loc[alpha].to_dict()

            # Select data points and update neuron parameters
            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)

                # Attempt to merge neurons
                merged = self.merge_neuron(alpha, y)
                while merged:
                    merged = self.merge_neuron(alpha, y)
            else:
                break  # No data points assigned, exit the loop

        return X

    def find_and_capture(self, X, y):

        while len(X) != 0:
            # Create a new neuron and add it to the neuron list
            X, neuron = self.create_new_neuron(X, y)
            self.neuron_list = pd.concat([self.neuron_list, pd.DataFrame([neuron])], ignore_index=True)

            # Update distance threshold for the class
            self.dist_ths_y_update(y)

            alpha = self.neuron_list.index[-1]

            # Select data points and update neuron parameters
            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)

            # Attempt to merge neurons
            merged = self.merge_neuron(alpha, y)
            while merged:
                merged = self.merge_neuron(alpha, y)

        return X

    def fit(self, X, y, classes=None):
        all_class = np.unique(y)
        self.distance_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]
            if not self.check_neuron_class_exist(y_):
                Xy = self.create_and_update(Xy, y_)
            Xy = self.find_and_update(Xy, y_)
            Xy = self.find_and_capture(Xy, y_)

        self.set_classes()

    def partial_fit(self, X, y, classes=None, verbose=0):
        self.fit(X, y)

    def calculate_proj_dist(self, x_centered, P, M, norm):
        P_d_x = np.tensordot(x_centered, P, axes=(1, 1))  # Project data points
        return (LA.norm(P_d_x / M, ord=norm, axis=1)) - self.r  # Calculate distance

    def get_neuron_data(self, neuron_list_test, idx):
        neuron = neuron_list_test.iloc[idx]
        return neuron['center'], neuron['eig_component'].real, np.sqrt(neuron['variance'])

    def dist_ths_y_update(self, y):
        ths = self.dist_ths[y]
        S = self.neuron_list
        # Count neurons with 'n' < self.N0 and 'y' == y
        m = np.sum((S['n'] < self.N0) & (S['y'] == y))
        # Check condition and update threshold
        if m > len(S[S['y'] == y]) / 2:
            self.dist_ths[y] = ths * 2

    def _calculate_boundary_distance(self, x_centered, P, M):
        distances = self._calculate_center_distance(x_centered, P, M)
        geo_distance = LA.norm(x_centered,axis=1,ord=self.norm) * (1-1/distances)
        return geo_distance

    def _calculate_center_distance(self, x_centered, P, M):
        normalized_projections = (x_centered @ P.T) / M
        distances = np.sum(np.abs(normalized_projections)**self.norm, axis=1) ** (1/self.norm)
        return distances

    def _find_non_overlapping_axes(self, edge_proj0, edge_proj1):
        """
        Determines the indices of the most non-overlapping axes, ensuring both
        sets have the same number of dimensions.
        """
        # 1. Correctly count the number of available non-overlapping axes for each ellipsoid.
        non_overlap_count0 = np.sum(edge_proj0 >= 0)
        non_overlap_count1 = np.sum(edge_proj1 >= 0)

        # 2. Determine the number of common axes to select. This is the minimum of the two counts.
        num_to_select = np.min([non_overlap_count0, non_overlap_count1])

        # 3. Handle the edge case: If there are no common non-overlapping axes,
        #    fall back to selecting the single "least overlapping" axis from each.
        num_to_select = max(1, num_to_select, self.min_dims)

        # 4. Select the indices corresponding to the largest projection distances (most non-overlapping).
        #    This part of your logic was already correct.
        indices0 = np.argsort(edge_proj0)[-num_to_select:]
        indices1 = np.argsort(edge_proj1)[-num_to_select:]

        non_overlap_index = [indices0, indices1]

        # Sanity check to ensure the dimensions match.
        assert len(non_overlap_index[0]) == len(non_overlap_index[1])

        return non_overlap_index

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
                if len(result) == self.min_dims:
                    break
        if len(result) < self.min_dims:
            all_rows = set(range(rows))
            all_cols = set(range(cols))
            remaining_rows = sorted(all_rows - used_rows, reverse=True)
            remaining_cols = sorted(all_cols - used_cols, reverse=True)
            remaining_pairs = list(zip(remaining_rows, remaining_cols))
            result.extend(remaining_pairs[:self.min_dims - len(result)])

        return np.array(result).T

    def reconstruct_cov_from_components(self, P, M, indices_to_keep):
        """
        Reconstructs a covariance matrix from a selected subset of its eigen-components.

        Args:
            P (np.ndarray): The full matrix with eigenvectors as rows (D x D).
            M (np.ndarray): The full 1D array of semi-axis widths (sqrt of eigenvalues) (D,).
            indices_to_keep (list or np.ndarray): The indices of the components to use.

        Returns:
            np.ndarray: The reconstructed, simplified covariance matrix (D x D).
        """
        P_k = P[indices_to_keep]
        M_k = M[indices_to_keep]

        # The eigenvalues are the square of the semi-axis widths
        Lambda_k = np.diag(M_k**2)

        # Reconstruct the covariance matrix: S_k = P_kᵀ * Λ_k * P_k
        S_k = P_k.T @ Lambda_k @ P_k
        return S_k

    def resolve_overlap_with_lda(self,x_overlap, neuron_A, neuron_B, num_components_to_keep=None, epsilon=1e-6):
        """
        Resolves classification for points in an overlap region using LDA,
        optionally on a simplified covariance matrix.
        """
        c_A, S_A, y_A = neuron_A['center'], neuron_A['cov'], neuron_A['y']
        c_B, S_B, y_B = neuron_B['center'], neuron_B['cov'], neuron_B['y']

        P_A = neuron_A["eig_component"]
        M_A = neuron_A["width"]
        P_B = neuron_B["eig_component"]
        M_B = neuron_B["width"]

        edgeset0 = get_axis_sample_points(c_A,P_A,M_A,self.num_sample)
        edgeset1 = get_axis_sample_points(c_B,P_B,M_B,self.num_sample)
        edge_proj0 = np.array([np.min(self._compute_distance(edges-c_B, P_B, M_B)) for edges in edgeset0])
        edge_proj1 = np.array([np.min(self._compute_distance(edges-c_A, P_A, M_A)) for edges in edgeset1])
        non_overlap_index = self._find_non_overlapping_axes(edge_proj0, edge_proj1)
        S_A = self.reconstruct_cov_from_components(P_A, M_A, non_overlap_index[0])
        S_B = self.reconstruct_cov_from_components(P_B, M_B, non_overlap_index[1])

        Sw = S_A + S_B
        D = Sw.shape[0]
        Sw_reg = Sw + epsilon * np.identity(D)

        try:
            Sw_inv = np.linalg.inv(Sw_reg)
        except np.linalg.LinAlgError:
            dist_to_A = np.linalg.norm(x_overlap - c_A, axis=1)
            dist_to_B = np.linalg.norm(x_overlap - c_B, axis=1)
            is_B_closer = dist_to_B < dist_to_A
            return np.where(is_B_closer, y_B, y_A), None

        w = Sw_inv @ (c_A - c_B)
        w = w / np.linalg.norm(w)

        proj_x = x_overlap @ w
        proj_c_A = c_A @ w
        proj_c_B = c_B @ w

        is_B_closer = np.abs(proj_x - proj_c_B) < np.abs(proj_x - proj_c_A)
        predictions = np.where(is_B_closer, y_B, y_A)

        return predictions, w

    def _get_principal_components(self, eig_c, width, k):
        if self.pca_strategy.lower() == 'top':
            top_k_eig_c = eig_c[:k]
            top_k_width = width[:k]
        elif self.pca_strategy.lower() == 'bottom':
            top_k_eig_c = eig_c[-k:]
            top_k_width = width[-k:]
        else:
            raise ValueError(f"pca_strategy should be 'top' or 'bottom'")
        return top_k_eig_c, top_k_width

    def _compute_distance(self, x_centered, P, M):
        if self.distance_metric.lower() == 'boundary':
            distances = self._calculate_boundary_distance(x_centered, P, M)
        elif self.distance_metric.lower() == 'center':
            distances = self._calculate_center_distance(x_centered, P, M)**self.norm - 1
        elif self.distance_metric.lower() == "projection_ratio":
            distances = self._calculate_projection_ratio(x_centered, P, M)
        else:
            raise ValueError(f"distance_metric should be 'boundary' or 'center'")
        return distances

    def predict(self, X):

        neuron_list_test = self.neuron_list.query(f'n >= {self.N0}')

        if self.min_dims is None:
            self.min_dims = max(1, X.shape[1] - self.reduce_dims)

        k_lists = []
        if self.components is not None:
            k = self.components
        elif self.variance_threshold == 1:
            k = X.shape[1]
        else:
            k_lists = []
            for idx, neuron in neuron_list_test.iterrows():
                variances = neuron['variance']
                total_variance = np.sum(variances)
                cumulative_variance_ratio = np.cumsum(variances) / total_variance
                k_lists.append(np.argmax(cumulative_variance_ratio >= self.variance_threshold) + 1)
            k = max(k_lists)

        dist_df = {}
        for idx, neuron in enumerate(neuron_list_test.to_dict('records')):
            center = neuron['center']
            eig_c = neuron['eig_component']
            width = neuron['width'] + self.epsilon
            top_k_eig_c, top_k_width = self._get_principal_components(eig_c, width, k)
            dist_df[idx] = self._compute_distance(X - center, top_k_eig_c, top_k_width)
        dist_df = pd.DataFrame(dist_df)
        argsorted_indices = np.argsort(dist_df.values, axis=1)
        if argsorted_indices.shape[1] == 1:
            first_index = argsorted_indices[:, 0]
            y_pred = neuron_list_test['y'].values[first_index]
            return y_pred
        first_index, second_index = argsorted_indices[:, 0], argsorted_indices[:, 1]
        first_values = dist_df.values[np.arange(len(X)), first_index]
        second_values = dist_df.values[np.arange(len(X)), second_index]
        top_two_index = np.array([sorted(idx) for idx in zip(first_index, second_index)])
        unique_top_two = list(set(map(tuple, top_two_index)))
        y_pred = neuron_list_test['y'].values[first_index]

        for unique_pair in unique_top_two:
            initial_overlap_mask = (first_values <= 0) & (second_values <= 0)
            if np.any(initial_overlap_mask):
                f_vals_overlap = first_values[initial_overlap_mask]
                s_vals_overlap = second_values[initial_overlap_mask]
                overlap_scores = np.maximum(f_vals_overlap, s_vals_overlap)
                projection_threshold = np.percentile(overlap_scores, self.threshold_percentile)
                should_project_mask = (initial_overlap_mask) & (np.maximum(first_values, second_values) <= projection_threshold)
                mask = (top_two_index == unique_pair).all(axis=1) & should_project_mask
                if np.any(mask) and 'overlap' in self.method.split("-") and self.reduce_dims > 0:
                    if neuron_list_test.iloc[unique_pair[0]]["y"] != neuron_list_test.iloc[unique_pair[1]]["y"]:
                        self.count_overlap += sum(mask)
                        x_unique_pair = X[mask]

                        centers = [neuron_list_test.iloc[idx]["center"] for idx in unique_pair]
                        eig_cs = [neuron_list_test.iloc[idx]["eig_component"] for idx in unique_pair]
                        widths = [neuron_list_test.iloc[idx]["width"] for idx in unique_pair]
                        eig_c_1_k, width_1_k = self._get_principal_components(eig_cs[0], widths[0], k)
                        eig_c_2_k, width_2_k = self._get_principal_components(eig_cs[1], widths[1], k)
                        eig_cs_k = [eig_c_1_k, eig_c_2_k]
                        widths_k = [width_1_k, width_2_k]

                        if self.overlap_selection.lower() == "sampling":
                            edgeset0 = get_axis_sample_points(centers[0],eig_cs_k[0],widths_k[0],self.num_sample)
                            edgeset1 = get_axis_sample_points(centers[1],eig_cs_k[1],widths_k[1],self.num_sample)
                            edge_proj0 = np.array([np.min(self._compute_distance(edges-centers[1], eig_cs_k[1], widths_k[1])) for edges in edgeset0])
                            edge_proj1 = np.array([np.min(self._compute_distance(edges-centers[0], eig_cs_k[0], widths_k[0])) for edges in edgeset1])

                            if np.any(edge_proj0 < 0) or np.any(edge_proj1 < 0):
                                non_overlap_index = self._find_non_overlapping_axes(edge_proj0, edge_proj1)
                                Ps = [eig_c[non_overlap_index[i]] for i, eig_c in enumerate(eig_cs_k)]
                                Ms = [width[non_overlap_index[i]] for i, width in enumerate(widths_k)]

                                x_centereds = [x_unique_pair - center for center in centers]
                                x_proj_dist = {
                                    y: self._compute_distance(x_centered, P, M)
                                    for y, x_centered, P, M in zip(unique_pair, x_centereds, Ps, Ms)
                                }
                                x_proj_dist_df = pd.DataFrame(x_proj_dist)
                                y_predict_unique_pair = neuron_list_test['y'].iloc[x_proj_dist_df.idxmin(axis=1).tolist()]
                                y_pred[mask] = y_predict_unique_pair

                        elif self.overlap_selection.lower() == "parallel":
                            eig_angle = self.angle_between_unit_vectors(eig_cs_k[0], eig_cs_k[1])
                            non_overlap_index = self.find_index_pairs(eig_angle)
                            Ps = [eig_c[non_overlap_index[i]] for i, eig_c in enumerate(eig_cs_k)]
                            Ms = [width[non_overlap_index[i]] for i, width in enumerate(widths_k)]
                            x_centereds = [x_unique_pair - center for center in centers]
                            x_proj_dist = {
                                y: self._compute_distance(x_centered, P, M)
                                for y, x_centered, P, M in zip(unique_pair, x_centereds, Ps, Ms)
                            }
                            x_proj_dist_df = pd.DataFrame(x_proj_dist)
                            y_predict_unique_pair = neuron_list_test['y'].iloc[x_proj_dist_df.idxmin(axis=1).tolist()]
                            y_pred[mask] = y_predict_unique_pair

                        else:
                            raise ValueError("")

            mask = (top_two_index == unique_pair).all(axis=1) & (first_values > 0) & (second_values > 0)
            if np.any(mask) and 'outside' in self.method.split("-"):
                if neuron_list_test.iloc[unique_pair[0]]["y"] != neuron_list_test.iloc[unique_pair[1]]["y"]:
                    self.count_outside += sum(mask)
                    x_unique_pair = X[mask]
                    centers = [neuron_list_test.iloc[idx]["center"] + neuron_list_test.iloc[idx]["displacement"] for idx in unique_pair]
                    eig_cs = [neuron_list_test.iloc[idx]["eig_component"] for idx in unique_pair]
                    widths = [neuron_list_test.iloc[idx]["width"] * neuron_list_test.iloc[idx]["expansion"] for idx in unique_pair]
                    eig_c_1_k, width_1_k = self._get_principal_components(eig_cs[0], widths[0], k)
                    eig_c_2_k, width_2_k = self._get_principal_components(eig_cs[1], widths[1], k)
                    eig_cs_k = [eig_c_1_k, eig_c_2_k]
                    widths_k = [width_1_k, width_2_k]
                    x_centereds = [x_unique_pair - center for center in centers]

                    x_proj_dist = {
                        y: self._compute_distance(x_centered, P, M)
                        for y, x_centered, P, M in zip(unique_pair, x_centereds, eig_cs_k, widths_k)
                    }

                    x_proj_dist_df = pd.DataFrame(x_proj_dist)
                    y_predict_unique_pair = neuron_list_test['y'].iloc[x_proj_dist_df.idxmin(axis=1).tolist()]

                    y_pred[mask] = y_predict_unique_pair
        return y_pred
