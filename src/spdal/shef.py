import numpy as np
import numpy.linalg as LA
import pandas as pd

from ._base import ScalableHyperelipsoidBaseClassifier


class SHEF(ScalableHyperelipsoidBaseClassifier):
    def __init__(self, M = 3,r = 1.5, epsilon=1e-10):
        self.neuron_list = pd.DataFrame([])
        self.dist_ths = {}
        self.M = M
        self.r = r
        self.epsilon = epsilon

    def create_new_neuron(self, x_i, y_i):
        cen = x_i
        cov = np.zeros([len(x_i),len(x_i)]) + np.identity(len(x_i))*self.epsilon
        n = 1
        neuron = {'y':y_i,'cov':cov,'center':cen,'n':n}
        return neuron

    def merge_neuron(self, alpha, y):
        """
        Merges two neurons if certain conditions are met.

        Args:
            alpha: The index of the first neuron.
            y: The class label of the neurons to merge.

        Returns:
            A tuple containing:
                - beta: The index of the second neuron (or 0 if no merge occurred).
                - merged: A boolean indicating whether a merge occurred.
        """

        neuron_list_y = self.neuron_list.query(f'y == {y}')
        n_alpha = self.neuron_list.at[alpha, "n"]
        cen_alpha = self.neuron_list.at[alpha, "center"]
        cov_alpha = self.neuron_list.at[alpha, "cov"]
        if len(neuron_list_y) > 1 and n_alpha > self.M:
            distances = [LA.norm(cen_alpha-neuron_list_y.at[idx,'center']) for idx in neuron_list_y.index]
            beta = neuron_list_y.index[np.argsort(distances)[1]]
            cen_beta = self.neuron_list.at[beta, 'center']
            cov_beta = self.neuron_list.at[beta, 'cov']

            # Precompute some matrices for efficiency
            cov_tilde_inv = LA.inv(cov_beta) / (self.r ** 2)
            cov_tilde = cov_alpha / (self.r ** 2)
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
                # Gather neuron parameters
                n_beta = self.neuron_list.at[beta, 'n']

                # Calculate merged parameters
                n_gamma = n_alpha + n_beta
                cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)

                # Update neuron beta with merged parameters
                self.neuron_list.at[beta, 'n'] = n_gamma
                self.neuron_list.at[beta, 'center'] = cen_gamma
                self.neuron_list.at[beta, 'cov'] = cov_gamma

                # Remove neuron alpha
                self.neuron_list = self.neuron_list.drop(alpha, axis=0)

    def fit(self, X, y, classes=None):
        self.distance_init(X, y)
        for x_i,y_i in zip(X, y):
            if self.check_neuron_class_exist(y_i):
                neuron_list_y = self.neuron_list.query('y==@y_i')
                # Finding the closest neuron
                alpha = neuron_list_y.index[np.argmin([LA.norm(x_i-neuron_list_y.at[idx,'center']) for idx in neuron_list_y.index])]
                if LA.norm(x_i-neuron_list_y.at[alpha,'center']) > self.dist_ths[y_i]:
                    # Create new neuron
                    neuron = self.create_new_neuron(x_i, y_i)
                    self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                    alpha = self.neuron_list.index[-1]
                    self.dist_ths_y_update(y_i)
                else:
                    cov_alpha = self.neuron_list.at[alpha,'cov']
                    cen_alpha = self.neuron_list.at[alpha,'center']
                    n_alpha = self.neuron_list.at[alpha,'n']
                    new_cen = (n_alpha*cen_alpha+x_i)/(n_alpha+1)
                    new_cov = n_alpha/(n_alpha+1)*(cov_alpha+np.matmul(np.array([cen_alpha-x_i]).T,np.array([cen_alpha-x_i]))/(n_alpha+1))
                    new_n = n_alpha+1
                    self.neuron_list.at[alpha,'cov'] = new_cov
                    self.neuron_list.at[alpha,'center'] = new_cen
                    self.neuron_list.at[alpha,'n'] = new_n
                self.merge_neuron(alpha, y_i)
            else:
                # Create new neuron
                neuron = self.create_new_neuron(x_i, y_i)
                self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                self.dist_ths_y_update(y_i)

        self.set_classes()

    def partial_fit(self, X, y, classes=None):
        self.fit(X, y)

    def _vectorized_discriminant_vector(self, c1, S1, c2, S2):
        """Calculates the discriminant vector between two SHEFs for a batch of neurons."""
        S_sum = S1 + S2
        S_sum_inv = LA.inv(S_sum) # np.linalg.inv handles batches of matrices
        c_diff = c2 - c1
        # Batch matrix-vector multiplication
        w = np.einsum('...ij,...j->...i', S_sum_inv, c_diff)
        return w

    def _vectorized_projection_distance(self, x, c, S, w):
        """Calculates the projection ratio distance for a batch of points."""
        x_centered = x - c
        # Batch dot product for the numerator
        numerator = np.abs(np.einsum('...i,...i->...', x_centered, w))

        # Batch quadratic form (w.T @ S @ w) for the denominator
        wSw = np.einsum('...i,...ij,...j->...', w, S, w)

        denominator = self.r * np.sqrt(wSw)
        denominator[denominator < self.epsilon] = self.epsilon # Vectorized handling of epsilon
        return numerator / denominator

    def predict(self, X):
        """
        New prediction function using vectorization to process all data points at once.
        Time complexity is O(k) where k is the number of neurons for the first part,
        and the second part is vectorized over the number of samples.
        """
        if self.neuron_list.shape[0] == 1:
            return np.array([self.neuron_list['y'].iloc[0]] * len(X))

        # --- Part 1: Calculate initial distances (same as your original code) ---
        self.neuron_list['cov_inv'] = self.neuron_list['cov'].apply(lambda x: LA.inv(x))
        dist = {}
        for idx, neuron in enumerate(self.neuron_list.to_dict('records')):
            center = neuron['center']
            cov = neuron['cov']
            cov_inv = neuron['cov_inv']
            x_centered = X - center
            wp = np.tensordot(x_centered, cov_inv, axes=(1, 1))
            wp_norm = LA.norm(wp, axis=1)[:, None]
            wp_norm[wp_norm < self.epsilon] = self.epsilon
            wp = wp / wp_norm
            r_wSw = self.r * np.sqrt(np.einsum('ij,ij->i', np.tensordot(wp, cov, axes=(1, 1)), wp))
            r_wSw[r_wSw < self.epsilon] = self.epsilon
            proj_dist = np.abs(np.einsum('ij,ij->i', x_centered, wp)) / r_wSw
            dist[idx] = proj_dist
        dist_df = pd.DataFrame(dist)

        # --- Part 2: Vectorized Prediction Logic ---

        # 1. Get sorted distances and classes of the 2 nearest neurons
        dist_np = dist_df.to_numpy()
        sort_idx = np.argsort(dist_np, axis=1)[:, :2]
        sort_dist = np.take_along_axis(dist_np, sort_idx, axis=1)

        y_values = self.neuron_list['y'].to_numpy()
        class1 = y_values[sort_idx[:, 0]]
        class2 = y_values[sort_idx[:, 1]]

        # 2. Initialize prediction array
        y_predict = np.empty(X.shape[0], dtype=class1.dtype)

        # 3. Apply classification logic using boolean masks
        # Condition 1: The two nearest neurons belong to the same class.
        mask_same_class = (class1 == class2)
        y_predict[mask_same_class] = class1[mask_same_class]

        # Condition 2: The point is clearly inside the first neuron's boundary.
        mask_clear_win = (sort_dist[:, 0] <= 1) & (sort_dist[:, 1] > 1)
        y_predict[mask_clear_win & ~mask_same_class] = class1[mask_clear_win & ~mask_same_class]

        # Condition 3: Ambiguous cases that need the discriminant vector.
        mask_ambiguous = ~mask_same_class & ~mask_clear_win

        # Only proceed if there are ambiguous points to classify.
        if np.any(mask_ambiguous):
            # Efficiently gather data for only the ambiguous points
            X_ambiguous = X[mask_ambiguous]
            neuron1_idx = sort_idx[mask_ambiguous, 0]
            neuron2_idx = sort_idx[mask_ambiguous, 1]

            centers = np.stack(self.neuron_list['center'].values)
            covs = np.stack(self.neuron_list['cov'].values)

            c1, S1 = centers[neuron1_idx], covs[neuron1_idx]
            c2, S2 = centers[neuron2_idx], covs[neuron2_idx]

            # Calculate discriminant vectors and new distances for all ambiguous points at once
            w = self._vectorized_discriminant_vector(c1, S1, c2, S2)
            dist_1_on_w = self._vectorized_projection_distance(X_ambiguous, c1, S1, w)
            dist_2_on_w = self._vectorized_projection_distance(X_ambiguous, c2, S2, w)

            # Get corresponding classes and predict using np.where
            class1_ambiguous = class1[mask_ambiguous]
            class2_ambiguous = class2[mask_ambiguous]
            predictions_ambiguous = np.where(dist_1_on_w <= dist_2_on_w, class1_ambiguous, class2_ambiguous)

            # Place the results back into the main prediction array
            y_predict[mask_ambiguous] = predictions_ambiguous

        return y_predict
