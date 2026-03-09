import numpy as np
import numpy.linalg as LA
import pandas as pd

from ._base import VersatileEllipticBaseClassifier, _SQRT_2PI


class SCIL(VersatileEllipticBaseClassifier):
    def __init__(self, N0=3, eta=2, delta=1, epsilon=1e-10, theta=0):
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.delta = delta
        self.N0 = N0
        self.eta = eta
        self.theta = theta
        self.epsilon = epsilon

    def create_new_neuron(self, X, y):
        select_index = 0
        cen = X[select_index, :]
        cov = np.zeros([len(cen)]*2)
        eig_c = np.identity(len(cen))
        pca_var = np.ones(len(cen))
        width = self.init_width[y]
        n = 1
        neuron = {'y':y,'cov':cov,'center':cen,'eig_component':eig_c,'variance':pca_var,'width':width,'n':n}
        X_ = np.delete(X, select_index, axis=0)
        return X_, neuron

    def select_update_data(self, X, neuron):
        center = neuron["center"]
        eig_c = neuron["eig_component"]
        width = neuron['width'] + self.epsilon
        n = neuron["n"]

        center_temp = (n * center + X) / (n + 1)
        x_centered = X - center_temp
        P_d_x = np.tensordot(x_centered, eig_c, axes=(1,1))
        Psi = LA.norm(P_d_x/width,ord=2,axis=1)**2 - 1

        Y_index = np.where(Psi <= 0)[0]
        Y = X[Y_index]

        return Y, Y_index

    def update_parameter(self, neuron, alpha, X, Y, Y_index):
        """
        Updates the parameters of a neuron based on new data.

        Args:
            neuron: A dictionary containing the neuron's parameters.
            alpha: The index of the neuron to update.
            X: The original data array.
            Y: The new data points assigned to the neuron.
            Y_index: The indices of the new data points in X.

        Returns:
            The updated data array X with the assigned data points removed.
        """

        cen_alpha = neuron["center"]
        cov_alpha = neuron["cov"]
        n_alpha = neuron["n"]
        width_alpha = neuron["width"]
        n_Y = len(Y)
        cen_Y = np.mean(Y, axis=0)
        n_new = n_alpha + n_Y
        cen_new = (n_alpha * cen_alpha + n_Y * cen_Y) / n_new

        Y_sum = np.sum(Y[:, :, np.newaxis] * Y[:, np.newaxis, :], axis=0)  # Vectorized sum of outer products
        cov_new = (
            n_alpha * (cov_alpha + np.outer(cen_alpha, cen_alpha)) / n_new
            + Y_sum / n_new
            - np.outer(cen_new, cen_new)
        )

        eig_c_new, pca_var_new = self.compute_sorted_eigencomponent(cov_new)
        width_new = np.array([width_alpha[d]+np.abs(np.matmul(cen_new-cen_Y,eig_c_new[d].T)) for d in range(len(width_alpha))])
        max_psi = np.max([self.hyperellipsoidal_fn(y,cen_new,eig_c_new,width_new) for y in Y])
        if max_psi > 0:
            width_new = np.sqrt(1+self.eta*max_psi)*width_new
        # Efficiently remove assigned data points from X
        X_new = np.delete(X, Y_index, axis=0)

        # Update neuron parameters in the neuron list
        self.neuron_list.at[alpha, 'center'] = cen_new
        self.neuron_list.at[alpha, 'eig_component'] = eig_c_new
        self.neuron_list.at[alpha, 'n'] = n_new
        self.neuron_list.at[alpha, 'cov'] = cov_new
        self.neuron_list.at[alpha, 'variance'] = pca_var_new
        self.neuron_list.at[alpha, 'width'] = width_new

        return X_new

    def create_and_update(self, X, y):
        """
        Creates a new neuron, updates its parameters, and potentially merges it with existing neurons.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: The class label for the new neuron.

        Returns:
            The updated data array X with assigned data points removed.
        """

        # Create a new neuron and add it to the neuron list
        X, neuron = self.create_new_neuron(X, y)
        self.neuron_list = pd.concat([self.neuron_list, pd.DataFrame([neuron])], ignore_index=True)
        alpha = self.neuron_list.index[-1]

        # Select data points and update neuron parameters
        Y, Y_index = self.select_update_data(X, neuron)
        if len(Y) != 0:
            X = self.update_parameter(neuron, alpha, X, Y, Y_index)

        return X

    def find_and_update(self, X, y):
        """
        Finds the closest neuron to the mean of the data and updates its parameters.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: The class label of the neurons to consider.

        Returns:
            The updated data array X with assigned data points removed.
        """

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
                self.merge_neuron(alpha, y)
            else:
                break  # No data points assigned, exit the loop

        return X

    def find_and_capture(self, X, y):
        """
        Creates new neurons and updates their parameters until all data points are captured.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: The class label of the neurons to create.

        Returns:
            The updated data array X (which should be empty if all points are captured).
        """

        while len(X) != 0:
            # Create a new neuron and add it to the neuron list
            X, neuron = self.create_new_neuron(X, y)
            self.neuron_list = pd.concat([self.neuron_list, pd.DataFrame([neuron])], ignore_index=True)
            alpha = self.neuron_list.index[-1]

            # Select data points and update neuron parameters
            Y, Y_index = self.select_update_data(X, neuron)
            if len(Y) != 0:
                X = self.update_parameter(neuron, alpha, X, Y, Y_index)

        return X

    def merge_neuron(self, alpha, y):
        neuron_list_y = self.neuron_list.query(f'y=={y}')
        if len(neuron_list_y) > 1:
            cov_alpha = self.neuron_list.at[alpha,'cov']
            cen_alpha = self.neuron_list.at[alpha,'center']
            n_alpha = self.neuron_list.at[alpha,'n']
            width_alpha = self.neuron_list.at[alpha,'width']
            eig_c_alpha = self.neuron_list.at[alpha,'eig_component']
            for beta in neuron_list_y.index:
                cov_beta = self.neuron_list.at[beta,'cov']
                cen_beta = self.neuron_list.at[beta,'center']
                n_beta = self.neuron_list.at[beta,'n']
                width_beta = self.neuron_list.at[beta,'width']
                eig_c_beta = self.neuron_list.at[beta,'eig_component']
                if alpha != beta:
                    psi_alpha = self.hyperellipsoidal_fn(cen_alpha, cen_beta, eig_c_beta, width_beta)
                    psi_beta = self.hyperellipsoidal_fn(cen_beta, cen_alpha, eig_c_alpha, width_alpha)
                    if psi_alpha <= self.theta or psi_beta <= self.theta:
                        n_gamma = n_alpha + n_beta
                        cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                        cov_gamma = self._merge_covariance(n_alpha, cov_alpha, cen_alpha, n_beta, cov_beta, cen_beta)
                        eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)

                        width_gamma = np.array([1.96*np.sqrt(np.abs(pca_var_gamma[d])/n_gamma) for d in range(len(pca_var_gamma))])  # 1.96 = z-score for 95% CI
                        self.neuron_list.at[beta,'n'] = n_gamma
                        self.neuron_list.at[beta,'center'] = cen_gamma
                        self.neuron_list.at[beta,'cov'] = cov_gamma
                        self.neuron_list.at[beta,'eig_component'] = eig_c_gamma
                        self.neuron_list.at[beta,'width'] = width_gamma
                        self.neuron_list.at[beta,'variance'] = pca_var_gamma

                        self.neuron_list = self.neuron_list.drop(alpha,axis=0)
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

    def predict(self,X):
        neuron_list_test = self.neuron_list.query('n >= @self.N0').copy()
        for idx in neuron_list_test.index:
            variance = neuron_list_test.at[idx,'variance']
            neuron_list_test.at[idx,'width'] = _SQRT_2PI * np.sqrt(variance)

        distance = {}
        for idx, neuron in enumerate(neuron_list_test.to_dict('records')):
            center = neuron['center']
            eig_c = neuron['eig_component'].real
            width = np.array(neuron['width']).reshape(len(center))
            x_centered = X - center
            P_d_x = np.tensordot(x_centered, eig_c, axes=(1,1))
            Psi = LA.norm(P_d_x/width,ord=2,axis=1)**2 - 1
            distance[idx] = Psi
        distance_df = pd.DataFrame(distance)
        y_pred = neuron_list_test['y'].iloc[distance_df.idxmin(axis=1).values].values
        return y_pred
