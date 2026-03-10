import copy
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
import numpy.linalg as LA

def get_axis_edge_points(center, axes, widths):
    """Generates edge points for each axis."""
    edge_points = []
    for i in range(len(axes)):
        edge_points.append([center - widths[i] * axes[i],center + widths[i] * axes[i]])
    return np.array(edge_points)

def get_axis_sample_points(center, axes, widths, num_samples=11):
    """
    Generates a specified number of sample points along each principal axis.

    This function is used to create a set of points for robustly checking
    if an axis of one hyper-ellipsoid intersects with another.

    Args:
        center (np.ndarray): The center vector of the hyper-ellipsoid.
        axes (np.ndarray): A 2D array where each row is an eigenvector (principal axis).
        widths (np.ndarray): A 1D array of the semi-axis lengths (r * sqrt(eigenvalue)).
        num_samples (int, optional): The number of points to sample along each axis.
                                    Defaults to 11. An odd number is recommended
                                    to include the center point.

    Returns:
        list: A list of numpy arrays. Each array contains the `num_samples`
                points sampled along the corresponding principal axis.
    """
    all_axis_points = []
    # Ensure axes and widths are iterable
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    if not isinstance(widths, (list, np.ndarray)):
        widths = [widths]
        
    for i in range(len(axes)):
        # Define the start and end points of the axis segment
        start_point = center - widths[i] * axes[i]
        end_point = center + widths[i] * axes[i]

        # Use np.linspace to generate num_samples evenly spaced points
        # between the start and end points.
        # The 'axis=1' argument stacks the start and end points for broadcasting.
        axis_points = np.linspace(start_point, end_point, num_samples)
        
        all_axis_points.append(axis_points)
        
    return all_axis_points

class HyperellipsoidBaseClassifier(BaseEstimator):
    def compute_sorted_eigencomponent(self, cov : np.ndarray):
        """
        Computes the sorted eigenvectors and eigenvalues of a covariance matrix.

        Args:
            cov: A numpy array representing the covariance matrix.

        Returns:
            A tuple containing:
                - eig_c: A numpy array of sorted eigenvectors (principal components).
                - pca_var: A numpy array of sorted eigenvalues (variances).
        """
        
        eig_value, eig_vector = LA.eig(cov)
        sort_idx = eig_value.argsort()[::-1]
        pca_var = eig_value[sort_idx]
        eig_c = eig_vector[:,sort_idx].T

        # Find the first non-positive eigenvalue
        first_zero_index =  np.searchsorted(-pca_var,0)

        # Replace non-positive eigenvalues with the lower bound
        lowest_lambda = min(pca_var[first_zero_index-1]/2,self.epsilon)
        pca_var[first_zero_index:] = lowest_lambda

        return eig_c.real, pca_var.real
    
    def check_neuron_class_exist(self, y):
        """
        Checks if a neuron with the given class label 'y' exists in the neuron list.

        Args:
            y: The class label.

        Returns:
            True if a neuron with class 'y' exists, False otherwise.
        """

        return y in self.neuron_list['y'].values if len(self.neuron_list) > 0 else False
    
    def set_classes(self):
        self.classes_ = np.unique(self.neuron_list["y"])
    
class VersatileEllipticBaseClassifier(HyperellipsoidBaseClassifier):
    
    def average_pairwise_distance(self, X):
        n, d = X.shape
        return np.array([self.delta/(n**2)*sum(sp.spatial.distance.pdist(X))*2]*d)
    
    def hyperellipsoidal_fn(self, x, center, eig_c, width):
        x_centered = x - center
        Margin = width + self.epsilon
        P_d_x = x_centered @ eig_c.T
        return (LA.norm(P_d_x/Margin,ord=2))**2-1
    
    def create_new_neuron(self, x_i, y_i):
        cen = x_i
        cov = np.identity(len(x_i))
        width = self.init_width[y_i]
        n = 1
        eig_c = np.identity(len(x_i))
        neuron = {'y':y_i,'cov':cov,'center':cen,'eig_component':eig_c,'width':width,'n':n}
        return neuron
    
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
                        cov_gamma = (
                            1 / n_gamma
                            * (
                                n_alpha * cov_alpha
                                + n_beta * cov_beta
                                + (n_alpha * n_beta) / n_gamma * np.outer(cen_alpha - cen_beta, cen_alpha - cen_beta)
                            )
                        )
                        eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)
                    
                        width_gamma = np.array([np.sqrt(2*np.pi*np.abs(pca_var_gamma[d])) for d in range(len(pca_var_gamma))])
                        self.neuron_list.at[beta,'n'] = n_gamma
                        self.neuron_list.at[beta,'center'] = cen_gamma
                        self.neuron_list.at[beta,'cov'] = cov_gamma
                        self.neuron_list.at[beta,'eig_component'] = eig_c_gamma
                        self.neuron_list.at[beta,'width'] = width_gamma
                        
                        self.neuron_list = self.neuron_list.drop(alpha,axis=0)
                        break

class PrincipleProjectionBaseClassifier(BaseEstimator):
    def angle_between_unit_vectors(self, v1, v2):
        dot_product = np.einsum('ij,kj->ik', v1, v2)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        return np.minimum(angle, np.pi - angle) * 180 / np.pi
    
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
                if len(result) == self.max_d:
                    break
        if len(result) < self.max_d:
            all_rows = set(range(rows))
            all_cols = set(range(cols))
            remaining_rows = sorted(all_rows - used_rows, reverse=True)
            remaining_cols = sorted(all_cols - used_cols, reverse=True)
            remaining_pairs = list(zip(remaining_rows, remaining_cols))
            result.extend(remaining_pairs[:self.max_d - len(result)])

        return np.array(result)
    
    def calculate_proj_dist(self, x_centered, P, M, norm):
        """
            Calculates the projected distance between a data point and a neuron

        Args:
            x_centered: A numpy array of shape (n_samples, n_features) representing the centered data points.
            P: A numpy array representing the projection matrix.
            M: A numpy array representing the margin or width of hyperellipsoids.
            norm: The order of the norm to use for distance calculation.

        Returns:
        A numpy array of distances between the data points and the hyperellipsoids.
        """

        P_d_x = np.tensordot(x_centered, P, axes=(1, 1))  # Project data points
        return (LA.norm(P_d_x / M, ord=norm, axis=1)) - self.r  # Calculate distance
    
    def predict_with_eigen_proj(self, neuron_list_test, X, argsort_dist):
        """
        Predicts class labels for data points using eigenprojection and distance calculations.

        Args:
            neuron_list_test: A pandas DataFrame containing neuron data.
            X: A numpy array of shape (n_samples, n_features) representing the data.
            argsort_dist: A numpy array of shape (n_samples, 2) containing the indices of the two
                            closest neurons for each data point.

        Returns:
            A numpy array of predicted class labels.
        """

        first_index, second_index = argsort_dist[:, 0], argsort_dist[:, 1]
        top_two_index = np.array([sorted(idx) for idx in zip(first_index, second_index)])
        unique_top_two = list(set(map(tuple, top_two_index)))
        y_pred = np.full(len(X), np.nan)
        
        for unique_pair in unique_top_two:
            mask = (top_two_index == unique_pair).all(axis=1)
            if neuron_list_test.iloc[unique_pair[0]]["y"] == neuron_list_test.iloc[unique_pair[1]]["y"]:
                y_pred[mask] = neuron_list_test.iloc[unique_pair[0]]["y"]
            else:
                x_unique_pair = X[mask]
                
                centers, eig_cs, widths = zip(*[self.get_neuron_data(neuron_list_test, idx) for idx in unique_pair])
                
                eig_angle = self.angle_between_unit_vectors(eig_cs[0], eig_cs[1])
                top_index = self.find_index_pairs(eig_angle)
                
                Ps = [eig_c[top_index[:, i]] for i, eig_c in enumerate(eig_cs)]
                ws = [width[top_index[:, i]] for i, width in enumerate(widths)]
                Ms = [w + self.epsilon for w in ws]
                
                x_centereds = [x_unique_pair - center for center in centers]
                
                x_proj_dist = {
                    y: self.calculate_proj_dist(x_centered, P, M, self.norm)
                    for y, x_centered, P, M in zip(unique_pair, x_centereds, Ps, Ms)
                }
                
                x_proj_dist_df = pd.DataFrame(x_proj_dist)
                y_predict_unique_pair = neuron_list_test['y'].iloc[x_proj_dist_df.idxmin(axis=1).tolist()]
                
                y_pred[mask] = y_predict_unique_pair
        
        return y_pred


class LRHE(VersatileEllipticBaseClassifier):
    def __init__(self, alpha=0.5, theta=0, delta=1, epsilon=1e-10): 
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.delta = delta
        self.alpha = alpha
        self.theta = theta
        self.epsilon = epsilon
    
    def width_init(self, X, y):
        """
        Initializes width for new classes.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]  
            self.init_width[y_] = self.average_pairwise_distance(Xy)
    
    def fit(self,X,y,**kwargs):
        self.width_init(X,y)
        for x_i,y_i in zip(X,y):
            if self.check_neuron_class_exist(y_i):
                neuron_list_y = self.neuron_list.query(f'y=={y_i}')
                xi = neuron_list_y.index[np.argmin([LA.norm(x_i-neuron_list_y['center'][idx]) for idx in neuron_list_y.index])]
                cov_xi = self.neuron_list.at[xi,'cov']
                cen_xi = self.neuron_list.at[xi,'center']
                n_xi = self.neuron_list.at[xi,'n']
                width_xi = self.neuron_list.at[xi,'width']
                eig_c_xi = self.neuron_list.at[xi,'eig_component']
                psi_xi = self.hyperellipsoidal_fn(x_i,cen_xi,eig_c_xi,width_xi)
                
                self.shift_and_shrink_neuron(x_i, y_i, psi_xi)
                
                cen_temp = (n_xi*cen_xi+x_i)/(n_xi+1)
                psi_temp = self.hyperellipsoidal_fn(x_i,cen_temp,eig_c_xi,width_xi)
                
                if psi_temp > 0: 
                    # Create new neuron
                    neuron = self.create_new_neuron(x_i, y_i)
                    self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                    alpha = self.neuron_list.index[-1]

                else:
                    # Update
                    n_temp = n_xi+1
                    cov_temp = (n_xi*cov_xi+np.matmul(np.array([x_i]).T,[x_i])-np.matmul(np.array([cen_xi]).T,[cen_xi]))/(n_xi+1)- np.matmul(np.array([cen_temp]).T,[cen_temp])+np.matmul(np.array([cen_xi]).T,[cen_xi])
                    eig_c_temp, pca_var_temp = self.compute_sorted_eigencomponent(cov_temp)
                    width_temp = np.array([width_xi[d]+np.abs(np.matmul(cen_temp-cen_xi,eig_c_temp[d].T)) for d in range(len(width_xi))])
                    self.neuron_list.at[xi,'cov'] = cov_temp
                    self.neuron_list.at[xi,'width'] = width_temp
                    self.neuron_list.at[xi,'center'] = cen_temp
                    self.neuron_list.at[xi,'n'] = n_temp
                    self.neuron_list.at[xi,'eig_component'] = eig_c_temp
                    alpha = xi
                    
                self.merge_neuron(alpha, y_i)
                
            else:
                neuron = self.create_new_neuron(x_i, y_i)
                self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                
        self.set_classes()

    def partial_fit(self,X,y,**kwargs):
        self.fit(X,y)
        
    def predict(self,X):
        ## New prediction function by compute multiple data at once, with the time complexity O(k) where k is the number of neurons
        distance = {}
        for idx, neuron in enumerate(self.neuron_list.to_dict('records')):
            center = neuron['center']
            eig_c = neuron['eig_component'].real
            width = np.array(neuron['width']).reshape(len(center))
            x_centered = X - center
            P_d_x = np.tensordot(x_centered, eig_c, axes=(1,1))
            Psi = LA.norm(P_d_x/width,ord=2,axis=1)**2 - 1
            distance[idx] = Psi
        distance_df = pd.DataFrame(distance)
        y_pred = self.neuron_list['y'].iloc[distance_df.idxmin(axis=1).values].values
        return y_pred 

    def shift_and_shrink_neuron(self, x_i, y_i, psi_xi):
        neuron_list_not_y = self.neuron_list.query(f'y!={y_i}')
        for idx in neuron_list_not_y.index:
            cen_idx = self.neuron_list.at[idx,'center']
            n_idx = self.neuron_list.at[idx,'n']
            width_idx = self.neuron_list.at[idx,'width']
            eig_c_idx = self.neuron_list.at[idx,'eig_component']
            psi_idx = self.hyperellipsoidal_fn(x_i,cen_idx,eig_c_idx,width_idx)
            if psi_idx <= 0 and psi_idx <= psi_xi:
                new_width = np.array([max([(n_idx*width_idx[d]+np.matmul(x_i-cen_idx,eig_c_idx[d].T))/(n_idx+1),self.alpha*width_idx[d]]) for d in range(len(width_idx))])
                new_cen = cen_idx-1/n_idx*(x_i-cen_idx)
                self.neuron_list.at[idx,'width'] = new_width
                self.neuron_list.at[idx,'center'] = new_cen
                

class VEBF(VersatileEllipticBaseClassifier):
    def __init__(self, theta=0, delta=1, epsilon=1e-10): 
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.delta = delta
        self.theta = theta
        self.epsilon = epsilon
        
    def width_init(self, X, y):
        """
        Initializes width for new classes.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class
        if len(new_class) > 0:
            average_distance = self.average_pairwise_distance(X)
            for y_ in new_class:
                self.init_width[y_] = average_distance
        
    def fit(self,X,y,**kwargs):
        self.width_init(X,y)
        for x_i,y_i in zip(X,y):
            if self.check_neuron_class_exist(y_i):
                neuron_list_y = self.neuron_list.query(f'y=={y_i}')
                xi = neuron_list_y.index[np.argmin([LA.norm(x_i-neuron_list_y['center'][idx]) for idx in neuron_list_y.index])]
                cov_xi = self.neuron_list.at[xi,'cov']
                cen_xi = self.neuron_list.at[xi,'center']
                n_xi = self.neuron_list.at[xi,'n']
                width_xi = self.neuron_list.at[xi,'width']
                eig_c_xi = self.neuron_list.at[xi,'eig_component']
                
                cen_temp = (n_xi*cen_xi+x_i)/(n_xi+1)
                psi_temp = self.hyperellipsoidal_fn(x_i,cen_temp,eig_c_xi,width_xi)
                
                if psi_temp > 0: 
                    # Create new neuron
                    neuron = self.create_new_neuron(x_i, y_i)
                    self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                    alpha = self.neuron_list.index[-1]

                else:
                    # Update
                    n_temp = n_xi+1
                    cov_temp = (n_xi*cov_xi+np.matmul(np.array([x_i]).T,[x_i])-np.matmul(np.array([cen_xi]).T,[cen_xi]))/(n_xi+1)- np.matmul(np.array([cen_temp]).T,[cen_temp])+np.matmul(np.array([cen_xi]).T,[cen_xi])
                    eig_c_temp, pca_var_temp = self.compute_sorted_eigencomponent(cov_temp)
                    width_temp = np.array([width_xi[d]+np.abs(np.matmul(cen_temp-cen_xi,eig_c_temp[d].T)) for d in range(len(width_xi))])
                    self.neuron_list.at[xi,'cov'] = cov_temp
                    self.neuron_list.at[xi,'width'] = width_temp
                    self.neuron_list.at[xi,'center'] = cen_temp
                    self.neuron_list.at[xi,'n'] = n_temp
                    self.neuron_list.at[xi,'eig_component'] = eig_c_temp
                    alpha = xi
                    
                self.merge_neuron(alpha, y_i)
                
            else:
                neuron = self.create_new_neuron(x_i, y_i)
                self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                
        self.set_classes()
    
    def partial_fit(self,X,y,**kwargs):
        self.fit(X,y)
        
        
    def predict(self,X):
        ## New prediction function by compute multiple data at once, with the time complexity O(k) where k is the number of neurons
        distance = {}
        for idx, neuron in enumerate(self.neuron_list.to_dict('records')):
            center = neuron['center']
            eig_c = neuron['eig_component'].real
            width = np.array(neuron['width']).reshape(len(center))
            x_centered = X - center
            P_d_x = np.tensordot(x_centered, eig_c, axes=(1,1))
            Psi = LA.norm(P_d_x/width,ord=2,axis=1)**2 - 1
            distance[idx] = Psi
        distance_df = pd.DataFrame(distance)
        y_pred = self.neuron_list['y'].iloc[distance_df.idxmin(axis=1).values].values
        return y_pred 
    
                
class SCIL(VersatileEllipticBaseClassifier):
    def __init__(self, N0=3, eta=2, delta=1, epsilon=1e-10, theta=0): 
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.delta = delta
        self.N0 = N0
        self.eta = eta
        self.theta = theta
        self.epsilon = epsilon
        
    def width_init(self, X, y):
        """
        Initializes width for new classes.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

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
        width_new = np.array([width_alpha[d]+np.abs(np.matmul(cen_new-cen_alpha,eig_c_new[d].T)) for d in range(len(width_alpha))])
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
                        cov_gamma = (
                            1 / n_gamma
                            * (
                                n_alpha * cov_alpha
                                + n_beta * cov_beta
                                + (n_alpha * n_beta) / n_gamma * np.outer(cen_alpha - cen_beta, cen_alpha - cen_beta)
                            )
                        )
                        eig_c_gamma, pca_var_gamma = self.compute_sorted_eigencomponent(cov_gamma)
                    
                        width_gamma = np.array([1.96*np.sqrt(np.abs(pca_var_gamma[d])/n_gamma) for d in range(len(pca_var_gamma))])
                        self.neuron_list.at[beta,'n'] = n_gamma
                        self.neuron_list.at[beta,'center'] = cen_gamma
                        self.neuron_list.at[beta,'cov'] = cov_gamma
                        self.neuron_list.at[beta,'eig_component'] = eig_c_gamma
                        self.neuron_list.at[beta,'width'] = width_gamma
                        self.neuron_list.at[beta,'variance'] = pca_var_gamma
                        
                        self.neuron_list = self.neuron_list.drop(alpha,axis=0)
                        break
        
    def fit(self,X,y,**kwargs):
        all_class = np.unique(y)
        self.width_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]  
            if not self.check_neuron_class_exist(y_):
                Xy = self.create_and_update(Xy, y_)
            Xy = self.find_and_update(Xy, y_)
            Xy = self.find_and_capture(Xy, y_)
        self.set_classes()
        
    def partial_fit(self,X,y,**kwargs):
        self.fit(X, y)
        
    def predict(self,X):
        neuron_list_test = self.neuron_list.query('n >= @self.N0').copy()
        for idx in neuron_list_test.index:
            variance = neuron_list_test.at[idx,'variance']
            neuron_list_test.at[idx,'width'] = np.sqrt(2*np.pi*variance)

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
    
class ScalableHyperelipsoidBaseClassifier(HyperellipsoidBaseClassifier):

    def find_median_dist_to_neighbor(self, X):
        """
        Calculates the median distance between each point in X and its nearest neighbor.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.

        Returns:
            The median distance to the nearest neighbor.
        """
        
        if len(X) == 1:
            return np.sqrt(np.e)
        else:
            neigh1 = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X).kneighbors()[0]
            median_dist = np.median(neigh1)
            if median_dist > self.epsilon:
                return median_dist
            else:
                return self.epsilon
        
    def find_mean_dist_to_neighbor(self, X):
        """
        Calculates the mean distance between each point in X and its nearest neighbor.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.

        Returns:
            The mean distance to the nearest neighbor.
        """
        
        if len(X) == 1:
            return np.sqrt(np.e)
        else:
            neigh1 = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X).kneighbors()[0]
            return np.mean(neigh1)
        
    def dist_ths_y_update(self, y):
        """
        Updates the distance threshold for a given class label 'y'.

        Args:
            y: The class label.
        """

        ths = self.dist_ths[y]
        S = self.neuron_list

        # Count neurons with 'n' < self.M and 'y' == y
        m = np.sum((S['n'] < self.M) & (S['y'] == y))

        # Check condition and update threshold
        if m > len(S[S['y'] == y]) / 2:
            self.dist_ths[y] = ths * 2
            
    def discriminant_vector_between_two_shef(self,cen_1,cov_1,cen_2,cov_2):
        cov = cov_1+cov_2
        discriminant = np.matmul(LA.inv(cov),np.array([cen_1-cen_2]).T)
        return discriminant/LA.norm(discriminant)
    
    def projection_ration_distance(self,x,center,cov,wp):
        return np.abs(np.matmul(wp.T,np.array([x-center]).T))/(self.r*np.sqrt(np.matmul(wp.T,np.matmul(cov,wp))))
    
    def Mahalanobis_dist(x,c,S):
        return np.sqrt(np.matmul(np.array([x-c]),np.matmul(LA.inv(S),np.array([x-c]).T)))
        
class SHEF(ScalableHyperelipsoidBaseClassifier):
    def __init__(self, M = 3,r = 1.5, epsilon=1e-10): 
        self.neuron_list = pd.DataFrame([])
        self.dist_ths = {}
        self.M = M
        self.r = r
        self.epsilon = epsilon
        
    def distance_init(self, X, y):
        """
        Initializes distance thresholds for new classes.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.dist_ths.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]  
            initial_dist = self.find_median_dist_to_neighbor(Xy)
            self.dist_ths[y_] = initial_dist
            
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
            # print("alpha", alpha, "beta", beta)
            
            # Precompute some matrices for efficiency
            cov_tlide_inv = LA.inv(cov_beta) / (self.r ** 2)
            cov_tlide = cov_alpha / (self.r ** 2)
            F = (-cen_alpha + cen_beta) @ cov_tlide_inv
            D = cov_tlide @ cov_tlide_inv + np.outer(cen_alpha, F)

            # Construct P matrix using more efficient methods
            P_upper = np.hstack([D, -D @ cen_beta[:, np.newaxis] + cen_alpha[:, np.newaxis]])
            P_lower = np.hstack([F, -F @ cen_beta[:, np.newaxis] + 1])
            P = np.vstack([P_upper, P_lower])
            
            eig_P = LA.eig(P)[0]

            # Check merge conditions
            if all(np.isreal(eig_P)) and len(np.unique(eig_P)) == len(eig_P) and any(np.less(eig_P, 0)):
                pass  # Conditions not met, continue to next neuron
            else:
                # print("Merge")
                # Gather neuron parameters
                n_beta = self.neuron_list.at[beta, 'n']

                # Calculate merged parameters
                n_gamma = n_alpha + n_beta
                cen_gamma = (n_alpha * cen_alpha + n_beta * cen_beta) / n_gamma
                cov_gamma = (
                    1 / n_gamma
                    * (
                        n_alpha * cov_alpha
                        + n_beta * cov_beta
                        + (n_alpha * n_beta) / n_gamma * np.outer(cen_alpha - cen_beta, cen_alpha - cen_beta)
                    )
                )

                # Update neuron beta with merged parameters
                self.neuron_list.at[beta, 'n'] = n_gamma
                self.neuron_list.at[beta, 'center'] = cen_gamma
                self.neuron_list.at[beta, 'cov'] = cov_gamma

                # Remove neuron alpha
                self.neuron_list = self.neuron_list.drop(alpha, axis=0)

        
    def fit(self,X,y,**kwargs):
        self.distance_init(X, y)
        for x_i,y_i in zip(X, y):
            if self.check_neuron_class_exist(y_i):
                neuron_list_y = self.neuron_list.query('y==@y_i')
                # Finding the closest neuron
                alpha = neuron_list_y.index[np.argmin([LA.norm(x_i-neuron_list_y.at[idx,'center']) for idx in neuron_list_y.index])]
                # print(x_i, y_i)
                # print(alpha)
                # print("Dist",LA.norm(x_i-neuron_list_y.at[alpha,'center']))
                # print("Thersold", self.dist_ths[y_i])
                # print("-----")
                if LA.norm(x_i-neuron_list_y.at[alpha,'center']) > self.dist_ths[y_i]:
                    # Create new neuron
                    neuron = self.create_new_neuron(x_i, y_i)
                    self.neuron_list = pd.concat([self.neuron_list,pd.DataFrame([neuron])] ,ignore_index=True)
                    alpha = self.neuron_list.index[-1]
                    self.dist_ths_y_update(y_i)
                    # print(f"Create neuron of {y_i}, current threshold {self.dist_ths[y_i]}")
                else:
                    #if verbose > 0 :print("Datum is in SHEF number %d ,updating this SHEF" %neuron_list_y.index[xi])
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
                # print(f"Create new neuron of {y_i}, current threshold {self.dist_ths[y_i]}")
        
        self.set_classes()
                
    
    def partial_fit(self,X,y,**kwargs):
        self.fit(X,y)
            
    def predict_(self,X):
        ## New prediction function by compute multiple data at once, with the time complexity O(k) where k is the number of neurons
        if self.neuron_list.shape[0] == 1:
            return np.array([self.neuron_list['y']]*len(X))
        self.neuron_list['cov_inv'] = self.neuron_list['cov'].apply(lambda x: LA.inv(x))
        dist = {}
        for idx, neuron in enumerate(self.neuron_list.to_dict('records')):
            center = neuron['center']
            cov = neuron['cov']
            cov_inv = neuron['cov_inv']
            x_centered = X - center
            wp = np.tensordot(x_centered, cov_inv, axes = (1,1))
            wp_norm = LA.norm(wp,axis=1)[:,None]
            wp_norm[wp_norm<self.epsilon] = self.epsilon
            wp = wp/wp_norm
            r_wSw = self.r*np.sqrt(np.einsum('ij,ij->i',np.tensordot(wp,cov, axes=(1,1)),wp))
            r_wSw[r_wSw<self.epsilon] = self.epsilon
            proj_dist = np.abs(np.einsum('ij,ij->i', x_centered, wp))/r_wSw
            dist[idx] = proj_dist
        dist_df = pd.DataFrame(dist)
        sort_idx = np.argsort(dist_df.to_numpy(),axis=1)
        sort_dist = np.take_along_axis(dist_df.to_numpy(), sort_idx, axis=1)
        y_list = self.neuron_list['y'].tolist()
        sort_class1 = [y_list[i] for i in sort_idx[:,0]]
        sort_class2 = [y_list[i] for i in sort_idx[:,1]]
        y_predict = []
        for i in range(len(sort_class1)):
            if sort_class1[i] == sort_class2[i]:
                y_predict.append(sort_class1[i])
            elif sort_dist[i,0] <= 1 < sort_dist[i,1]:
                y_predict.append(sort_class1[i])
            else:
                x = X[i]
                w = self.discriminant_vector_between_two_shef(self.neuron_list.iloc[sort_idx[i,0]]['center'],self.neuron_list.iloc[sort_idx[i,0]]['cov'],self.neuron_list.iloc[sort_idx[i,1]]['center'],self.neuron_list.iloc[sort_idx[i,1]]['cov'])
                dist_1_on_w = self.projection_ration_distance(x,self.neuron_list.iloc[sort_idx[i,0]]['center'],self.neuron_list.iloc[sort_idx[i,0]]['cov'],w)
                dist_2_on_w = self.projection_ration_distance(x,self.neuron_list.iloc[sort_idx[i,1]]['center'],self.neuron_list.iloc[sort_idx[i,1]]['cov'],w)
                if dist_1_on_w <= dist_2_on_w:
                    y_predict.append(sort_class1[i])
                else:
                    y_predict.append(sort_class2[i])
        return np.array(y_predict)
    
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
    
class D4(VersatileEllipticBaseClassifier, PrincipleProjectionBaseClassifier):
    def __init__(self,norm=2,delta=1,alpha=1,max_d=None,epsilon=1e-10,r=1.5,threshold=15):
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.norm = norm
        self.delta = delta
        self.alpha = alpha
        self.max_d = max_d
        self.epsilon = epsilon
        self.threshold = threshold
        self.r = r
        
    def width_init(self, X, y):
        """
        Initializes width for new classes.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the data.
            y: A numpy array of shape (n_samples,) representing the class labels.
        """

        all_class = np.unique(y)
        exist_class = set(self.init_width.keys())
        new_class = set(all_class) - exist_class

        for y_ in new_class:
            Xy = X[y == y_]  
            self.init_width[y_] = self.average_pairwise_distance(Xy)
            
    def create_new_neuron(self, X, y):
        n, d = X.shape
        if n > 1:
            cen = X.mean(axis=0)
            cov = np.cov(X.T)
            eig_c, pca_var = self.compute_sorted_eigencomponent(cov)
            initial_width = self.init_width[y]
            width = np.sqrt(2*np.pi*np.abs(pca_var))*self.alpha + (initial_width)*(1-self.alpha)
        else:
            cen = X[0]
            cov = np.identity(d)
            eig_c = np.identity(d)
            pca_var = np.ones(d)
            width = self.init_width[y]
        neuron = {'y':y,'cov':cov,'center':cen,'eig_component':eig_c,'width':width,'variance':pca_var,'n':n}
        return neuron
        
    def fit(self,X,y,**kwargs):
        all_class = np.unique(y)
        self.width_init(X, y)
        for y_ in all_class:
            Xy = X[y==y_]
            if not self.check_neuron_class_exist(y_):
                neuron = self.create_new_neuron(Xy, y_)
                self.neuron_list = pd.concat([self.neuron_list, pd.DataFrame([neuron])], ignore_index=True)
            else:
                idx = self.neuron_list.query('y==@y_').index[0]
                new_n, d = Xy.shape
                if new_n > 1:
                    new_cen = Xy.mean(axis=0)
                    new_cov = np.cov(Xy.T)
                else:
                    new_cov = np.identity(d)
                    new_cen = Xy[0]
                old_cov = self.neuron_list.at[idx,'cov']
                old_cen = self.neuron_list.at[idx,'center']
                old_n = self.neuron_list.at[idx,'n']
                old_w = self.neuron_list.at[idx,'width']
                merge_n = new_n+old_n
                merge_cen = (new_n*new_cen+old_n*old_cen)/merge_n
                merge_cov = 1/merge_n*(new_n*new_cov+old_n*old_cov+(new_n*old_n)/merge_n*np.matmul(np.array([new_cen-old_cen]).T,np.array([new_cen-old_cen])))
                eig_c, pca_var = self.compute_sorted_eigencomponent(merge_cov)
                new_w = np.sqrt(2*np.pi*np.abs(pca_var))*self.alpha + (old_w + np.abs(np.matmul(eig_c,merge_cen-old_cen)))*(1-self.alpha)
                    
                self.neuron_list.at[idx,'n'] = merge_n
                self.neuron_list.at[idx,'center'] = merge_cen
                self.neuron_list.at[idx,'cov'] = merge_cov
                self.neuron_list.at[idx,'eig_component'] = eig_c
                self.neuron_list.at[idx,'variance'] = pca_var
                self.neuron_list.at[idx,'width'] = new_w
                
    def partial_fit(self,X,y,**kwargs):
        self.fit(X,y)
        
    def get_neuron_data(self, neuron_list, idx):
        neuron = neuron_list.iloc[idx]
        return neuron['center'], neuron['eig_component'].real, neuron['width']
    
    def predict(self,X):
        if self.max_d == None:
            self.max_d = len(X[0])
        x_proj_dist = {}
        for idx, neuron in enumerate(self.neuron_list.to_dict('records')):
            center, eig_c, w = self.get_neuron_data(self.neuron_list, idx)
            x_centered = X - center
            Margin = w + self.epsilon
            x_proj_dist[idx] = self.calculate_proj_dist(x_centered, eig_c, Margin, self.norm)
        x_proj_dist_df = pd.DataFrame(x_proj_dist)
        argsort_dist = np.argsort(x_proj_dist_df,axis=1)
        if argsort_dist.shape[1] > 1:
            y_pred = self.predict_with_eigen_proj(self.neuron_list, X,  argsort_dist)
        else:
            y_pred = self.neuron_list['y'].iloc[x_proj_dist_df.idxmin(axis=1).values].values
        return np.array(y_pred)
    
class TRACED(ScalableHyperelipsoidBaseClassifier, PrincipleProjectionBaseClassifier):
    def __init__(self, norm=2, epsilon=1e-10, method="overlap-outside", r=np.sqrt(2*np.pi), N0=3, delta=2, 
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
                    cov_tlide_alpha = eig_alpha @ np.diag(width_alpha**2) @ eig_alpha.T
                    cov_tlide_inv_beta = eig_beta @ np.diag(1 / (width_beta**2)) @ eig_beta.T
                    
                    F = (-cen_alpha + cen_beta) @ cov_tlide_inv_beta
                    D = cov_tlide_alpha  @ cov_tlide_inv_beta + np.outer(cen_alpha, F)
                    
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
                        cov_gamma = (
                            1 / n_gamma
                            * (
                                n_alpha * cov_alpha
                                + n_beta * cov_beta
                                + (n_alpha * n_beta) / n_gamma * np.outer(cen_alpha - cen_beta, cen_alpha - cen_beta)
                            )
                        )

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

    def fit(self, X, y, **kwargs):
        all_class = np.unique(y)
        self.distance_init(X, y)
        for y_ in all_class:
            Xy = X[y == y_]  
            if not self.check_neuron_class_exist(y_):
                Xy = self.create_and_update(Xy, y_)
            Xy = self.find_and_update(Xy, y_)
            Xy = self.find_and_capture(Xy, y_)
            
        self.set_classes()

    def partial_fit(self, X, y, verbose=0, **kwargs):
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
        # Count neurons with 'n' < self.M and 'y' == y
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
        
        # Corrected print statement to show the number of selected axes.
        # print(f"Original dims: {len(edge_proj0)} -> Selected dims: {len(non_overlap_index[0])}")
        
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
    
    def _find_non_overlapping_axes_raw(self, edge_proj0, edge_proj1):
        """
        Determines the indices of the most non-overlapping axes, ensuring both
        sets have the same number of dimensions.
        """
        # 1. Correctly count the number of available non-overlapping axes for each ellipsoid.
        non_overlap_count0 = np.sum(edge_proj0 >= -self.overlap_tolerance)
        non_overlap_count1 = np.sum(edge_proj1 >= -self.overlap_tolerance)
        
        indices0 = np.argsort(edge_proj0)[-non_overlap_count0:]
        indices1 = np.argsort(edge_proj1)[-non_overlap_count1:]
        
        non_overlap_index = [indices0, indices1]
        
        
        return non_overlap_index
    
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
                        # print(f"Total Overlap: {sum(mask)}")
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
                        
                        # neuron_a = neuron_list_test.iloc[unique_pair[0]]
                        # neuron_b = neuron_list_test.iloc[unique_pair[1]]
                        # y_predict, _ =  self.resolve_overlap_with_lda(x_unique_pair, neuron_a, neuron_b, self.epsilon)
                        # y_pred[mask] = y_predict
                    
            mask = (top_two_index == unique_pair).all(axis=1) & (first_values > 0) & (second_values > 0)
            if np.any(mask) and 'outside' in self.method.split("-"):
                if neuron_list_test.iloc[unique_pair[0]]["y"] != neuron_list_test.iloc[unique_pair[1]]["y"]:
                    # print(f"Total Outside: {sum(mask)}")
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