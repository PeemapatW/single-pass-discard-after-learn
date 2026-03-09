import numpy as np
import numpy.linalg as LA
import pandas as pd

from ._base import VersatileEllipticBaseClassifier


class LRHE(VersatileEllipticBaseClassifier):
    def __init__(self, alpha=0.5, theta=0, delta=1, epsilon=1e-10):
        self.neuron_list = pd.DataFrame([])
        self.init_width = {}
        self.delta = delta
        self.alpha = alpha
        self.theta = theta
        self.epsilon = epsilon

    def fit(self, X, y, classes=None):
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

    def partial_fit(self, X, y, classes=None):
        self.fit(X, y)

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
