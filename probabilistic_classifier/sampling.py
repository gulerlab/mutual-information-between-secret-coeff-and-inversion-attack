import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_sampling(number_of_samples, number_of_neighbors, z_data):
    ## train and rest idx
    selected_samples = np.random.choice(number_of_samples, int(number_of_samples // number_of_neighbors), replace=False)
    rest_of_samples = np.arange(number_of_samples, dtype=int)
    rest_of_samples_mask = [x for x in range(number_of_samples) if x not in selected_samples]
    rest_of_samples = rest_of_samples[rest_of_samples_mask]

    ## train knn model
    knn_model = NearestNeighbors(n_neighbors=number_of_neighbors, metric='euclidean')
    knn_model.fit(z_data[rest_of_samples, :])

    ## get indices based on selected z
    knn_x_idx = knn_model.kneighbors(z_data[selected_samples, :], return_distance=False)
    knn_x_idx = np.concatenate(knn_x_idx).reshape(-1)

    ## adjust the idx set for selected data
    selected_samples = np.stack([selected_samples for _ in range(number_of_neighbors)]).T
    selected_samples = selected_samples.reshape(-1)

    return selected_samples, knn_x_idx


def marginal_sampling(number_of_samples):
    x_idx = np.random.permutation(np.arange(number_of_samples))
    y_idx = np.random.permutation(np.arange(number_of_samples))
    return x_idx, y_idx


def multiclass_conditional_sampling(number_of_samples):
    all_marginal_idx_x = np.random.permutation(np.arange(number_of_samples))
    all_marginal_idx_y = np.random.permutation(np.arange(number_of_samples))
    all_marginal_idx_z = np.random.permutation(np.arange(number_of_samples))

    marginal_y_joint_xz_y = np.random.permutation(np.arange(number_of_samples))
    marginal_y_joint_xz_xz = np.random.permutation(np.arange(number_of_samples))

    marginal_x_joint_yz_x = np.random.permutation(np.arange(number_of_samples))
    marginal_x_joint_yz_yz = np.random.permutation(np.arange(number_of_samples))

    return ((all_marginal_idx_x, all_marginal_idx_y, all_marginal_idx_z),
            (marginal_y_joint_xz_y, marginal_y_joint_xz_xz), (marginal_x_joint_yz_x, marginal_x_joint_yz_yz))
