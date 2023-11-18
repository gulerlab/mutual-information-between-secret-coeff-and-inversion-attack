import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def create_probabilistic_classifier_dataset_gaussian(mean, cov, num_of_samples):
    # (x, y) ~ p(x, y) = 0
    # (x, y) ~ p(x) * p(y) = 1
    half_samples = int(num_of_samples / 2)
    joint_data = np.random.multivariate_normal(mean=mean, cov=cov, size=half_samples)

    joint_label = np.zeros(half_samples)

    marginal_data_x_idx = np.random.randint(0, joint_data.shape[0], joint_data.shape[0])
    marginal_data_y_idx = np.random.randint(0, joint_data.shape[0], joint_data.shape[0])

    marginal_data = np.empty(joint_data.shape)
    marginal_data[:, 0] = joint_data[marginal_data_x_idx, 0]
    marginal_data[:, 1] = joint_data[marginal_data_y_idx, 0]

    marginal_label = np.ones(half_samples)

    data = np.concatenate([joint_data, marginal_data])
    label = np.concatenate([joint_label, marginal_label])

    random_idx = np.random.permutation(num_of_samples)
    return data[random_idx], label[random_idx]


def create_test_dataset_gaussian(mean, cov, num_of_samples):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=num_of_samples)


def create_train_test_split(data, label, train_test_ratio=0.8):
    train_size = int(data.shape[0] * train_test_ratio)
    train_data, train_label = data[list(range(train_size))], label[list(range(train_size))]
    test_data, test_label = data[list(range(train_size, data.shape[0]))], label[list(range(train_size, data.shape[0]))]
    return train_data, train_label, test_data, test_label


def create_batched_tensors(data, label, batch_size):
    data, label = torch.from_numpy(data).to(torch.float32), torch.from_numpy(label).to(torch.long)
    return torch.split(data, batch_size), torch.split(label, batch_size)


def create_batched_tensors_train_test(train_data, train_label, test_data, test_label, train_batch_size,
                                      test_batch_size):
    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label)
    test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)
    return (torch.split(train_data, train_batch_size), torch.split(train_label, train_batch_size),
            torch.split(test_data, test_batch_size), torch.split(test_label, test_batch_size))


######### KNN SAMPLING DATASET ##################
def create_knn_sampling_joint_cond_marginal_dataset(dataset, number_of_neighbors, x_idx, y_idx, z_idx):
    # joint: 0, conditional marginal: 1
    # conditional marginal dataset construction
    ## select N / kNN samples to train kNN classifier
    number_of_samples = dataset.shape[0]
    x_data = dataset[:, x_idx]
    y_data = dataset[:, y_idx]
    z_data = dataset[:, z_idx]

    if len(x_data.shape) < 2:
        x_data = x_data.reshape(-1, 1)

    if len(y_data.shape) < 2:
        y_data = y_data.reshape(-1, 1)

    if len(z_data.shape) < 2:
        z_data = z_data.reshape(-1, 1)

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

    ## concat data
    cond_marginal_data = np.concatenate([x_data[knn_x_idx, :], y_data[selected_samples, :],
                                        z_data[selected_samples, :]], axis=1)

    ## cond marginal data label
    cond_marginal_label = np.ones(cond_marginal_data.shape[0])

    # joint dataset construction
    joint_data = dataset

    ## joint data label
    joint_label = np.zeros(joint_data.shape[0])

    return joint_data, joint_label, cond_marginal_data, cond_marginal_label
