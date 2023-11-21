import numpy as np
import torch
from probabilistic_classifier.sampling import knn_sampling, marginal_sampling, multiclass_conditional_sampling


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

    selected_samples, knn_x_idx = knn_sampling(number_of_samples, number_of_neighbors, z_data)

    ## concat data
    cond_marginal_data = np.concatenate([x_data[knn_x_idx, :], y_data[selected_samples, :],
                                        z_data[selected_samples, :]], axis=1)

    ## cond marginal data label
    cond_marginal_label = np.ones(cond_marginal_data.shape[0])

    # joint dataset construction
    joint_data = np.concatenate([x_data, y_data, z_data], axis=1)

    ## joint data label
    joint_label = np.zeros(joint_data.shape[0])

    return joint_data, joint_label, cond_marginal_data, cond_marginal_label


######### MARGINAL SAMPLING DATASET ##################
def create_joint_marginal_dataset(dataset, x_idx, y_idx):
    # joint: 0, marginal: 1
    # marginal dataset construction
    number_of_samples = dataset.shape[0]
    x_data = dataset[:, x_idx]
    y_data = dataset[:, y_idx]

    if len(x_data.shape) < 2:
        x_data = x_data.reshape(-1, 1)

    if len(y_data.shape) < 2:
        y_data = y_data.reshape(-1, 1)

    selected_x_idx, selected_y_idx = marginal_sampling(number_of_samples)

    ## concat data samples
    marginal_data = np.concatenate([x_data[selected_x_idx, :], y_data[selected_y_idx, :]], axis=1)

    ## create labels
    marginal_label = np.ones(number_of_samples)

    # joint dataset construction
    joint_data = np.concatenate([x_data, y_data], axis=1)

    ## create labels
    joint_label = np.zeros(number_of_samples)

    return joint_data, joint_label, marginal_data, marginal_label


######### MULTICLASS CONDITIONAL DATASET ##################
def create_multiclass_conditional_dataset(dataset, x_idx, y_idx, z_idx):
    # joint: 0, all marginal: 1, marginal y joint xz: 2, marginal x joint yz
    # marginal dataset construction
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

    all_marginal, marginal_y_joint_xz, marginal_x_joint_yz = multiclass_conditional_sampling(number_of_samples)

    # all marginal
    selected_x_idx, selected_y_idx, selected_z_idx = all_marginal
    ## concat data samples
    all_marginal_data = np.concatenate([x_data[selected_x_idx, :], y_data[selected_y_idx, :],
                                        z_data[selected_y_idx, :]], axis=1)

    ## create labels
    all_marginal_label = np.ones(number_of_samples)

    # marginal y joint xz
    selected_y_idx, selected_xz_idx = marginal_y_joint_xz

    ## concat data samples
    marginal_y_joint_xz_data = np.concatenate([x_data[selected_xz_idx, :], y_data[selected_y_idx, :],
                                               z_data[selected_xz_idx, :]], axis=1)

    ## create labels
    marginal_y_joint_xz_label = np.empty(number_of_samples, dtype=int)
    marginal_y_joint_xz_label.fill(2)

    # marginal y joint xz
    selected_x_idx, selected_yz_idx = marginal_x_joint_yz

    ## concat data samples
    marginal_x_joint_yz_data = np.concatenate([x_data[selected_x_idx, :], y_data[selected_yz_idx, :],
                                               z_data[selected_yz_idx, :]], axis=1)

    ## create labels
    marginal_x_joint_yz_label = np.empty(number_of_samples, dtype=int)
    marginal_x_joint_yz_label.fill(3)

    # joint dataset construction
    joint_data = np.concatenate([x_data, y_data, z_data], axis=1)

    ## create labels
    joint_label = np.zeros(number_of_samples)

    return (joint_data, joint_label, all_marginal_data, all_marginal_label, marginal_y_joint_xz_data,
            marginal_y_joint_xz_label, marginal_x_joint_yz_data, marginal_x_joint_yz_label)
