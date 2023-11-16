import numpy as np
import torch


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


def create_probabilistic_classifier_dataset_for_conditional_mutual_information_gaussian():
    pass


def create_probabilistic_classifier_for_knn_approach_gaussian():
    pass


def create_train_test_split(data, label, train_test_ratio=0.8):
    train_size = int(data.shape[0] * train_test_ratio)
    train_data, train_label = data[list(range(train_size))], label[list(range(train_size))]
    test_data, test_label = data[list(range(train_size, data.shape[0]))], label[list(range(train_size, data.shape[0]))]
    return train_data, train_label, test_data, test_label


def create_batched_tensors(data, label, batch_size):
    data, label = torch.from_numpy(data).to(torch.float32), torch.from_numpy(label).to(torch.long)
    return torch.split(data, batch_size), torch.split(label, batch_size)


def create_batched_tensors_train_test(train_data, train_label, test_data, test_label, train_batch_size, test_batch_size):
    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label)
    test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)
    return torch.split(train_data, train_batch_size), torch.split(train_label, train_batch_size), torch.split(test_data, test_batch_size), torch.split(test_label, test_batch_size)
