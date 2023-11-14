import numpy as np


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
    marginal_data[:, 1] = joint_data[marginal_data_y_idx, 1]

    marginal_label = np.ones(half_samples)

    data = np.concatenate([joint_data, marginal_data])
    label = np.concatenate([joint_label, marginal_label])

    random_idx = np.random.permutation(num_of_samples)
    return data[random_idx], label[random_idx]


def create_probabilistic_classifier_dataset_for_conditional_mutual_information_gaussian():
    pass


def create_probabilistic_classifier_for_knn_approach_gaussian():
    pass
