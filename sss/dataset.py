import numpy as np
from utils import pol_mul_mod


def create_sss_dataset(prime, data_domain_range, num_of_samples, degree, norm=False):
    # creating the dataset
    ## starting from basic shamir secret sharing with real numbers
    ## creating the polynomials first (reverse order - higher degree in the beginning)
    first_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    first_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    first_poly_dataset = np.hstack([first_poly_data, first_poly_coeff])

    # to create a resulting polynomial without any zero in the largest degree
    second_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    second_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    second_poly_dataset = np.hstack([second_poly_data, second_poly_coeff])

    multiplied_poly_dataset = np.empty((num_of_samples, 2 * degree + 1))
    for idx in range(num_of_samples):
        multiplied_poly_dataset[idx] = pol_mul_mod(first_poly_dataset[idx], second_poly_dataset[idx], prime)

    mutual_information_dataset = np.empty((num_of_samples, 4 * degree + 3))
    mutual_information_dataset[:, (2 * degree + 2):] = multiplied_poly_dataset
    mutual_information_dataset[:, (degree + 1):(2 * degree + 2)] = second_poly_dataset
    mutual_information_dataset[:, :(degree + 1)] = first_poly_dataset
    mutual_information_dataset = mutual_information_dataset.astype(np.float64)

    if norm:
        mean_mi_dataset = np.mean(mutual_information_dataset)
        std_mi_dataset = np.std(mutual_information_dataset)
        mutual_information_dataset = (mutual_information_dataset - mean_mi_dataset) / std_mi_dataset

    return mutual_information_dataset


def create_cond_mi_dataset_by_subsampling(prime, data_domain_range, num_of_samples, degree, condition_dict):
    # creating the dataset
    ## starting from basic shamir secret sharing with real numbers
    ## creating the polynomials first (reverse order - higher degree in the beginning)
    first_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    first_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    first_poly_dataset = np.hstack([first_poly_data, first_poly_coeff])

    # to create a resulting polynomial without any zero in the largest degree
    second_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    second_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    second_poly_dataset = np.hstack([second_poly_data, second_poly_coeff])

    multiplied_poly_dataset = np.empty((num_of_samples, 2 * degree + 1))
    for idx in range(num_of_samples):
        multiplied_poly_dataset[idx] = pol_mul_mod(first_poly_dataset[idx], second_poly_dataset[idx], prime)

    mutual_information_dataset = np.empty((num_of_samples, 2 * degree + 3))
    mutual_information_dataset[:, 2:] = multiplied_poly_dataset
    mutual_information_dataset[:, 1] = second_poly_dataset[:, 0]
    mutual_information_dataset[:, 0] = first_poly_dataset[:, 0]
    mutual_information_dataset = mutual_information_dataset.astype(np.float64)

    # based on condition dict generate a mask
    cond_mask = np.ones(len(mutual_information_dataset), dtype=bool)
    for key, value in condition_dict.items():
        curr_mask = mutual_information_dataset[:, key] == value
        cond_mask = np.logical_and(curr_mask, cond_mask)

    cond_idx_mask = np.arange(len(mutual_information_dataset))[cond_mask]
    mutual_information_dataset = mutual_information_dataset[cond_idx_mask]

    left_idx_arr = [x for x in range(mutual_information_dataset.shape[-1]) if x not in condition_dict.keys()]

    mutual_information_dataset = mutual_information_dataset[:, left_idx_arr]
    return mutual_information_dataset


def satisfied_multiplied_poly(multiplied_poly, condition_dict, offset):
    for key, value in condition_dict.items():
        if multiplied_poly[key - offset] == value:
            continue
        else:
            return False

    return True


def create_cond_mi_dataset_by_conditionally_init(prime, data_domain_range, num_of_samples, degree, condition_dict):
    mutual_information_dataset = np.empty((num_of_samples, 2 * degree + 3))
    idx = 0
    while True:
        # creating the dataset
        ## starting from basic shamir secret sharing with real numbers
        ## creating the polynomials first (reverse order - higher degree in the beginning)
        first_poly_data = np.random.randint(data_domain_range, size=(1, 1))
        first_poly_coeff = np.random.randint(prime, size=(1, degree))
        first_poly_dataset = np.hstack([first_poly_data, first_poly_coeff])

        # to create a resulting polynomial without any zero in the largest degree
        second_poly_data = np.random.randint(data_domain_range, size=(1, 1))
        second_poly_coeff = np.random.randint(prime, size=(1, degree))
        second_poly_dataset = np.hstack([second_poly_data, second_poly_coeff])

        multiplied_poly = pol_mul_mod(np.squeeze(first_poly_dataset), np.squeeze(second_poly_dataset), prime)
        if satisfied_multiplied_poly(multiplied_poly, condition_dict, 2):
            mutual_information_dataset[idx, 2:] = multiplied_poly
            mutual_information_dataset[idx, 1] = second_poly_dataset[0, 0]
            mutual_information_dataset[idx, 0] = first_poly_dataset[0, 0]
            idx += 1

        if idx == num_of_samples:
            break

    mutual_information_dataset = mutual_information_dataset.astype(np.float64)

    left_idx_arr = [x for x in range(mutual_information_dataset.shape[-1]) if x not in condition_dict.keys()]
    mutual_information_dataset = mutual_information_dataset[:, left_idx_arr]
    return mutual_information_dataset
