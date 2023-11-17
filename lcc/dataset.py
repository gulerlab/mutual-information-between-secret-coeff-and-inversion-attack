import numpy as np
from lcc.polynomials import LCCPoly, InterpolatedPoly
from lcc.utils import create_encoded_dataset, calculate_gradient_samplewise


def create_lcc_dataset_k1_t1_scalar(prime, data_range, num_of_samples, weight):
    para_param, priv_param = 1, 1
    beta_arr, alpha_arr = [0, 1], [2, 3, 4]
    secret_data = np.random.randint(low=0, high=data_range, size=(num_of_samples, 1))
    secret_label = np.random.randint(low=0, high=data_range, size=(num_of_samples, 1))
    encoded_secret_data_pol = [LCCPoly(beta_arr, [x[0]], para_param, priv_param, prime) for x in secret_data]
    encoded_secret_label_pol = [LCCPoly(beta_arr, [x[0]], para_param, priv_param, prime) for x in secret_label]

    # client 0
    client_0_encoded_data = create_encoded_dataset(encoded_secret_data_pol, encoded_secret_label_pol, alpha_arr[0])
    client_0_encoded_gradient = calculate_gradient_samplewise(client_0_encoded_data, weight, prime)

    # client 1
    client_1_encoded_data = create_encoded_dataset(encoded_secret_data_pol, encoded_secret_label_pol, alpha_arr[1])
    client_1_encoded_gradient = calculate_gradient_samplewise(client_1_encoded_data, weight, prime)

    # client 2
    client_2_encoded_data = create_encoded_dataset(encoded_secret_data_pol, encoded_secret_label_pol, alpha_arr[2])
    client_2_encoded_gradient = calculate_gradient_samplewise(client_2_encoded_data, weight, prime)

    revealed_poly_arr = []
    revealed_gradients = np.empty((client_0_encoded_gradient.shape[0], 1))
    revealed_random = np.empty((client_0_encoded_gradient.shape[0], 1))
    revealed_poly_constructed_coeff = np.empty((client_0_encoded_gradient.shape[0], 2 * para_param + 1))
    for gradient_idx in range(client_0_encoded_gradient.shape[0]):
        revealed_poly = InterpolatedPoly(
            [int(client_0_encoded_gradient[gradient_idx]), int(client_1_encoded_gradient[gradient_idx]),
             int(client_2_encoded_gradient[gradient_idx])], alpha_arr, prime)
        revealed_gradients[gradient_idx][0] = revealed_poly(beta_arr[0])
        revealed_random[gradient_idx][0] = revealed_poly(beta_arr[1])
        revealed_poly_constructed_coeff[gradient_idx] = revealed_poly.coefficients
        revealed_poly_arr.append(revealed_poly)

    # create dataset
    resulting_dataset = np.concatenate(
        [secret_data, secret_label, revealed_gradients, revealed_random, revealed_poly_constructed_coeff], axis=1)

    return resulting_dataset
