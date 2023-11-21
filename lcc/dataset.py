import numpy as np
from lcc.polynomials import LCCPoly, InterpolatedPoly
from lcc.utils import create_encoded_dataset, calculate_gradient_samplewise


def create_lcc_dataset_k1_t1_scalar(prime, data_range, num_of_samples, weight):
    para_param, priv_param = 1, 1
    beta_arr, alpha_arr = [0, 1], [2, 3, 4]
    feature_size = 1
    return create_lcc_dataset(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr, para_param,
                              priv_param)


# TODO: we need to implement for K > 1
def create_lcc_dataset(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr, para_param,
                       priv_param):
    number_of_required_clients = 2 * (para_param + priv_param - 1) + 1
    assert len(alpha_arr) == number_of_required_clients, 'alpha array has to have more parameters'
    assert len(beta_arr) == (para_param + priv_param), 'beta array has to have more parameters'
    secret_data = np.random.randint(low=0, high=data_range, size=(num_of_samples, feature_size, 1))
    secret_label = np.random.randint(low=0, high=data_range, size=(num_of_samples, 1))
    encoded_secret_data_pol = [LCCPoly(beta_arr, [x], para_param, priv_param, prime, size=(feature_size, 1)) for x in
                               secret_data]
    encoded_secret_label_pol = [LCCPoly(beta_arr, [x[0]], para_param, priv_param, prime) for x in secret_label]
    print('pols created')

    # clients gradient calculations
    client_encoded_data = np.empty((number_of_required_clients, *secret_data.shape), dtype=int)
    client_encoded_label = np.empty((number_of_required_clients, *secret_label.shape), dtype=int)
    client_encoded_gradient = np.empty((number_of_required_clients, num_of_samples, *weight.shape), dtype=int)
    for client_idx in range(number_of_required_clients):
        client_encoded_data[client_idx], client_encoded_label[client_idx] = create_encoded_dataset(
            encoded_secret_data_pol, (feature_size, 1), encoded_secret_label_pol, alpha_arr[client_idx]
        )
        client_encoded_gradient[client_idx] = calculate_gradient_samplewise(client_encoded_data[client_idx],
                                                                            client_encoded_label[client_idx], weight,
                                                                            prime)

    revealed_gradients = np.empty((num_of_samples, *weight.shape), dtype=int)
    revealed_poly_constructed_coeff = np.empty((num_of_samples,
                                                2 * (para_param + priv_param - 1) + 1,
                                                *weight.shape), dtype=int)
    for gradient_idx in range(num_of_samples):
        revealed_poly = InterpolatedPoly([encoded_gradient[gradient_idx] for encoded_gradient in client_encoded_gradient],
                                         alpha_arr, prime)
        revealed_gradients[gradient_idx] = revealed_poly(beta_arr[0])
        revealed_poly_constructed_coeff[gradient_idx] = revealed_poly.coefficients

    # create dataset
    resulting_dataset = np.concatenate(
        [secret_data.reshape(num_of_samples, -1), secret_label, revealed_gradients.reshape(num_of_samples, -1),
         revealed_poly_constructed_coeff.reshape(num_of_samples, -1)],
        axis=1)
    print('dataset is created')
    return resulting_dataset
