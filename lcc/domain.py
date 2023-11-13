from .polynomials import InterpolatedPoly
import numpy as np
import itertools


def create_lcc_domain(K, data_range, T, prime, beta_arr):
    data_range_domain = np.arange(data_range)
    random_range_domain = np.arange(prime)
    data_range_domain = [data_range_domain for _ in range(K)]
    random_range_domain = [random_range_domain for _ in range(T)]
    secret_random_domain_generator = itertools.product(*data_range_domain, *random_range_domain, *data_range_domain,
                                                       *random_range_domain)

    domain_size = ((data_range ** K) * (prime ** T)) ** 2
    domain_feature_size = 3 * (K + T) + 2 * (K + T - 1) + 1
    domain = np.empty((domain_size, domain_feature_size))

    for idx, secret_random in enumerate(secret_random_domain_generator):
        secret_random = list(secret_random)
        domain[idx, :(2 * (K + T))] = secret_random
        first_interpolated_poly = InterpolatedPoly(secret_random[:(K + T)], beta_arr, prime)
        second_interpolated_poly = InterpolatedPoly(secret_random[(K + T):], beta_arr, prime)
        multiplied_poly = first_interpolated_poly * second_interpolated_poly
        for beta_idx, beta in enumerate(beta_arr):
            domain[idx, (2 * (K + T)) + beta_idx] = multiplied_poly(beta)

        domain[idx, (3 * (K + T)):] = multiplied_poly.coefficients

    return domain


def create_lcc_gradient_domain_basic_setup(data_range, beta_arr, alpha_arr, weight, prime):
    # point-wise gradient calculation for linear regression for size 1
    # K, T = 1
    # g = -2(xy - x^2w)
    # x = y
    data_range_domain = np.arange(data_range)
    random_range_domain = np.arange(prime)
    random_range_domain = [random_range_domain for _ in range(2)]
    secret_random_domain_generator = itertools.product(data_range_domain, *random_range_domain)

    domain_size = data_range * prime * prime
    domain_feature_size = 3 * 2 + 3
    domain = np.empty((domain_size, domain_feature_size), dtype=int)

    def calculate_gradient(encoded_data, encoded_label):
        gradient = (encoded_label - encoded_data * weight) % prime
        gradient = (encoded_data * gradient) % prime
        gradient = (-2 * gradient) % prime
        return gradient

    for idx, secret_random in enumerate(secret_random_domain_generator):
        secret_random = list(secret_random)
        label = secret_random[0]
        domain[idx, 0], domain[idx, 1], domain[idx, 2], domain[idx, 3] = secret_random[0], secret_random[1], label, secret_random[2]

        secret_data_pol = InterpolatedPoly(domain[idx, :2], beta_arr, prime)
        secret_label_pol = InterpolatedPoly(domain[idx, 2:4], beta_arr, prime)

        encoded_data_01 = secret_data_pol(alpha_arr[0])
        encoded_data_02 = secret_data_pol(alpha_arr[1])
        encoded_data_03 = secret_data_pol(alpha_arr[2])

        encoded_label_01 = secret_label_pol(alpha_arr[0])
        encoded_label_02 = secret_label_pol(alpha_arr[1])
        encoded_label_03 = secret_label_pol(alpha_arr[2])

        encoded_gradient_01 = calculate_gradient(encoded_data_01, encoded_label_01)
        encoded_gradient_02 = calculate_gradient(encoded_data_02, encoded_label_02)
        encoded_gradient_03 = calculate_gradient(encoded_data_03, encoded_label_03)

        gradient_pol = InterpolatedPoly([encoded_gradient_01, encoded_gradient_02, encoded_gradient_03],
                                        alpha_arr, prime)
        domain[idx, 4], domain[idx, 5] = gradient_pol(beta_arr[0]), gradient_pol(beta_arr[1])
        domain[idx, 6:] = gradient_pol.coefficients
    return domain
