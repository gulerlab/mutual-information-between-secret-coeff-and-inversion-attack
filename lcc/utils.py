import numpy as np
import galois
import numbers


# utils
def modulo_inverse(x, p):
    return pow(x, -1, p)


def finite_field_division(quotient, dividend, p):
    # find modulo inverse of dividend and multiply with the quotient
    mod_inv_div = modulo_inverse(dividend, p)
    return (mod_inv_div * quotient) % p


def poly_coefficients(evaluated_points, evaluation_points, p):
    vander_matrix = np.vander(evaluation_points, increasing=True) % p
    galois_field = galois.GF(p)
    vander_matrix = galois_field(vander_matrix)
    inv_vander_matrix = np.linalg.inv(vander_matrix)

    if isinstance(evaluated_points[0], numbers.Number):
        evaluated_points_in_field = galois_field(np.asarray(evaluated_points)[:, np.newaxis])
        return np.asarray([x for x in np.squeeze(inv_vander_matrix @ evaluated_points_in_field)])
    else:
        evaluated_points_in_field = np.asarray(evaluated_points)
        num_of_evaluations, dim_0, dim_1 = evaluated_points_in_field.shape
        evaluated_points_in_field = galois_field(evaluated_points_in_field.reshape(num_of_evaluations, -1))
        inverted_coefficients = np.squeeze(inv_vander_matrix @ evaluated_points_in_field).reshape(num_of_evaluations,
                                                                                                  dim_0, dim_1)
        resulting_coefficients = np.empty(inverted_coefficients.shape)
        for i in range(inverted_coefficients.shape[0]):
            for j in range(inverted_coefficients.shape[1]):
                for k in range(inverted_coefficients.shape[2]):
                    resulting_coefficients[i, j, k] = inverted_coefficients[i, j, k].item()

        return resulting_coefficients


def pol_mul_mod(first, second, prime):
    resulting_pol = np.zeros(2 * (len(first) - 1) + 1)
    for first_idx, first_el in enumerate(first):
        for second_idx, second_el in enumerate(second):
            resulting_pol[first_idx + second_idx] = (resulting_pol[first_idx + second_idx] + (
                    (first_el * second_el) % prime)) % prime
    return resulting_pol


def create_encoded_dataset(encoded_data_pol, data_size, encoded_label_pol, alpha):
    encoded_data = np.empty((len(encoded_data_pol), *data_size))
    encoded_label = np.empty((len(encoded_label_pol), 1))
    for idx, (data_pol, label_pol) in enumerate(zip(encoded_data_pol, encoded_label_pol)):
        encoded_data[idx] = data_pol(alpha)
        encoded_label[idx][0] = label_pol(alpha)
    return encoded_data, encoded_label


def calculate_gradient_samplewise(encoded_data, encoded_label, curr_weight, p):
    # g = -2(xy - x^2w)
    gradient = np.empty((encoded_data.shape[0], *curr_weight.shape))
    for idx in range(encoded_data.shape[0]):
        data, label = encoded_data[idx], encoded_label[idx]
        gradient[idx] = (label - (data.T @ curr_weight)[0][0]) % p
        gradient[idx] = (data * gradient[idx]) % p
        gradient[idx] = (-2 * gradient[idx]) % p
    gradient = gradient.astype(int)
    return gradient
