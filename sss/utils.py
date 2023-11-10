import numpy as np


def pol_mul_mod(first, second, prime):
    resulting_pol = np.zeros(2 * (len(first) - 1) + 1)
    for first_idx, first_el in enumerate(first):
        for second_idx, second_el in enumerate(second):
            resulting_pol[first_idx + second_idx] = (resulting_pol[first_idx + second_idx] + (
                    (first_el * second_el) % prime)) % prime
    return resulting_pol


def pol_mul_mod_whole_data(data, degree, prime):
    first = data[:(degree + 1)]
    second = data[(degree + 1):]
    resulting_pol = np.zeros(2*degree + 1)
    for first_idx, first_el in enumerate(first):
        for second_idx, second_el in enumerate(second):
            resulting_pol[first_idx + second_idx] = (resulting_pol[first_idx + second_idx] + ((first_el * second_el) % prime)) % prime
    return resulting_pol
