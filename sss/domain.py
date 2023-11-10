import numpy as np
from utils import pol_mul_mod_whole_data
import itertools


def create_poly_mul_domain(degree, data_range, prime):
    data_range_domain = np.arange(data_range)
    coeff_range_domain = np.arange(prime)
    coeff_range_domain = [coeff_range_domain for _ in range(degree)]
    polynomial_combination_domain_generator = itertools.product(data_range_domain, *coeff_range_domain,
                                                                data_range_domain, *coeff_range_domain)

    len_generator = (data_range ** 2) * (prime ** (2 * degree))
    len_mul_poly = 2 * degree + 1
    len_poly = 2 * (1 + degree)
    pol_dom = np.empty((len_generator, len_poly + len_mul_poly))

    for idx, comb in enumerate(polynomial_combination_domain_generator):
        pol_coeff = np.asarray(comb)
        mul_pol_coeff = pol_mul_mod_whole_data(pol_coeff, degree, prime)

        pol_dom[idx] = np.concatenate([pol_coeff, mul_pol_coeff])

    return pol_dom
