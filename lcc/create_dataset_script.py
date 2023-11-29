import numpy as np
import os
import argparse

import sys
sys.path.append('../')

from lcc.dataset import create_lcc_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prime', type=int, required=True)
    parser.add_argument('-dr', '--data_range', type=int, required=True)
    parser.add_argument('-nos', '--number_of_samples', type=int, required=True)
    parser.add_argument('-fs', '--feature_size', type=int, required=True)
    parser.add_argument('-b', '--beta_arr', nargs='+', type=int, required=True)
    parser.add_argument('-a', '--alpha_arr', nargs='+', type=int, required=True)
    parser.add_argument('-pap', '--para_param', type=int, required=True)
    parser.add_argument('-prp', '--priv_param', type=int, required=True)
    parser.add_argument('-n', '--name', type=str)

    # parsed_args = parser.parse_args("-p 5 -dr 2 -fs 2 -b 0 1 -a 2 3 4 -pap 1 -prp 1")
    args = parser.parse_args()
    prime = args.prime
    data_range = args.data_range
    number_of_samples = args.number_of_samples
    feature_size = args.feature_size
    beta_arr = args.beta_arr
    alpha_arr = args.alpha_arr
    para_param = args.para_param
    priv_param = args.priv_param
    name = args.name

    if name is None:
        name = 'p{}_dr{}_nos{}_fs{}.npy'.format(prime, data_range, number_of_samples, feature_size)

    print('SAVING {}'.format(name))

    weight = np.ones((feature_size, 1))
    dataset = create_lcc_dataset(prime, data_range, number_of_samples, weight, feature_size, beta_arr, alpha_arr,
                                 para_param, priv_param)

    with open(os.path.join('../data', name), 'wb') as fp:
        np.save(fp, dataset)

    print('DONE')
