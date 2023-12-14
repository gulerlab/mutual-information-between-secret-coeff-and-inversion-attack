import numpy as np
import sys

sys.path.append('..')

from lcc.dataset import create_lcc_dataset
from probabilistic_classifier.dataset import create_joint_marginal_dataset, create_multiclass_conditional_dataset
from probabilistic_classifier.train import (train_binary_classifier_v2, train_multiclass_classifier_v2,
                                            train_binary_classifier_v3)
from probabilistic_classifier.estimate import (estimate_mi_for_binary_classification,
                                               estimate_mi_for_multiclass_classification)


def midiff_experiment(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr, para_param,
                      priv_param, x_idx, y_idx, z_idx, hidden_size_arr, lr, num_of_outer_iteration,
                      num_of_inner_iteration, batch_size, save_avg=200, print_progress=True, return_loss=False,
                      load_data=None, device=None):
    yz_idx = y_idx + z_idx

    if load_data is None:
        dataset = create_lcc_dataset(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr,
                                     para_param, priv_param)
    else:
        with open(load_data, 'rb') as fp:
            dataset = np.load(fp)

    # first mutual information
    joint_data, joint_label, marginal_data, marginal_label = create_joint_marginal_dataset(dataset, x_idx, yz_idx)
    data, label = np.concatenate([joint_data, marginal_data]), np.concatenate([joint_label, marginal_label])
    randomize_idx = np.random.permutation(np.arange(2 * num_of_samples))
    data, label = data[randomize_idx], label[randomize_idx]

    # train
    num_input_features = len(x_idx) + len(yz_idx)

    first_outer_running_loss = []
    first_outer_running_loss_avg = []
    first_ldr_estimations = []
    first_dv_estimations = []
    first_nwj_estimations = []

    for outer_iter in range(num_of_outer_iteration):
        print('################################################################')
        model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal = train_binary_classifier_v2(
            data, label, num_input_features, hidden_size_arr, lr, num_of_inner_iteration, batch_size, outer_iter,
            save_avg=save_avg, print_progress=print_progress, device=device)
        first_outer_running_loss.append(inner_running_loss)
        first_outer_running_loss_avg.append(inner_running_loss_avg)

        ## estimate cmi
        curr_ldr, curr_dv, curr_nwj = estimate_mi_for_binary_classification(model, joint_data, num_of_joint,
                                                                            marginal_data, num_of_marginal)
        print('trial: {}, ldr: {}, dv: {}, nwj: {}'.format(outer_iter + 1, curr_ldr.item(), curr_dv.item(),
                                                           curr_nwj.item()))
        print('################################################################\n')
        first_ldr_estimations.append(curr_ldr.item())
        first_dv_estimations.append(curr_dv.item())
        first_nwj_estimations.append(curr_nwj.item())

    first_final_ldr = np.mean(first_ldr_estimations)
    first_final_dv = np.mean(first_dv_estimations)
    first_final_nwj = np.mean(first_nwj_estimations)
    print('final estimations first:\n\tldr: {}\n\tdv: {}\n\tnwj: {}\n'.format(first_final_ldr, first_final_dv,
                                                                              first_final_nwj))

    # second mutual information
    joint_data, joint_label, marginal_data, marginal_label = create_joint_marginal_dataset(dataset, x_idx, z_idx)
    data, label = np.concatenate([joint_data, marginal_data]), np.concatenate([joint_label, marginal_label])
    randomize_idx = np.random.permutation(np.arange(2 * num_of_samples))
    data, label = data[randomize_idx], label[randomize_idx]

    # train
    num_input_features = len(x_idx) + len(z_idx)

    second_outer_running_loss = []
    second_outer_running_loss_avg = []
    second_ldr_estimations = []
    second_dv_estimations = []
    second_nwj_estimations = []

    for outer_iter in range(num_of_outer_iteration):
        print('################################################################')
        model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal = train_binary_classifier_v2(
            data, label, num_input_features, hidden_size_arr, lr, num_of_inner_iteration, batch_size, outer_iter,
            save_avg=save_avg, print_progress=print_progress, device=device)
        second_outer_running_loss.append(inner_running_loss)
        second_outer_running_loss_avg.append(inner_running_loss_avg)

        ## estimate cmi
        curr_ldr, curr_dv, curr_nwj = estimate_mi_for_binary_classification(model, joint_data, num_of_joint,
                                                                            marginal_data, num_of_marginal)
        print('trial: {}, ldr: {}, dv: {}, nwj: {}'.format(outer_iter + 1, curr_ldr.item(), curr_dv.item(),
                                                           curr_nwj.item()))
        print('################################################################\n')
        second_ldr_estimations.append(curr_ldr.item())
        second_dv_estimations.append(curr_dv.item())
        second_nwj_estimations.append(curr_nwj.item())

    second_final_ldr = np.mean(second_ldr_estimations)
    second_final_dv = np.mean(second_dv_estimations)
    second_final_nwj = np.mean(second_nwj_estimations)
    print('final estimations second:\n\tldr: {}\n\tdv: {}\n\tnwj: {}\n'.format(second_final_ldr, second_final_dv,
                                                                               second_final_nwj))

    final_cond_mi_ldr = first_final_ldr - second_final_ldr
    final_cond_mi_dv = first_final_dv - second_final_dv
    final_cond_mi_nwj = first_final_nwj - second_final_nwj

    if return_loss:
        return (final_cond_mi_ldr, final_cond_mi_dv, final_cond_mi_nwj, first_outer_running_loss,
                second_outer_running_loss, first_outer_running_loss_avg, second_outer_running_loss_avg)
    else:
        return final_cond_mi_ldr, final_cond_mi_dv, final_cond_mi_nwj


def multiclass_probabilistic_classifier_experiment(prime, data_range, num_of_samples, weight, feature_size, beta_arr,
                                                   alpha_arr, para_param, priv_param, x_idx, y_idx, z_idx,
                                                   hidden_size_arr, lr, num_of_outer_iteration, num_of_inner_iteration,
                                                   batch_size, save_avg=200, print_progress=True, return_loss=False,
                                                   device=None, load_data=None):
    if load_data is None:
        dataset = create_lcc_dataset(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr,
                                     para_param, priv_param)
    else:
        with open(load_data, 'rb') as fp:
            dataset = np.load(fp)

    # creating the training dataset
    (joint_data, joint_label, all_marginal_data, all_marginal_label, marginal_y_joint_xz_data,
     marginal_y_joint_xz_label, marginal_x_joint_yz_data,
     marginal_x_joint_yz_label) = create_multiclass_conditional_dataset(dataset, x_idx, y_idx, z_idx)

    # train
    num_input_features = len(x_idx) + len(y_idx) + len(z_idx)

    # iterate over many times
    outer_running_loss = []
    outer_running_loss_avg = []
    ldr_estimations = []

    for outer_iter in range(num_of_outer_iteration):
        print('################################################################')
        (model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_all_marginal,
         num_of_marginal_y_joint_xz, num_of_marginal_x_joint_yz) = train_multiclass_classifier_v2(joint_data,
                                                                                                  joint_label,
                                                                                                  all_marginal_data,
                                                                                                  all_marginal_label,
                                                                                                  marginal_x_joint_yz_data,
                                                                                                  marginal_x_joint_yz_label,
                                                                                                  marginal_y_joint_xz_data,
                                                                                                  marginal_y_joint_xz_label,
                                                                                                  num_input_features,
                                                                                                  hidden_size_arr, lr,
                                                                                                  num_of_inner_iteration,
                                                                                                  batch_size,
                                                                                                  outer_iter,
                                                                                                  print_progress=print_progress,
                                                                                                  save_avg=save_avg,
                                                                                                  device=device)
        outer_running_loss.append(inner_running_loss)
        outer_running_loss_avg.append(inner_running_loss_avg)

        ## estimate cmi
        curr_ldr = estimate_mi_for_multiclass_classification(model, joint_data, num_of_joint, num_of_all_marginal,
                                                             num_of_marginal_y_joint_xz, num_of_marginal_x_joint_yz)
        print('trial: {}, ldr: {}'.format(outer_iter + 1, curr_ldr.item()))
        print('################################################################\n')
        ldr_estimations.append(curr_ldr.item())

    final_ldr = np.mean(ldr_estimations)
    print('final estimations:\n\tldr: {}'.format(final_ldr))

    if return_loss:
        return final_ldr, outer_running_loss, outer_running_loss_avg
    else:
        return final_ldr


def ccmi_experiment(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr, para_param,
                    priv_param, x_idx, y_idx, z_idx, hidden_size_arr, lr, num_of_outer_iteration,
                    num_of_inner_iteration, batch_size, save_avg=200, print_progress=True, return_loss=False,
                    load_data=None, device=None):
    yz_idx = y_idx + z_idx

    if load_data is None:
        dataset = create_lcc_dataset(prime, data_range, num_of_samples, weight, feature_size, beta_arr, alpha_arr,
                                     para_param, priv_param)
    else:
        with open(load_data, 'rb') as fp:
            dataset = np.load(fp)

    # first mutual information
    joint_data, joint_label, marginal_data, marginal_label = create_joint_marginal_dataset(dataset, x_idx, yz_idx)

    # train test split
    train_idx = np.arange(int(joint_data.shape[0] * 2 / 3))
    test_idx = np.arange(len(train_idx), joint_data.shape[0])
    joint_data_train, joint_label_train, joint_data_test = (joint_data[train_idx],
                                                            joint_label[train_idx],
                                                            joint_data[test_idx])

    marginal_data_train, marginal_label_train, marginal_data_test = (marginal_data[train_idx],
                                                                     marginal_label[train_idx],
                                                                     marginal_data[test_idx])

    # train
    num_input_features = len(x_idx) + len(yz_idx)

    first_outer_running_loss = []
    first_outer_running_loss_avg = []
    first_ldr_estimations = []
    first_dv_estimations = []
    first_nwj_estimations = []

    for outer_iter in range(num_of_outer_iteration):
        print('################################################################')
        model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal = train_binary_classifier_v3(
            joint_data_train, joint_label_train, marginal_data_train, marginal_label_train, num_input_features,
            hidden_size_arr, lr, num_of_inner_iteration, batch_size, outer_iter,
            save_avg=save_avg, print_progress=print_progress, device=device)
        first_outer_running_loss.append(inner_running_loss)
        first_outer_running_loss_avg.append(inner_running_loss_avg)

        ## estimate cmi
        curr_ldr, curr_dv, curr_nwj = estimate_mi_for_binary_classification(model, joint_data_test, num_of_joint,
                                                                            marginal_data_test, num_of_marginal)
        print('trial: {}, ldr: {}, dv: {}, nwj: {}'.format(outer_iter + 1, curr_ldr.item(), curr_dv.item(),
                                                           curr_nwj.item()))
        print('################################################################\n')
        first_ldr_estimations.append(curr_ldr.item())
        first_dv_estimations.append(curr_dv.item())
        first_nwj_estimations.append(curr_nwj.item())

    first_final_ldr = np.mean(first_ldr_estimations)
    first_final_dv = np.mean(first_dv_estimations)
    first_final_nwj = np.mean(first_nwj_estimations)
    print('final estimations first:\n\tldr: {}\n\tdv: {}\n\tnwj: {}\n'.format(first_final_ldr, first_final_dv,
                                                                              first_final_nwj))

    # second mutual information
    joint_data, joint_label, marginal_data, marginal_label = create_joint_marginal_dataset(dataset, x_idx, z_idx)

    # train test split
    train_idx = np.arange(int(joint_data.shape[0] * 2 / 3))
    test_idx = np.arange(len(train_idx), joint_data.shape[0])
    joint_data_train, joint_label_train, joint_data_test = (joint_data[train_idx],
                                                            joint_label[train_idx],
                                                            joint_data[test_idx])

    marginal_data_train, marginal_label_train, marginal_data_test = (marginal_data[train_idx],
                                                                     marginal_label[train_idx],
                                                                     marginal_data[test_idx])

    # train
    num_input_features = len(x_idx) + len(z_idx)

    second_outer_running_loss = []
    second_outer_running_loss_avg = []
    second_ldr_estimations = []
    second_dv_estimations = []
    second_nwj_estimations = []

    for outer_iter in range(num_of_outer_iteration):
        print('################################################################')
        model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal = train_binary_classifier_v3(
            joint_data_train, joint_label_train, marginal_data_train, marginal_label_train, num_input_features,
            hidden_size_arr, lr, num_of_inner_iteration, batch_size, outer_iter,
            save_avg=save_avg, print_progress=print_progress, device=device)
        second_outer_running_loss.append(inner_running_loss)
        second_outer_running_loss_avg.append(inner_running_loss_avg)

        ## estimate cmi
        curr_ldr, curr_dv, curr_nwj = estimate_mi_for_binary_classification(model, joint_data_test, num_of_joint,
                                                                            marginal_data_test, num_of_marginal)
        print('trial: {}, ldr: {}, dv: {}, nwj: {}'.format(outer_iter + 1, curr_ldr.item(), curr_dv.item(),
                                                           curr_nwj.item()))
        print('################################################################\n')
        second_ldr_estimations.append(curr_ldr.item())
        second_dv_estimations.append(curr_dv.item())
        second_nwj_estimations.append(curr_nwj.item())

    second_final_ldr = np.mean(second_ldr_estimations)
    second_final_dv = np.mean(second_dv_estimations)
    second_final_nwj = np.mean(second_nwj_estimations)
    print('final estimations second:\n\tldr: {}\n\tdv: {}\n\tnwj: {}\n'.format(second_final_ldr, second_final_dv,
                                                                               second_final_nwj))

    final_cond_mi_ldr = first_final_ldr - second_final_ldr
    final_cond_mi_dv = first_final_dv - second_final_dv
    final_cond_mi_nwj = first_final_nwj - second_final_nwj

    if return_loss:
        return (final_cond_mi_ldr, final_cond_mi_dv, final_cond_mi_nwj, first_outer_running_loss,
                second_outer_running_loss, first_outer_running_loss_avg, second_outer_running_loss_avg)
    else:
        return final_cond_mi_ldr, final_cond_mi_dv, final_cond_mi_nwj
