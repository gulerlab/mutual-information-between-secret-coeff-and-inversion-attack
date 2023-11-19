import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.nn.functional import one_hot

from probabilistic_classifier.probabilistic_classifier import ProbabilisticClassifier


def train_binary_classifier(data, label, num_input_features, hidden_size_arr, lr,
                            number_of_iterations, batch_size, outer_iter, save_avg=100, print_progress=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_output_features = 2
    model = ProbabilisticClassifier(num_input_features, hidden_size_arr, num_output_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ## train classifier
    model.train()
    inner_running_loss = []
    inner_running_loss_avg = []
    curr_inner_running_loss_avg = 0

    num_of_joint = 0
    num_of_marginal = 0
    for inner_iter in range(number_of_iterations):
        selected_samples = np.random.choice(data.shape[0], batch_size, replace=False)
        batch_data, batch_label = data[selected_samples], label[selected_samples]
        batch_data, batch_label = torch.from_numpy(batch_data).to(torch.float32), torch.from_numpy(batch_label).to(
            torch.long)
        num_of_marginal += torch.count_nonzero(batch_label)
        num_of_joint += batch_size - torch.count_nonzero(batch_label)
        batch_label = one_hot(batch_label, num_classes=num_output_features).to(torch.float32)
        batch_data, batch_label = batch_data.to(device), batch_label.to(device)

        optimizer.zero_grad()
        logits = model(batch_data)
        loss = criterion(logits, batch_label)
        loss.backward()
        optimizer.step()

        inner_running_loss.append(loss.item())
        curr_inner_running_loss_avg += loss.item()

        if inner_iter == 0 or ((inner_iter + 1) % save_avg) == 0:
            if inner_iter > 0:
                curr_inner_running_loss_avg = curr_inner_running_loss_avg / save_avg
            if print_progress:
                print('trial: {}, iter: {}, curr loss: {}, avg loss: {}'.format(outer_iter + 1, inner_iter + 1,
                                                                                loss.item(),
                                                                                curr_inner_running_loss_avg))
            inner_running_loss_avg.append(curr_inner_running_loss_avg)
            curr_inner_running_loss_avg = 0

    return model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal


def train_binary_classifier_v2(data, label, num_input_features, hidden_size_arr, lr,
                               number_of_epoch, batch_size, outer_iter, save_avg=100, print_progress=True):
    # the only difference is the samples are not randomly selected but all data are given to
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_output_features = 2
    model = ProbabilisticClassifier(num_input_features, hidden_size_arr, num_output_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ## train classifier
    model.train()
    inner_running_loss = []
    inner_running_loss_avg = []

    num_of_joint = 0
    num_of_marginal = 0
    for epoch in range(number_of_epoch):
        batch_idx_tuple = torch.split(torch.from_numpy(np.random.permutation(np.arange(data.shape[0]))), batch_size)
        curr_inner_running_loss_avg = 0
        for batch_idx, selected_samples in enumerate(batch_idx_tuple):
            batch_data, batch_label = data[selected_samples], label[selected_samples]
            batch_data, batch_label = torch.from_numpy(batch_data).to(torch.float32), torch.from_numpy(batch_label).to(
                torch.long)
            num_of_marginal += torch.count_nonzero(batch_label)
            num_of_joint += batch_size - torch.count_nonzero(batch_label)
            batch_label = one_hot(batch_label, num_classes=num_output_features).to(torch.float32)
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_label)
            loss.backward()
            optimizer.step()

            inner_running_loss.append(loss.item())
            curr_inner_running_loss_avg += loss.item()

            if batch_idx == 0 or ((batch_idx + 1) % save_avg) == 0:
                if batch_idx > 0:
                    curr_inner_running_loss_avg = curr_inner_running_loss_avg / save_avg
                if print_progress:
                    print(
                        f'trial: {outer_iter + 1}, epoch, {epoch + 1}, iter: {batch_idx + 1}, curr loss: {loss.item()},'
                        f' avg loss: {curr_inner_running_loss_avg}')
                inner_running_loss_avg.append(curr_inner_running_loss_avg)
                curr_inner_running_loss_avg = 0

    return model, inner_running_loss, inner_running_loss_avg, num_of_joint, num_of_marginal
