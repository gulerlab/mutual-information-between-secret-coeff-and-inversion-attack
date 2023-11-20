import torch
import torch.nn as nn


def estimate_mi_for_binary_classification(model, joint_data, num_of_joint_trained, marginal_data,
                                          num_of_marginal_trained):
    ## estimate cmi
    model.eval()
    with torch.no_grad():
        model = model.to('cpu')
        estimated_logits_for_joint = model(torch.from_numpy(joint_data).to(torch.float32))
        estimated_logits_for_marginal = model(torch.from_numpy(marginal_data).to(torch.float32))
        joint_prob = torch.sigmoid(estimated_logits_for_joint)
        marginal_prob = torch.sigmoid(estimated_logits_for_marginal)
        class_distribution_trained = num_of_marginal_trained / num_of_joint_trained
        pointwise_dependency_joint = torch.log(torch.div(joint_prob[:, 0],
                                                         joint_prob[:, 1]) * class_distribution_trained)
        pointwise_dependency_marginal = torch.div(marginal_prob[:, 0],
                                                  marginal_prob[:, 1]) * class_distribution_trained
        curr_ldr = torch.sum(pointwise_dependency_joint) / pointwise_dependency_joint.size(0)
        curr_dv = ((torch.sum(pointwise_dependency_joint) / pointwise_dependency_joint.size(0)) -
                   torch.log(torch.sum(pointwise_dependency_marginal) / pointwise_dependency_marginal.size(0)))
        curr_nwj = ((torch.sum(pointwise_dependency_joint) / pointwise_dependency_joint.size(0)) -
                    (torch.sum(pointwise_dependency_marginal) / pointwise_dependency_marginal.size(0)) + 1)
        return curr_ldr, curr_dv, curr_nwj


def estimate_mi_for_multiclass_classification(model, joint_data, num_of_joint_trained,
                                              num_of_all_marginal_trained,
                                              num_of_marginal_y_joint_xz_trained,
                                              num_of_marginal_x_joint_yz_trained):
    softmax = nn.Softmax(dim=1)
    ## estimate cmi
    model.eval()
    with torch.no_grad():
        model = model.to('cpu')
        estimated_logits_for_joint = model(torch.from_numpy(joint_data).to(torch.float32))

        joint_prob = softmax(estimated_logits_for_joint)

        class_distribution_trained = ((num_of_marginal_y_joint_xz_trained + num_of_marginal_x_joint_yz_trained) /
                                      (num_of_joint_trained + num_of_all_marginal_trained))

        pointwise_dependency = joint_prob[:, 0] * joint_prob[:, 1]
        pointwise_dependency = torch.div(pointwise_dependency, joint_prob[:, 2])
        pointwise_dependency = torch.div(pointwise_dependency, joint_prob[:, 3])

        pointwise_dependency_joint = torch.log(pointwise_dependency * class_distribution_trained)
        curr_ldr = torch.sum(pointwise_dependency_joint) / pointwise_dependency_joint.size(0)
        return curr_ldr
