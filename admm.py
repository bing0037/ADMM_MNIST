from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import operator
import random

from testers import *
from numpy import linalg as LA
import yaml


class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z

            except yaml.YAMLError as exc:
                print(exc)


def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(args, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "pattern"):
        print("pattern pruning...")
        shape = weight.shape

        pattern1 = [[0, 0], [0, 2], [2, 0], [2, 2]]
        pattern2 = [[0, 0], [0, 1], [2, 1], [2, 2]]
        pattern3 = [[0, 0], [0, 1], [2, 0], [2, 1]]
        pattern4 = [[0, 0], [0, 1], [1, 0], [1, 1]]

        pattern5 = [[0, 2], [1, 0], [1, 2], [2, 0]]
        pattern6 = [[0, 0], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1]]
        pattern8 = [[0, 1], [0, 2], [2, 1], [2, 2]]

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 2]]
        pattern10 = [[0, 0], [0, 2], [1, 0], [1, 2]]
        pattern11 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        pattern12 = [[1, 0], [1, 1], [2, 0], [2, 1]]
        pattern13 = [[0, 1], [0, 2], [1, 1], [1, 2]]

        patterns_dict = {1 : pattern1,
                         2 : pattern2,
                         3 : pattern3,
                         4 : pattern4,
                         5 : pattern5,
                         6 : pattern6,
                         7 : pattern7,
                         8 : pattern8,
                         9 : pattern9,
                         10 : pattern10,
                         11 : pattern11,
                         12 : pattern12,
                         13 : pattern13
                         }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {} # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "random-pattern"):
        print("random_pattern pruning...", weight.shape)
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)

        pattern1 = [0, 2, 6, 8]
        pattern2 = [0, 1, 7, 8]
        pattern3 = [0, 1, 6, 7]
        pattern4 = [0, 1, 3, 4]

        pattern5 = [2, 3, 5, 6]
        pattern6 = [0, 3, 5, 8]
        pattern7 = [1, 2, 6, 7]
        pattern8 = [1, 2, 7, 8]

        pattern9 = [3, 5, 6, 8]
        pattern10 = [0, 2, 3, 5]
        pattern11 = [4, 5, 7, 8]
        pattern12 = [3, 4, 6, 7]
        pattern13 = [1 ,2 ,4, 5]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12,
                         13: pattern13
                         }

        for i in range(shape[0]):
            zero_idx = []
            for j in range(shape[1]):
                pattern_j = np.array(patterns_dict[random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])])
                zero_idx.append(pattern_j + 9 * j)
            zero_idx = np.array(zero_idx)
            zero_idx = zero_idx.reshape(1, -1)
            # print(zero_idx)
            weight2d[i][zero_idx] = 0

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_ratios[name])  # get sparse model in cuda

        elif option == "random":
            _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable


def test_sparsity(args, ADMM, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    total_zeros = 0
    total_nonzeros = 0
    if args.sparsity_type == "irregular":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(total_weight_number / total_nonzeros))
    elif args.sparsity_type == "column":
        for i, (name, W) in enumerate(model.named_parameters()):

            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            column_l2_norm = LA.norm(W2d, 2, axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif args.sparsity_type == "filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            row_l2_norm = LA.norm(W2d, 2, axis=1)
            zero_row = np.sum(row_l2_norm == 0)
            nonzero_row = np.sum(row_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif args.sparsity_type == "bn_filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            nonzeros = np.sum(W != 0)
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))


def admm_initialization(args, ADMM, model):
    if not args.admm:
        return
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer):
    if not args.admm:
        return

    if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args.verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name],
                                          ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(name),
                                      torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), epoch)
                    writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(name),
                                      torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), epoch)
                # print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                # print ("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))
            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm:

        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue

            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = args.admm_epoch
    lr = None
    if epoch % admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.lr * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training using ADMM with masks (loss_sum = loss(output,target)+loss(U,Z)). -libn
# Used for training: to update: Weight, U, Z according to loss(output, target)+loss(U,Z). -libn
def admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch):
    model.train()
    masks = {}
    writer = None

    # get masks from parameters! -libn
    for i, (name, W) in enumerate(model.named_parameters()):
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    if epoch == 1:
        # inialize Z variable
        # print("Start admm training quantized network, quantization type: {}".format(args.quant_type))
        admm_initialization(args, ADMM, model)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        ce_loss = F.cross_entropy(output, target)

        z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        mixed_loss.backward()


        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)] [lr: {}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), args.lr, ce_loss.item()))
            # test_column_sparsity(model)

# ?1:
# Seemingly, there is no difference between: masked_retrain() and combined_masked_retrain(). -libn
# Used for retraining: to update: Weight according to loss(output, target). -libn
def combined_masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch):
    if not args.masked_retrain:
        return

    idx_loss_dict = {}

    model.train()
    masks = {}

    # The only difference between masked_retrain() and combined_masked_retrain() ??? -libn
    # get masks from parameters! -libn
    with open("./profile/" + args.config_file + ".yaml", "r") as stream:
        raw_dict = yaml.load(stream)
        prune_ratios = raw_dict['prune_ratios']
    # The only difference between masked_retrain() and combined_masked_retrain() ??? -libn
    
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        _, weight = weight_pruning(args, W, prune_ratios[name])
        W.data = W
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("({}) ({}) cross_entropy loss: {}".format(args.sparsity_type, args.optmzr, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [lr: {}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item()))

        if batch_idx % 10 == 0:
            idx_loss_dict[batch_idx] = loss.item()

        # test_filter_sparsity(model)


        # test_sparsity(args, ADMM, model)
    return idx_loss_dict

# ?1:
# Seemingly, there is no difference between: masked_retrain() and combined_masked_retrain(). -libn
def masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch):
    if not args.masked_retrain:
        return

    idx_loss_dict = {}

    model.train()
    masks = {}
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        above_threshold, W = weight_pruning(args, W, ADMM.prune_ratios[name])
        W.data = W
        masks[name] = above_threshold

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("({}) cross_entropy loss: {}".format(args.sparsity_type, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item()))

        if batch_idx % 1 == 0:
            idx_loss_dict[batch_idx] = loss.item()

        # test_sparsity(args, ADMM, model)
    # admm.test_sparsity(args, ADMM, model)
    return idx_loss_dict


