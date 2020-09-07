from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
from lenet import LeNet
import numpy as np
import admm

import testers

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/Mnist')

# Training using ADMM without masks (loss_sum = loss(output,target)+loss(U,Z)). -libn
def train(args, ADMM, model, device, train_loader, optimizer, epoch, writer):
    model.train()

    ce_loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)

        # ADMM 3: update Z & U and calculate mixed_loss
        admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        # ADMM 4: backprop of mixed_loss
        mixed_loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))

    if args.verbose:
        writer.add_scalar('Train/Cross_Entropy', ce_loss, epoch)
        for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))

        for k in ADMM.prune_ratios:
            writer.add_scalar('layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], epoch)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))



def main():
    if True: # training hyperparameters:
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 2)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--lr_decay', type=int, default=30, metavar='LR_decay',
                            help='how many every epoch before lr drop (default: 30)')
        parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
                            help='optimizer used (default: sgd)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save_model', type=str, default="pretrained_mnist.pt",
                            help='For Saving the current Model')
        parser.add_argument('--load_model', type=str, default=None,
                            help='For loading the model')
        parser.add_argument('--masked_retrain', action='store_true', default=False,
                            help='for masked retrain')
        parser.add_argument('--verbose', action='store_true', default=False,
                            help='whether to report admm convergence condition')
        parser.add_argument('--admm', action='store_true', default=False,
                            help="for admm training")
        parser.add_argument('--combine_progressive', action='store_true', default=False,
                            help="for filter pruning after column pruning")
        parser.add_argument('--admm_epoch', type=int, default=9,
                            help="how often we do admm update")
        parser.add_argument('--rho', type=float, default=0.001,
                            help="define rho for ADMM")
        parser.add_argument('--sparsity_type', type=str, default='pattern',
                            help="define sparsity_type: [irregular,column,filter,pattern,random-pattern]")
        parser.add_argument('--config_file', type=str, default='config',
                            help="prune config file")
        parser.add_argument('--rho_num', type=int, default=1,
                            help="define how many rohs for ADMM training")
        parser.add_argument('--lr_scheduler', type=str, default='cosine',
                            help='define lr scheduler')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # create data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = LeNet().to(device)  ## training lenet with bn




    """====================="""
    """ multi-rho admm train"""
    """====================="""

    initial_rho = args.rho
    if args.admm:
        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i
            if i == 0:
                model.load_state_dict(torch.load("./models/model_pretrain.pt"))  # admm train need basline model
                print('Pre-train model!!!!!!')
                testers.test_irregular_sparsity(model)
                model.cuda()
                save_dir = './models_ADMM'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            else:
                model.load_state_dict(torch.load(save_dir+'/model_mnist_{}_{}_{}.pt'.format(current_rho/10, args.config_file, args.sparsity_type)))
                model.cuda()
            # ADMM 1: intialization
            ADMM = admm.ADMM(model, "./profile/" + args.config_file + ".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM, model)  # intialize Z and U variables

            best_prec1 = 0.
            lr = args.lr / 10
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
            for epoch in range(1, args.epochs + 1):
                # ADMM 2: ADMM lr
                admm.admm_adjust_learning_rate(optimizer, epoch, args)

                if args.lr_scheduler == 'default':
                    adjust_learning_rate(optimizer, epoch, args)
                elif args.lr_scheduler == 'cosine':
                    scheduler.step()

                print("current rho: {}".format(current_rho))
                if args.combine_progressive: # ADMM training with masks to constraint weight gradients!
                    admm.admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch)
                else:                        # ADMM training without masks to constraint weight gradients!
                    train(args, ADMM, model, device, train_loader, optimizer, epoch, writer)

                prec1 = test(args, model, device, test_loader)
                best_prec1 = max(prec1, best_prec1)
            print("Saving ADMM model...")
            torch.save(model.state_dict(), save_dir+'/model_mnist_{}_{}_{}.pt'.format(current_rho, args.config_file, args.sparsity_type))
            print('ADMM-trained model!!!!!!')
            testers.test_irregular_sparsity(model)




    """========================"""
    """END multi-rho admm train"""
    """========================"""



    """=============="""
    """masked retrain"""
    """=============="""

    if args.masked_retrain:
        # load admm trained model
        print("\n>_ Loading file...")
        model.load_state_dict(torch.load("./models_ADMM/model_mnist_{}_{}_{}.pt".format(initial_rho*10**(args.rho_num-1), args.config_file, args.sparsity_type)))
        model.cuda()

        print("before retrain starts")
        test(args, model, device, test_loader)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        ADMM = admm.ADMM(model, file_name="./profile/" + args.config_file + ".yaml", rho=initial_rho)
        admm.hard_prune(args, ADMM, model)
        admm.test_sparsity(args, ADMM, model)
        best_prec1 = [0]
        epoch_loss_dict = {}
        testAcc = []
        for epoch in range(1, args.epochs + 1):
            if epoch == 1:
                save_dir = './models_ADMM_pruned'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                             eta_min=4e-08)
            scheduler.step()
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
            if args.combine_progressive:
                idx_loss_dict = admm.combined_masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch)
            else:
                idx_loss_dict = admm.masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch)
            prec1 = test(args, model, device, test_loader)
            if prec1 > max(best_prec1):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
                torch.save(model.state_dict(),save_dir+"/mnist_retrained_acc_{:.3f}_{}rhos_{}_{}.pt".format(
                               prec1, args.rho_num, args.config_file, args.sparsity_type))
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_prec1)))
                if len(best_prec1) > 1:
                    os.remove(save_dir+"/mnist_retrained_acc_{:.3f}_{}rhos_{}_{}.pt".format(
                        max(best_prec1), args.rho_num, args.config_file, args.sparsity_type))

            epoch_loss_dict[epoch] = idx_loss_dict
            testAcc.append(prec1)

            best_prec1.append(prec1)

        print("after retraining")
        test(args, model, device, test_loader)
        print('ADMM-pruned & trained model!!!!!!')
        testers.test_irregular_sparsity(model)
        # admm.test_sparsity(args, ADMM, model)

        print("Best Acc: {:.4f}".format(max(best_prec1)))
        save_dir = './plotable'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir+"/plotable_{}.npy".format(args.sparsity_type), epoch_loss_dict)
        np.save(save_dir+"/testAcc_{}.npy".format(args.sparsity_type), testAcc)


    """=============="""
    """masked retrain"""
    """=============="""


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
