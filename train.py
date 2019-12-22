"""
Reimplementation of DeepCov and DeepCon contact prediction models

DeepCov:
Fully convolutional neural networks for protein residue-residue contact prediction
David T. Jones and Shaun M. Kandathil - University College London
https://github.com/psipred/DeepCov

DeepCon:
Dilated convolution network with dropout (best reported performing model, Fig.3d)
https://github.com/ba-lab/DEEPCON/
"""

import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# import customized modules
from models import DeepCon, DeepCov, BottleNeck, BasicBlock
from data import DeepCovDataset


# helper function to get learning rate during training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, args, scale=0.7):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = args.lr * scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    args.lr = lr


def resume(model, optimizer, args):
    if os.path.isfile(args.resume):
        print("Loading checkpoint '{}'..".format(args.resume))
        #checkpoint = torch.load(args.resume, map_location=device)
        checkpoint = torch.load(args.resume)
        # each time of resume, try with a new random seed
        seed = np.random.randint(1000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint '{}' loaded. Resume training from epoch {}.".format(args.resume, checkpoint['epoch']))
        print('Random seed = {}'.format(seed))
        return best_val_loss, best_val_acc,


# convert convolution output into symmetric contacts
def get_output(x, length):
    p2 = x[0, 0:length, 0:length]
    p3 = (p2 + p2.transpose(1, 0)) / 2
    p3 = p3.reshape(1, length, length)
    return p3


# since short-range contacts are not meaningful, do masking the pairs that < 5 residues:
def set_mask(length):
    assert length > 40, "Sequence length is too short."
    weight_map = np.ones((1, length, length), dtype=np.float32)
    rows, cols = np.indices((length, length))
    for ofs in range(-4, 5):
        row_vals = np.diag(rows, k=ofs)
        col_vals = np.diag(cols, k=ofs)
        weight_map[0, row_vals, col_vals] = 0.0
    return weight_map


def save_checkpoint(state, filename=''):
    torch.save(state, filename+'_checkpoint.pth.tar')
    shutil.copyfile(filename+'_checkpoint.pth.tar', filename+'_best_model.pth.tar')


def train_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        cov, cmap = data['cov'].to(device), data['map'].to(device)
        length = cmap.shape[-1]
        optimizer.zero_grad()
        output = model(cov)
        # create a mask
        mask = set_mask(length)
        mask = torch.from_numpy(mask).to(device)
        output = get_output(output[0], length).to(device)
        loss = F.binary_cross_entropy(output, cmap, weight=mask)
        loss.backward()
        optimizer.step()
        # compute train accuracy
        train_loss += loss.item()
        acc = (torch.round(output) == cmap).sum().float() / length**2
        train_accuracy += acc
        if batch_idx % args.log_interval == 0:  # report every log_interval batches (100)
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return train_loss / len(train_loader.dataset), train_accuracy / len(train_loader.dataset)


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for data in val_loader:
            cov, cmap = data['cov'].to(device), data['map'].to(device)
            length = cmap.shape[-1]
            output = model(cov)
            # masking the pairs whose distance is less than 5 residues:
            mask = set_mask(length)
            mask = torch.from_numpy(mask).to(device)
            output = get_output(output[0], length).to(device)
            loss = F.binary_cross_entropy(output, cmap, weight=mask)
            val_loss += loss.item()
            acc = (torch.round(output) == cmap).sum().float() / length ** 2
            val_acc += acc

    # return mean val loss
    return val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training CNN-based Contact Prediction Model')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Only batch size of 1 accepted')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Only batch size of 1 accepted')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--model-name', type=str, choices=['DeepCon', 'DeepCov'],
                        default='DeepCon')
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (for training resume)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parsing arguments
    args = parser.parse_args()

    # Data preparation
    target_list = []
    with open('train_ecod.lst') as target_file:
        for line in target_file:
            target = line.rstrip()
            target_list.append(target)

    print('Total targets = {}'.format(len(target_list)))

    # get first 130 targets for validation (same as in DeepCov and DeepCon)
    # np.random.shuffle(target_list)
    train_target_list = target_list[130:]
    val_target_list = target_list[:130]
    print('Training on {} targets, validating on {} targets.'.format(len(train_target_list),
                                                                     len(val_target_list)))

    # Load data
    train_dataset = DeepCovDataset('data/DeepCov/', train_target_list)
    val_dataset = DeepCovDataset('data/DeepCov/', val_target_list)
    # Since the tensors are different in size, accept batch size of 1 only:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            num_workers=4, shuffle=False)

    # set device
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    # init model
    if args.model_name == 'DeepCon':
        net_args = {
            "block": BottleNeck,
            "layers": [8, 8]
        }
        model = DeepCon(**net_args)
    else:  # model_name == DeepCov
        net_args = {
            "block": BasicBlock,
            "layers": [10]
        }
        model = DeepCov(**net_args)

    model.to(device)
    print('Model: ', args.model_name)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # in case of resuming, load previous training status
    if args.resume:
        best_val_loss, best_val_acc = resume(model, optimizer, args)
    else:
        print("No checkpoint specified. Training from the scratch.")
        best_val_loss = np.inf
        best_val_acc = 0

    # start training
    lr_adjust_patience = 0
    for epoch in range(args.start_epoch, args.epochs):
        print('lr = {}'.format(get_lr(optimizer)))
        train_loss, train_acc = train_epoch(args, model, device, train_loader, optimizer, epoch)
        print('Train loss = {:.4f}  Train acc = {:.4f}'.format(train_loss, train_acc))
        print('Evaluating ..')
        val_loss, val_acc = validate(model, device, val_loader)
        print('Val loss = {:.4f} - Val acc = {:.4f}'.format(val_loss, val_acc))

        # is_best = val_acc > best_val_acc
        is_best = val_loss < best_val_loss  # checkpoint by val loss shows better performance

        # reduce lr by 30% after 5 consecutive epochs without improvement
        if lr_adjust_patience > 5:
            adjust_learning_rate(optimizer, args, 0.7)
            lr_adjust_patience = 0

        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            # save checkpoint
            save_checkpoint(state={
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict()},
                filename=args.model_name
            )
        else:
            print('No improvement.')
            lr_adjust_patience += 1

        print('Best val loss = {:.4f}  - Best val acc = {:.4f}'.format(best_val_loss, best_val_acc))


if __name__ == '__main__':
    main()
