import torch
import random
from torchvision import datasets, transforms
import resnet
import torch.optim as optim
import os
import numpy as np
import sys
from dataloader_cifar import cifar_dataloader
import argparse
import torch.nn.functional as F
from loss import NAL

parser = argparse.ArgumentParser(description='NAL PyTorch CIFAR10 and CIFAR100 Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size.')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate.')
parser.add_argument('--noise_mode', default='sym', help='the type of label noise, sym or asym.')
parser.add_argument('--lam', default=10, type=float, help='weight for penalty loss, pick from [0.5, 10, 50].')
parser.add_argument('--model', default='resnet34', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--delta', default=0.0, type=float,
                    help='')
parser.add_argument('--es', default=60, help='the epoch starts to update target, pick from [20, 60, 100].')
parser.add_argument('--beta', default=0.0, type=float,
                    help='')
parser.add_argument('--m', default=0.9, type=float,
                    help='the momentum in target estimation, pick from [0.7, 0.9, 0.99].')
parser.add_argument('--lr_s', default='CosineAnnealingWarmRestarts', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='NAL', type=str, help='loss function')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--seed', default=345)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--root_dir', default='/home/boy2/lyd/iterative_mixup/NAL_source_code/', type=str,
                    help='root path to dataset')
args = parser.parse_args()

assert args.num_epochs > args.es
if args.dataset == 'cifar10':
    print('############## Dataset CIFAR-10 ######################')
    num_class = 10
    _ = datasets.CIFAR10(root=args.root_dir, train=True, download=True)
    data_path = os.path.join(args.root_dir, 'cifar-10-batches-py')
elif args.dataset == 'cifar100':
    num_class = 100
    print('############## Dataset CIFAR-100 ######################')
    _ = datasets.CIFAR100(root=args.root_dir, train=True, download=True)
    data_path = os.path.join(args.root_dir, 'cifar-100-python')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fix the seed, for experiment
if args.seed:
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation


def train(log_interval, batch_size, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, confidence = model(data)
        if args.loss == 'NAL':
            loss = criterion(confidence, output, target, index, args.lam, epoch)
        elif args.loss == 'CE':
            loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * batch_size))

        if batch_idx % log_interval == 0:
            sys.stdout.write('\r')
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
            output_log.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
            output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return loss_per_epoch, acc_train_per_epoch


def test_cleaning(test_batch_size, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    sys.stdout.write('\r')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    output_log.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch


loader = cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                          num_workers=5,
                          root_dir=data_path,
                          noise_file='%s/%.1f_%s.json' % (data_path, args.r, args.noise_mode))

# load data
all_trainloader, noisy_labels, clean_labels = loader.run('train')
test_loader = loader.run('test')

if args.model == 'resnet34':
    model = model = resnet.ResNet34(num_classes=num_class).to(args.gpuid)

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

if args.lr_s == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
elif args.lr_s == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.001)

if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'NAL':
    criterion = NAL(noisy_labels, num_class, args.es, args.m, args.beta, args.delta)

exp_path = os.path.join('./',
                        '{0}_{1}_noise_{2}_{3}_{4}_{5}_{6}_bs={7}_lam={8}_beta={9}_delta={10}_es={11}_momentum={12}'.format(
                            args.dataset,
                            args.noise_mode,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size,
                            args.lam,
                            args.beta,
                            args.delta,
                            args.es,
                            args.m),
                        str(args.r) + '_seed=' + str(args.seed))

if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

output_log = open(exp_path + '/log.txt', 'w')

t = torch.zeros(50000, num_class).to(args.gpuid)
# cont = 0
acc_train_per_epoch_list = np.array([])
loss_train_per_epoch_list = np.array([])
acc_val_per_epoch_list = np.array([])
for epoch in range(1, args.num_epochs + 1):
    print('\t##### Training #####')
    loss_train_per_epoch, acc_train_per_epoch = train(
        args.log_interval,
        args.batch_size,
        model,
        args.gpuid,
        all_trainloader,
        optimizer,
        epoch)

    scheduler.step()
    _, acc_val_per_epoch = test_cleaning(args.batch_size, model, args.gpuid, test_loader)

    loss_train_per_epoch_list = np.append(loss_train_per_epoch_list, loss_train_per_epoch)
    acc_train_per_epoch_list = np.append(acc_train_per_epoch_list, acc_train_per_epoch)

    acc_val_per_epoch_list = np.append(acc_val_per_epoch_list, acc_val_per_epoch)

    # print('train acc: {}, test acc: {}'.format(acc_train_per_epoch,acc_val_per_epoch))


    if epoch == args.num_epochs:
        snapBest = 'trained_model_of_last_epoch'
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))


_,_=test_cleaning(args.batch_size, model, args.gpuid, test_loader)


