import torch
import random
from resnet_for_clothing1m import resnet50_conf
import torch.optim as optim
import os
import numpy as np
import dataloader_clothing1m as dataloader
import argparse
import torch.nn.functional as F

from loss import NAL

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='PyTorch Clothing1m Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size.')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate.')
parser.add_argument('--lam', default=50, type=float, help='weight for penalty loss, pick from [0.5, 10, 50].')
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--delta', default=0.0, type=float,
                    help='')
parser.add_argument('--es', default=60, help='the epoch starts to update target, pick from [20, 60, 100].')
parser.add_argument('--beta', default=0.1, type=float, help='')
parser.add_argument('--m', default=0.9, type=float, help='the momentum parameter in target update.')
parser.add_argument('--lr_s', default='CosineAnnealingWarmRestarts', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='NAL', type=str, help='loss function')
parser.add_argument('--num_epochs', default=400, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--seed', default=345)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--root_dir', default='', type=str, help='path to dataset')
parser.add_argument('--num_batches', default=2000, type=int)
parser.add_argument('--dataset', default='clothing1M', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix the seed, for experiment
if args.seed:
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation


# top k accuracy calculation
def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


def train(log_interval, batch_size, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, confidence = model(data)
        if args.loss == 'NAL':
            loss = criterion(confidence, output, target, index, args.lam, epoch)
        elif args.loss == 'CE':
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * batch_size))

        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size), optimizer.param_groups[0]['lr']))
            output_log.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size), optimizer.param_groups[0]['lr']))
            output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return loss_per_epoch, acc_train_per_epoch


def test_cleaning(test_batch_size, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_test_per_batch = []
    test_loss = 0
    correct = 0
    test_total = 0
    test_correct_top1 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            pred_top1 = accuracy(output, target, (1,))
            test_total += 1
            test_correct_top1 += pred_top1.item()

            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_test_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    acc_top1 = float(test_correct_top1) / float(test_total)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, correct, len(test_loader.dataset), acc_top1))

    output_log.write('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, correct, len(test_loader.dataset), acc_top1))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]

    return loss_per_epoch, [acc_top1]


def test_val(val_batch_size, model, device, val_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    test_total = 0
    test_correct_top1 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            pred_top1 = accuracy(output, target, (1,))
            test_total += 1
            test_correct_top1 += pred_top1.item()

            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * val_batch_size))

    test_loss /= len(val_loader.dataset)
    acc_top1 = float(test_correct_top1) / float(test_total)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, correct, len(val_loader.dataset), acc_top1))

    output_log.write('\nVal set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, correct, len(val_loader.dataset), acc_top1))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    return loss_per_epoch, [acc_top1]


loader = dataloader.clothing_dataloader(root=args.root_dir, batch_size=args.batch_size, num_workers=5,
                                        num_batches=args.num_batches)

train_loader, noisy_labels = loader.run('train')
test_loader = loader.run('test')
val_loader = loader.run('val')

if args.model == 'resnet50':
    model = resnet50_conf(num_classes=args.num_class).to(args.gpuid)

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

if args.lr_s == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.0001)

if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'NAL':
    criterion = NAL(noisy_labels, args.num_class, args.es, args.m, args.beta, args.delta)

exp_path = os.path.join('./',
                        '{0}_{1}_{2}_{3}_{4}_{5}_bs={6}_lam={7}_beta={8}_delta={9}_es={10}_momentum={11}_batches={12}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size,
                            args.lam,
                            args.beta,
                            args.delta,
                            args.es,
                            args.m,
                            args.num_batches), 'seed=' + str(args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

output_log = open(exp_path + '/log.txt', 'w')

t = torch.zeros(50000, args.num_class).to(args.gpuid)
cont = 0
acc_train_per_epoch_model = np.array([])
acc_train_noisy_per_epoch_model = np.array([])
acc_train_clean_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])
for epoch in range(1, args.num_epochs + 1):
    print('\t##### Training clothing1m #####')
    loss_train_per_epoch, acc_train_per_epoch = train(args.log_interval,
                                                      args.batch_size,
                                                      model,
                                                      args.gpuid,
                                                      train_loader,
                                                      optimizer,
                                                      epoch)
    scheduler.step()

    val_loss, val_acc = test_val(args.batch_size, model, args.gpuid, val_loader)
    if epoch == 1:
        best_acc_val = val_acc[-1]
        snapBest='Best_val_acc_epoch_%d_valLoss_%.5f_valAcc_%.5f' % (epoch,val_loss[-1],best_acc_val)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
    else:
        if val_acc[-1] > best_acc_val:
            best_acc_val = val_acc[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                except OSError:
                    pass
            snapBest = 'Best_val_acc_epoch_%d_valLoss_%.5f_valAcc_%.5f' % (epoch, val_loss[-1], best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
    cont += 1


model.load_state_dict(torch.load(os.path.join(exp_path,snapBest+'.pth')))

_, _ = test_cleaning(args.batch_size, model, args.gpuid, test_loader)


