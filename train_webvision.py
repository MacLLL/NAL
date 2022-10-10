import torch
import random
import torch.optim as optim
import os
import sys
import numpy as np
import dataloader_webvision as dataloader
import argparse
from InceptionResNetV2 import *
import torch.nn.functional as F
from loss import NAL

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='PyTorch webvision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize.')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate.')
parser.add_argument('--lam', default=50, type=float, help='weight for penalty loss.')
parser.add_argument('--delta', default=0.0, type=float, help='')
parser.add_argument('--es', default=200, help='the epoch starts update target.')
parser.add_argument('--beta', default=0.1, type=float, help='')
parser.add_argument('--m', default=0.9, type=float, help='the parameter in target update.')
parser.add_argument('--model', default='InceptionResNetV2', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--lr_s', default='CosineAnnealingWarmRestarts', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='NAL', type=str, help='loss function')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--seed', default=345)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='', type=str,
                    help='path to dataset')  # for webvision
parser.add_argument('--dataset', default='webvision', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix the seed, for experiment
if args.seed:
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation


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


def train( log_interval, batch_size, model, device, train_loader, optimizer, epoch):
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
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
            output_log.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
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
    correct_topk = 0

    test_total = 0
    test_correct_top1 = 0
    test_correct_top5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)

            pred_top5 = accuracy(output, target, (5,))
            pred_top1 = accuracy(output, target, (1,))
            test_total += 1
            test_correct_top1 += pred_top1.item()
            test_correct_top5 += pred_top5.item()

            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            k = 5

            pred_topk = torch.topk(output, k)[1]

            for i in range(k):
                correct_topk += (pred_topk[:, i].view_as(pred)).eq(target.view_as(pred)).sum().item()


            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))
            acc_val_per_batch.append(100. * correct_topk / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    acc_top1 = float(test_correct_top1) / float(test_total)
    acc_top5 = float(test_correct_top5) / float(test_total)

    print(
        '\nTest set: Average loss: {:.4f}, top 1 Accuracy: {:.2f}% top 5 Accuracy: {:.2f}% \n'.format(
            test_loss, acc_top1, acc_top5))

    output_log.write(
        '\nTest set: Average loss: {:.4f}, top 1 Accuracy: {:.2f}% top 5 Accuracy: {:.2f}% \n'.format(
            test_loss, acc_top1, acc_top5))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]

    return loss_per_epoch, [acc_top1], [acc_top5]



loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_workers=5, root_dir=args.data_path,
                                         num_class=args.num_class)

all_trainloader, noisy_labels = loader.run('train')
web_valloader = loader.run('test')
imagenet_valloader = loader.run('imagenet')



model = InceptionResNetV2(num_classes=args.num_class).to(args.gpuid)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
retrain = False

if args.lr_s == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
elif args.lr_s == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.001)

if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'NAL':
    criterion = NAL(noisy_labels, args.num_class, args.es, args.m, args.beta, args.delta)

exp_path = os.path.join('./',
                        '{0}_{1}_{2}_{3}_{4}_{5}_bs={6}_lam={7}_beta={8}_delta={9}_es={10}_momentum={11}'.format(
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
                            args.m), 'seed=' + str(args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

output_log = open(exp_path + '/log.txt', 'w')


t = torch.zeros(50000, args.num_class).to(args.gpuid)
cont = 0
best_results=[0.0]*4
for epoch in range(1, args.num_epochs + 1):
    print('\t##### Training Webvision #####')
    loss_train_per_epoch, acc_train_per_epoch = train(args.log_interval,
                                                      args.batch_size,
                                                      model,
                                                      args.gpuid,
                                                      all_trainloader,
                                                      optimizer,
                                                      epoch,)
    scheduler.step()

    print('On webvision:')
    output_log.write('On webvision:')
    output_log.flush()
    loss_per_epoch_webvision, acc_val_per_epoch_i_webvision, acc_val_per_epoch_i_topk_webvision = test_cleaning(
        args.batch_size, model, args.gpuid, web_valloader)
    print('On imagenet:')
    output_log.write('On imagenet:')
    output_log.flush()
    loss_per_epoch_imagenet, acc_val_per_epoch_i_imagenet, acc_val_per_epoch_i_topk_imagenet = test_cleaning(
        args.batch_size, model, args.gpuid, imagenet_valloader)
    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i_webvision[-1]
        best_acc_val_top5 = acc_val_per_epoch_i_topk_webvision[-1]
        snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccValTop1_%.5f_bestAccValTop5_%.5f' % (
            epoch, loss_per_epoch_webvision[-1], acc_val_per_epoch_i_webvision[-1], best_acc_val,
            best_acc_val_top5)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        _,corrected_labels=torch.max(criterion.soft_labels, dim=1)
        corrected_labels=corrected_labels.detach().cpu().numpy()
        np.save(os.path.join(exp_path, 'corrected_labels.npy'),corrected_labels)
        best_results[0] = acc_val_per_epoch_i_webvision[-1]
        best_results[1] = acc_val_per_epoch_i_topk_webvision[-1]
        best_results[2] = acc_val_per_epoch_i_imagenet[-1]
        best_results[3] = acc_val_per_epoch_i_topk_imagenet[-1]
    else:
        if acc_val_per_epoch_i_webvision[-1] > best_results[0]:
            best_results[0]=acc_val_per_epoch_i_webvision[-1]
        if acc_val_per_epoch_i_topk_webvision[-1] > best_results[1]:
            best_results[1] = acc_val_per_epoch_i_topk_webvision[-1]
        if acc_val_per_epoch_i_imagenet[-1] > best_results[2]:
            best_results[2] = acc_val_per_epoch_i_imagenet[-1]
        if acc_val_per_epoch_i_topk_imagenet[-1] > best_results[3]:
            best_results[3] = acc_val_per_epoch_i_topk_imagenet[-1]
        if acc_val_per_epoch_i_webvision[-1] > best_acc_val:
            best_acc_val = acc_val_per_epoch_i_webvision[-1]
            best_acc_val_top5 = acc_val_per_epoch_i_topk_webvision[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                except OSError:
                    pass
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccValTop1_%.5f_bestAccValTop5_%.5f' % (
                epoch, loss_per_epoch_webvision[-1], acc_val_per_epoch_i_webvision[-1], best_acc_val,
                best_acc_val_top5)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
            _, corrected_labels = torch.max(criterion.soft_labels, dim=1)
            corrected_labels = corrected_labels.detach().cpu().numpy()
            np.save(os.path.join(exp_path, 'corrected_labels.npy'), corrected_labels)
    cont += 1

print('\nValidation accuracy on webvision Top1 {:.2f}%, Top5 {:.2f}%; on imagenet Top1 {:.2f}%, Top5 {:.2f}%\n'.format(best_results[0],best_results[1],best_results[2],best_results[3]))
output_log.write('Validation accuracy on webvision Top1 {:.2f}%, Top5 {:.2f}%; on imagenet Top1 {:.2f}%, Top5 {:.2f}%\n'.format(best_results[0],best_results[1],best_results[2],best_results[3]))
output_log.flush()