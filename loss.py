import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NAL(torch.nn.Module):
    def __init__(self, labels, num_classes, es=60, momentum=0.9, beta=0.1, threshold_update=0.0):
        super(NAL, self).__init__()
        self.num_classes = num_classes
        # self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).to(1)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.es = es
        self.momentum = momentum
        self.beta = beta
        self.threshold_update = threshold_update

    def forward(self, confidence, logits, labels, index, lam, epoch):
        # sigmoid scale 0 to 1
        confidence = torch.sigmoid(confidence)

        output = F.softmax(logits, dim=1)
        eps = 1e-12
        output = torch.clamp(output, 0. + eps, 1. - eps)
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
        one_hot = torch.zeros(len(labels), self.num_classes)
        one_hot[torch.arange(len(labels)), labels] = 1
        one_hot = one_hot.to(device)
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)  # A=-4

        if epoch < self.es:
            pred = confidence * output + (1 - confidence) * one_hot
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            loss1 = -torch.mean(torch.sum(torch.log(pred) * one_hot, dim=1))
            rce = (-1 * torch.sum(pred * torch.log(one_hot), dim=1))
        else:
            if epoch % 10 == 0:
                temp_p = F.softmax(logits.detach(), dim=1)
                tp_f = confidence > self.threshold_update # only change the data has confidence >= threshold
                change_index = index[tp_f.view(tp_f.size()[0])]
                tp_f = tp_f.repeat(1, self.num_classes)
                self.soft_labels[change_index] = self.momentum * self.soft_labels[change_index] + (
                        1 - self.momentum) * temp_p[tp_f].view(-1, self.num_classes)
                self.soft_labels = torch.clamp(self.soft_labels, min=1e-4, max=1.0)
            pred = confidence * output + (1 - confidence) * self.soft_labels[index]
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            loss1 = -torch.mean(torch.sum(torch.log(pred) * self.soft_labels[index], dim=1))
            rce = (-1 * torch.sum(pred * torch.log(self.soft_labels[index]), dim=1))

        loss2 = -torch.mean(torch.log(confidence))
        return loss1 + lam * loss2 + self.beta * rce.mean()
