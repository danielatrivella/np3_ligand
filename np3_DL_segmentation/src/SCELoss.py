# Code taken from https://github.com/HanxunH/SCELoss-Reproduce
# implementation in pyTorch - official repo using tensorflow  https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
# article: ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" https://arxiv.org/abs/1908.06112
import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, device, alpha=0.1, beta=5.0, num_classes=10, weight=None):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        if weight is None:
            self.cross_entropy = torch.nn.CrossEntropyLoss()
        else:
            weight = torch.Tensor(weight)
            weight = weight.to(device)
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        # if device is cuda only the index is being passed
        if not hasattr(device, 'type') or device.type != 'cpu':
            self.cross_entropy = self.cross_entropy.cuda(device)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
