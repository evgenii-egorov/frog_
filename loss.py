import torch
import torch.nn as nn
import abc

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


class Loss(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Loss, self).__init__()

    @abc.abstractmethod
    def forward(self, logits, labels):
        pass

    def get_metrics(self, logits, labels):
        # C_ij, i #true labels. j #predictions
        metrics = {}
        b_size = logits.shape[0]

        logits = logits.reshape(b_size, 16)
        _, prediction = torch.max(logits, 1)
        prediction = prediction.data.cpu().numpy()

        labels = labels.data.cpu().numpy()

        confusion = confusion_matrix(labels, prediction, labels=range(16))
        N = np.sum(confusion)
        acc = np.sum(np.diag(confusion)) / N
        std_acc = np.sqrt(acc * (1.- acc) / N)

        metrics['acc'] = acc
        metrics['std_acc'] = std_acc
        metrics['confusion'] = confusion

        plt.tight_layout()
        fig, ax = plt.subplots(1,1, figsize=(32, 32))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=range(16))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        metrics['fig_confusion'] = fig
        plt.close(fig)

        return metrics


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, labels):
        b_size = logits.shape[0]
        logits = logits.reshape(b_size, 16)
        return self.loss(logits, labels)


class FocalLoss(Loss):
    def __init__(self, gamma=2.):
        super().__init__()
        self.loss = nn.LogSoftmax(dim=-1)
        self.gamma = gamma

    def forward(self, logits, labels):
        b_size = logits.shape[0]
        logits = logits.reshape(b_size, 16)

        log_p = self.loss(logits).gather(1, labels.view(-1, 1)).view(-1)
        p = torch.exp(log_p)
        loss = -1 * (1 - p) ** self.gamma * log_p
        return loss.mean()


