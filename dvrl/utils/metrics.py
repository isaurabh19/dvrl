import torch
import torch.nn.functional as F


class AccuracyTracker:
    def __init__(self):
        self.y_acc = []

    def track(self, y_true, logits):
        self.y_acc.append((torch.argmax(F.softmax(logits, dim=1), dim=1) == y_true).float())

    def compute(self):
        return torch.mean(torch.cat(self.y_acc))