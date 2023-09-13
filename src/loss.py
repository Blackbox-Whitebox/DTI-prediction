import torch
from torch import nn
from torch.nn.functional import cosine_similarity, normalize

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        (first, second) = pred

        first = normalize(first, p=2, dim=1) # l2 norm
        second = normalize(second, p=2, dim=1) # l2 norm 

        # (N, e) @ (e, N) -> (N, N)
        # 
        
        distance = torch.abs(cosine_similarity(first, second)) # (N)

        loss = .5 * (labels * (1-distance) + (1-labels) * distance)
        loss = loss.mean()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        labels = labels.long()

        return self.ce(pred, labels)