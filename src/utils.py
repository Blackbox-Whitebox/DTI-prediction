import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn.functional import cosine_similarity, normalize

import yaml


def write_to_tb(t, writer:SummaryWriter, net, scalars={}, histograms={}, embeddings={}):
    for tag in scalars.keys():
        writer.add_scalar(tag, scalar_value=scalars[tag], global_step=t)

    for tag in embeddings.keys():
        writer.add_embedding(mat=embeddings[tag], global_step=t, tag=tag)

    for tag in histograms.keys():
        writer.add_histogram(tag=tag, values=histograms[tag], global_step=t)

    for name, parameter in net.named_parameters():
        if parameter.requires_grad and not isinstance(parameter.grad, type(None)):
            writer.add_histogram(name, parameter, t)
            writer.add_histogram(f"{name}.grad", parameter.grad, t)

def parse_yaml(f):
    return yaml.safe_load(open(f, "r"))

def get_predictions_from_similarity(repr1, repr2, thresh=.5):
    repr1 = normalize(repr1, p=2, dim=1) # l2 norm
    repr2 = normalize(repr2, p=2, dim=1) # l2 norm 
    distance = torch.abs(cosine_similarity(repr1, repr2)) # (N)
    
    pred = (distance > thresh).long()

    return pred

def get_predictions_from_prob(prob, thresh=.5):
    return (prob > thresh).long()


def dataloader_collate_fn(data):
    """
    Pad (x1, x2) in data to same length in batch
    """

    x1, x2, y = zip(*data)

    y = torch.Tensor(list(y))

    lengths_x1 = [len(x) for x in x1]
    lengths_x2 = [len(x) for x in x2]

    max_len_x1 = max(lengths_x1)
    max_len_x2 = max(lengths_x2)

    batch_size = len(lengths_x1)

    x1_padded = torch.zeros(batch_size, max_len_x1).fill_(51)
    x2_padded = torch.zeros(batch_size, max_len_x2).fill_(21) # 21 is the <PAD> token in the amino acids vocab

    for i, seq in enumerate(x1):
        end = lengths_x1[i]
        x1_padded[i, :end] = seq[:end]

    for i, seq in enumerate(x2):
        end = lengths_x2[i]
        x2_padded[i, :end] = seq[:end]


    x1_padded = x1_padded.long()
    x2_padded = x2_padded.long()


    return x1_padded, x2_padded, y