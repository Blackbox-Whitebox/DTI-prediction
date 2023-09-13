import torch
import matplotlib.pyplot as plt

from models.pretraining import GRUModel
from models.transformer import Transformer

from vocab import ProteinVocab, SMILEVocab

from data import DTIDataset
from utils import dataloader_collate_fn

from tqdm import tqdm

from torch.utils.data import DataLoader

from ignite.contrib.metrics import ROC_AUC, PrecisionRecallCurve

PROTEIN_GRU_CHECKPOINT = "../checkpoints/pretraining/protein_gru.pth"
SMILE_GRU_CHECKPOINT = "../checkpoints/pretraining/smile_gru.pth"
TRANSFORMER_CHECKPOINT = "../checkpoints/classification/transformer-classification-baseline/model.pt"


amino_vocab = ProteinVocab()
smile_vocab = SMILEVocab()

amino_embedding = GRUModel(
    vocab_size=len(amino_vocab),
    embed_size=64,
    hidden_size=128,
    num_layers=1,
    dropout=0,
    bidirectional=False
)

smile_embedding = GRUModel(
    vocab_size=len(smile_vocab),
    embed_size=64,
    hidden_size=128,
    num_layers=1,
    dropout=0,
    bidirectional=False
)


checkpoint = torch.load(PROTEIN_GRU_CHECKPOINT, map_location="cpu")
amino_embedding.load_state_dict(checkpoint)
amino_embedding = amino_embedding.embedding

checkpoint = torch.load(SMILE_GRU_CHECKPOINT, map_location="cpu")
smile_embedding.load_state_dict(checkpoint)
smile_embedding = smile_embedding.embedding

amino_embedding.requires_grad_(False)
smile_embedding.requires_grad_(False)

transformer_model = Transformer(
    hidden_size=256,
    n_head=4,
    num_layers=2,
    dropout=.2,
    mode="classification"
)

transformer_model.set_drug_protein_embeddings(
    smile_embedding,
    amino_embedding
)

checkpoint = torch.load(TRANSFORMER_CHECKPOINT, map_location="cpu")
print(checkpoint["loss"], checkpoint["accuracy"], checkpoint["epoch"])
transformer_model.load_state_dict(checkpoint["model"])

transformer_model.eval()
transformer_model.requires_grad_(False)


data = DTIDataset(
    train=False,
    smile_vocab=smile_vocab,
    protein_vocab=amino_vocab,
    smile_embedding=smile_embedding,
    amino_embedding=amino_embedding,
    device="cpu"
)


data = DataLoader(
    data,
    batch_size=64,
    collate_fn=dataloader_collate_fn
)

roc_auc = ROC_AUC()

with torch.no_grad():
    for x1, x2, y in tqdm(data):
        p = transformer_model((x1, x2)).softmax(-1).softmax(-1) # (N, 2)
        p = p[:, 1] # get prob of being 1
        roc_auc.update((p, y))

roc_auc_value = roc_auc.compute()

print(roc_auc_value)