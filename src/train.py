import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import ContrastiveLoss, CrossEntropyLoss
from models import build
from models.pretraining import GRUModel
from tqdm import tqdm
from data import DTIDataset
from vocab import ProteinVocab, SMILEVocab
import utils
import numpy as np
from trainer import Trainer
from metrics import accuracy_from_contrastive_model, accuracy_from_classification_model


def train(args):
    config_file = utils.parse_yaml(args.config)

    if os.path.exists(f"../logs/{config_file['RUN_NAME']}"):
        raise ValueError(f"This run has happened before")
    
    os.makedirs(f"../checkpoints/classification/{config_file['RUN_NAME']}")
    
    np.random.seed(int(config_file["SEED"]))
    torch.manual_seed(int(config_file["SEED"]))

    writer = SummaryWriter(log_dir=f"../logs/{config_file['RUN_NAME']}")

    amino_vocab = ProteinVocab()
    smile_vocab = SMILEVocab()

    pretrained_smile_embeddings = GRUModel(
        vocab_size=len(smile_vocab),
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        dropout=0,
        bidirectional=False
    ).to(args.device)

    pretrained_amino_embeddings = GRUModel(
        vocab_size=len(amino_vocab),
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        dropout=0,
        bidirectional=False
    ).to(args.device)

    pretrained_smile_embeddings.load_state_dict(torch.load("../checkpoints/pretraining/smile_gru.pth"))
    pretrained_amino_embeddings.load_state_dict(torch.load("../checkpoints/pretraining/protein_gru.pth"))

    pretrained_smile_embeddings = pretrained_smile_embeddings.embedding
    pretrained_amino_embeddings = pretrained_amino_embeddings.embedding


    net = build(config_file)

    net.set_drug_protein_embeddings(
        pretrained_smile_embeddings.to(args.device),
        pretrained_amino_embeddings.to(args.device)
        
    )

    net.to(args.device)

    train = DTIDataset(
        train=True,
        smile_vocab=smile_vocab,
        protein_vocab=amino_vocab,
        smile_embedding=pretrained_smile_embeddings,
        amino_embedding=pretrained_amino_embeddings,
        device=args.device
    )

    test = DTIDataset(
        train=False,
        smile_vocab=smile_vocab,
        protein_vocab=amino_vocab,
        smile_embedding=pretrained_smile_embeddings,
        amino_embedding=pretrained_amino_embeddings,
        device=args.device
    )

    train = DataLoader(train, int(config_file["BATCH_SIZE"]), shuffle=True, collate_fn=utils.dataloader_collate_fn)
    test = DataLoader(test, int(config_file["BATCH_SIZE"]), shuffle=True, collate_fn=utils.dataloader_collate_fn)

    if bool(config_file.get("CONTRASTIVE")):
        lossfn = ContrastiveLoss()
        
    else:
        lossfn = CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=float(config_file["LR"]))

    trainer = Trainer(
        model=net,
        batch_size=int(config_file["BATCH_SIZE"]),
        train=train,
        test=test,
        epochs=config_file["EPOCHS"],
        optimizer=optimizer,
        lossfn=lossfn,
        metrics={
            "Accuracy": accuracy_from_contrastive_model if bool(config_file.get("CONTRASTIVE")) else accuracy_from_classification_model
        },
        writer=writer,
        smooth=.6,
        device="cuda",
        config_file=config_file
    )

    trainer.run()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train(args)