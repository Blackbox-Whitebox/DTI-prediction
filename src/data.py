import json
import torch
from torch.utils.data import Dataset
import pandas as pd


class DTIDataset(Dataset):
    def __init__(self, train, smile_vocab, protein_vocab, smile_embedding, amino_embedding, device="cuda"):
        self.device = device
        if train:
            self.df = pd.read_csv("../data/combined_train.csv")
        else:
            self.df = pd.read_csv("../data/combined_test.csv")

        self.df.dropna(inplace=True)

        self.train = train
        self.smile_vocab = smile_vocab
        self.protein_vocab = protein_vocab
        self.smile_embedding = smile_embedding
        self.amino_embedding = amino_embedding

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        drug = row["Drug"]
        target = row["Target"]
        y = row["Y"]
        
        drug = torch.Tensor(self.smile_vocab.encode(drug)).long().to(self.device) # get tokenized
        # drug = self.smile_embedding(drug).mean(1).squeeze() # get embeddings
        
        target = torch.Tensor(self.protein_vocab.encode(target)).long().to(self.device) # get tokenized
        # target = self.amino_embedding(target).mean(1).squeeze() # get embeddings
        

        drug = drug.to(self.device)
        target = target.to(self.device)[:3000] # the amino acids sequences are usually unreasonably long. So we truncate for feasible processing times.
        
        return drug, target, y