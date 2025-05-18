import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered."""
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in processed_dir, processing is skipped."""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        # Not implemented because data is assumed to be already downloaded.
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                print(f"Invalid SMILES at index {index}: {row['smiles']}")
                continue

            try:
                f = featurizer._featurize(mol)
                data = f.to_pyg_graph()
                data.y = self._get_label(row["HIV_active"])
                data.smiles = row["smiles"]

                filename = f'data_test_{index}.pt' if self.test else f'data_{index}.pt'
                torch.save(data, os.path.join(self.processed_dir, filename))
            except Exception as e:
                print(f"Featurization failed at index {index}: {e}")
                continue

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """Equivalent to __getitem__ in PyTorch"""
        filename = f'data_test_{idx}.pt' if self.test else f'data_{idx}.pt'
        return torch.load(os.path.join(self.processed_dir, filename))

