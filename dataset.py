import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
from tqdm import tqdm
import deepchem as dc
from config import MAX_MOLECULE_SIZE
from utils import slice_atom_type_from_node_feats
import re

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, length=0):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.length = length
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        processed_files = [f for f in os.listdir(self.processed_dir) if not f.startswith("pre")]
    
        if self.test:
            processed_files = [file for file in processed_files if "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index
            return [f'data_test_{i}.pt' for i in list(range(0, index))]
        else:
            processed_files = [file for file in processed_files if not "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index
            return [f'data_{i}.pt' for i in list(range(0, index))]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for _, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            f = featurizer.featurize(mol["smiles"])
            data = f[0].to_pyg_graph()
            data.y = self._get_label(mol["HIV_active"])
            data.smiles = mol["smiles"]

            # Get the molecule's atom types
            atom_types = slice_atom_type_from_node_feats(data.x)

            # Only save if molecule is in permitted size
            if (data.x.shape[0] < MAX_MOLECULE_SIZE) and -1 not in atom_types:
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{self.length}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{self.length}.pt'))
                self.length += 1
            else:
                print("Skipping invalid mol (too big/unknown atoms): ", data.smiles)
        print(f"Done. Stored {self.length} preprocessed molecules.")

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        """ 
        - Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data