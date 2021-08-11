import torch

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# Supported atoms 
SUPPORTED_ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
ATOMIC_NUMBERS =  [6, 7, 8, 9, 15, 16, 17, 35, 53, -1]

# Dataset (if you change this, delete the processed files to run again)
MAX_MOLECULE_SIZE = 30