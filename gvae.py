import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import BatchNorm
from config import SUPPORTED_ATOMS, SUPPORTED_EDGES, MAX_MOLECULE_SIZE, ATOMIC_NUMBERS
from utils import graph_representation_to_molecule, to_one_hot
from tqdm import tqdm

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        self.encoder_embedding_size = 64
        self.edge_dim = 11
        self.latent_embedding_size = 128
        self.num_edge_types = len(SUPPORTED_EDGES) 
        self.num_atom_types = len(SUPPORTED_ATOMS)
        self.max_num_atoms = MAX_MOLECULE_SIZE 
        self.decoder_hidden_neurons = 512

        # Encoder layers
        self.conv1 = TransformerConv(feature_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.conv2 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.conv3 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        self.conv4 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)

        # Latent transform layers
        self.mu_transform = Linear(self.encoder_embedding_size*2, 
                                            self.latent_embedding_size)
        self.logvar_transform = Linear(self.encoder_embedding_size*2, 
                                            self.latent_embedding_size)

        # Decoder layers
        # --- Shared layers
        self.linear_1 = Linear(self.latent_embedding_size, self.decoder_hidden_neurons)
        self.linear_2 = Linear(self.decoder_hidden_neurons, self.decoder_hidden_neurons)

        # --- Atom decoding (outputs a matrix: (max_num_atoms) * (# atom_types + "none"-type))   
        atom_output_dim = self.max_num_atoms*(self.num_atom_types + 1)
        self.atom_decode = Linear(self.decoder_hidden_neurons, atom_output_dim)

        # --- Edge decoding (outputs a triu tensor: (max_num_atoms*(max_num_atoms-1)/2*(#edge_types + 1) ))
        edge_output_dim = int(((self.max_num_atoms * (self.max_num_atoms - 1)) / 2) * (self.num_edge_types + 1))
        self.edge_decode = Linear(self.decoder_hidden_neurons, edge_output_dim)
        

    def encode(self, x, edge_attr, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_attr).relu()

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)
        return mu, logvar

    def decode_graph(self, graph_z):  
        """
        Decodes a latent vector into a continuous graph representation
        consisting of node types and edge types.
        """
        # Pass through shared layers
        z = self.linear_1(graph_z).relu()
        z = self.linear_2(z).relu()
        # Decode atom types
        atom_logits = self.atom_decode(z)
        # Decode edge types
        edge_logits = self.edge_decode(z)

        return atom_logits, edge_logits


    def decode(self, z, batch_index):
        node_logits = []
        triu_logits = []
        # Iterate over molecules in batch
        for graph_id in torch.unique(batch_index):
            # Get latent vector for this graph
            graph_z = z[graph_id]

            # Recover graph from latent vector
            atom_logits, edge_logits = self.decode_graph(graph_z)

            # Store per graph results
            node_logits.append(atom_logits)
            triu_logits.append(edge_logits)

        # Concatenate all outputs of the batch
        node_logits = torch.cat(node_logits)
        triu_logits = torch.cat(triu_logits)
        return triu_logits, node_logits


    def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Encode the molecule
        mu, logvar = self.encode(x, edge_attr, edge_index, batch_index)
        # Sample latent vector (per atom)
        z = self.reparameterize(mu, logvar)
        # Decode latent vector into original molecule
        triu_logits, node_logits = self.decode(z, batch_index)

        return triu_logits, node_logits, mu, logvar

    
    def sample_mols(self, num=10000):
        print("Sampling molecules ... ")

        n_valid = 0
        # Sample molecules and check if they are valid
        for _ in tqdm(range(num)):
            # Sample latent space
            z = torch.randn(1, self.latent_embedding_size)

            # Get model output (this could also be batched)
            dummy_batch_index = torch.Tensor([0]).int()
            triu_logits, node_logits = self.decode(z, dummy_batch_index)

            # Reshape triu predictions 
            edge_matrix_shape = (int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1))/2), len(SUPPORTED_EDGES) + 1) 
            triu_preds_matrix = triu_logits.reshape(edge_matrix_shape)
            triu_preds = torch.argmax(triu_preds_matrix, dim=1)

            # Reshape node predictions
            node_matrix_shape = (MAX_MOLECULE_SIZE, (len(SUPPORTED_ATOMS) + 1)) 
            node_preds_matrix = node_logits.reshape(node_matrix_shape)
            node_preds = torch.argmax(node_preds_matrix[:, :9], dim=1)
            
            # Get atomic numbers 
            node_preds_one_hot = to_one_hot(node_preds, options=ATOMIC_NUMBERS)
            atom_numbers_dummy = torch.Tensor(ATOMIC_NUMBERS).repeat(node_preds_one_hot.shape[0], 1)
            atom_types = torch.masked_select(atom_numbers_dummy, node_preds_one_hot.bool())

            # Attempt to create valid molecule
            smiles, _ = graph_representation_to_molecule(atom_types, triu_preds.float())

            # A dot means disconnected
            if smiles and "." not in smiles:
                print("Successfully generated: ", smiles)
                n_valid += 1    
        return n_valid