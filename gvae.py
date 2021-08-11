import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, x_conv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dense_to_sparse, dropout_adj
from torch_geometric.nn import BatchNorm
from torch.nn import BatchNorm1d
from config import DEVICE as device
from config import SUPPORTED_ATOMS, ATOMIC_NUMBERS, MAX_MOLECULE_SIZE

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        encoder_embedding_size = 64
        decoder_size = 128
        edge_dim = 11
        self.latent_embedding_size = 16
        # no edge, single, aromatic, double, triple
        self.num_edge_types = 5 

        # Encoder layers
        self.conv1 = TransformerConv(feature_size, 
                                    encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn1 = BatchNorm(encoder_embedding_size)
        self.conv2 = TransformerConv(encoder_embedding_size, 
                                    encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn2 = BatchNorm(encoder_embedding_size)
        self.conv3 = TransformerConv(encoder_embedding_size, 
                                    encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn3 = BatchNorm(encoder_embedding_size)

        # Pooling layers
        self.pooling = Set2Set(encoder_embedding_size, processing_steps=3)

        # Latent transform layers
        self.mu_transform = Linear(encoder_embedding_size*2, 
                                            self.latent_embedding_size)
        self.logvar_transform = Linear(encoder_embedding_size*2, 
                                            self.latent_embedding_size)

        # Decoder layers
        # --- Atom decoding
        self.type_count = MAX_MOLECULE_SIZE
        self.num_atoms =  len(SUPPORTED_ATOMS)
        self.bag_of_atoms_1 = Linear(self.latent_embedding_size, self.num_atoms*self.type_count)
        self.bag_of_atoms_2 = Linear(self.num_atoms*self.type_count, self.num_atoms*self.type_count)
        # --- Fully connected MP 
        self.decode_conv1 = TransformerConv(self.latent_embedding_size + 1, 
                                            self.latent_embedding_size, 
                                            heads=4, 
                                            concat=False,
                                            beta=True)
        self.decode_conv2 = TransformerConv(self.latent_embedding_size, 
                                            self.latent_embedding_size, 
                                            heads=4, 
                                            concat=False,
                                            beta=True)
        self.decode_conv3 = TransformerConv(self.latent_embedding_size, 
                                            self.latent_embedding_size, 
                                            heads=4, 
                                            concat=False,
                                            beta=True)
        # --- Edge decoding
        self.decoder_dense_1 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_1 = BatchNorm1d(decoder_size)
        self.decoder_dense_2 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_2 = BatchNorm1d(decoder_size)
        self.decoder_dense_3 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_3 = BatchNorm1d(decoder_size)
        self.decoder_dense_4 = Linear(decoder_size, self.num_edge_types)
        

    def encode(self, x, edge_attr, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)
        return mu, logvar

    def bag_of_atoms_to_gnn_input(self, atom_types, graph_z):
        """
        Construct fully connected GNN input
        TODO: onehot required?
        """
        num_atoms = sum(atom_types)
        # Repeat each of the atom types #count times
        atomic_feats = torch.repeat_interleave(torch.Tensor(ATOMIC_NUMBERS), atom_types)
        latent_feats = graph_z.repeat(num_atoms, 1)
        x = torch.cat([atomic_feats.reshape(-1, 1), latent_feats], dim=1)        

        # Create fully connected adjacency matrix
        edge_index = dense_to_sparse(torch.ones(num_atoms, num_atoms))[0]
        # Reduce size of adjacency matrix
        edge_index = dropout_adj(edge_index, p=0.5)[0]

        return x, edge_index


    def decode_bag_of_atoms(self, graph_z, num_atoms):
        """
        Critical things:
        - Enforce that the number of atoms matches the graph size
        --> Not necessary --> Just always fetch the first X feats? 
        --> Ignore rest of matrix?
        """   
        if num_atoms > self.type_count:
            print("Too large molecule! Limit dataset!!")
            assert False

        # Predict molecular formula (count of each atom type)     
        atom_type_matrix = self.bag_of_atoms_1(graph_z)
        atom_type_matrix = self.bag_of_atoms_2(atom_type_matrix)
        atom_type_matrix = torch.reshape(atom_type_matrix, (self.num_atoms, self.type_count))
        atom_type_counts = torch.argmax(atom_type_matrix, dim=1)

        # If predicted num is smaller, fill with carbon
        if num_atoms > sum(atom_type_counts):
            diff = num_atoms - sum(atom_type_counts)
            atom_type_counts[0] += diff
        return atom_type_counts

    def decode(self, z, batch_index):
        inputs = []
        # Iterate over molecules in batch
        for graph_id in torch.unique(batch_index):
            # Get latent vector for this graph
            graph_z = z[graph_id]

            # Recover atom types
            num_atoms = torch.sum(torch.eq(batch_index, graph_id))
            atom_types = self.decode_bag_of_atoms(graph_z, num_atoms)

            # Construct inputs for Decoder GNN
            x, edge_index = self.bag_of_atoms_to_gnn_input(atom_types, graph_z)

            # Message passing
            x = self.decode_conv1(x, edge_index).relu()
            x = self.decode_conv2(x, edge_index).relu()
            x = self.decode_conv3(x, edge_index).relu()

            # Get indices for triangular upper part of adjacency matrix
            graph_mask = torch.eq(batch_index, graph_id)
            edge_indices = torch.triu_indices(num_atoms, num_atoms, offset=1)

            # Repeat indices to match dim of latent codes
            dim = self.latent_embedding_size
            source_indices = torch.reshape(edge_indices[0].repeat_interleave(dim), (edge_indices.shape[1], dim))
            target_indices = torch.reshape(edge_indices[1].repeat_interleave(dim), (edge_indices.shape[1], dim))

            # Gather features
            sources_feats = torch.gather(x, 0, source_indices.to(device))
            target_feats = torch.gather(x, 0, target_indices.to(device))

            # Concatenate inputs of all source and target nodes
            graph_inputs = torch.cat([sources_feats, target_feats], axis=1)
            inputs.append(graph_inputs)

        # Concatenate all inputs of all graphs in the batch
        inputs = torch.cat(inputs)

        # Get predictions
        x = self.decoder_dense_1(inputs).relu()
        x = self.decoder_bn_1(x)
        x = self.decoder_dense_2(inputs).relu()
        x = self.decoder_bn_2(x)
        x = self.decoder_dense_3(inputs).relu()
        x = self.decoder_bn_3(x)
        edge_logits = self.decoder_dense_4(x)

        return edge_logits


    def reparameterize(self, mu, logvar):
        """
        The reparametrization trick is required to 
        backpropagate through the network.
        We cannot backpropagate through a "sampled"
        node as it is not deterministic.
        The trick is to separate the randomness
        from the network.
        """
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
        triu_logits = self.decode(z, batch_index)

        return triu_logits, mu, logvar