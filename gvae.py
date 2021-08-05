import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dense_to_sparse, to_dense_adj, remove_self_loops
from torch_geometric.nn import BatchNorm
from torch.nn import BatchNorm1d
from config import DEVICE as device

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        encoder_embedding_size = 32
        self.latent_embedding_size = 16
        decoder_size = 128
        edge_dim = 11

        # Encoder layers
        self.conv1 = TransformerConv(feature_size, 
                                    encoder_embedding_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn1 = BatchNorm(encoder_embedding_size)
        self.conv2 = TransformerConv(encoder_embedding_size, 
                                    encoder_embedding_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn2 = BatchNorm(encoder_embedding_size)
        self.conv3 = TransformerConv(encoder_embedding_size, 
                                    encoder_embedding_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim)
        self.bn3 = BatchNorm(encoder_embedding_size)

        # Latent transform layers
        self.mu_transform = TransformerConv(encoder_embedding_size, 
                                            self.latent_embedding_size,
                                            heads=3,
                                            concat=False,
                                            beta=True,
                                            edge_dim=edge_dim)
        self.logvar_transform = TransformerConv(encoder_embedding_size, 
                                            self.latent_embedding_size,
                                            heads=3,
                                            concat=False,
                                            beta=True,
                                            edge_dim=edge_dim)

        # Decoder layers
        self.decoder_dense_1 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_1 = BatchNorm1d(decoder_size)
        self.decoder_dense_2 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_2 = BatchNorm1d(decoder_size)
        self.decoder_dense_3 = Linear(self.latent_embedding_size*2, decoder_size)
        self.decoder_bn_3 = BatchNorm1d(decoder_size)
        self.decoder_dense_4 = Linear(decoder_size, 1)
        

    def encode(self, x, edge_attr, edge_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)

        # Latent transform layers
        mu = self.mu_transform(x, edge_index, edge_attr)
        logvar = self.logvar_transform(x, edge_index, edge_attr)

        return mu, logvar


    def decode(self, z, batch_index):
        """
        Takes n latent vectors (one per node) and decodes them
        into the upper triangular part of their adjacency matrix.
        """
        inputs = []

        # Iterate over molecules in batch
        for graph_id in torch.unique(batch_index):
            graph_mask = torch.eq(batch_index, graph_id)
            graph_z = z[graph_mask]

            # Get indices for triangular upper part of adjacency matrix
            edge_indices = torch.triu_indices(graph_z.shape[0], graph_z.shape[0], offset=1)

            # Repeat indices to match dim of latent codes
            dim = self.latent_embedding_size
            source_indices = torch.reshape(edge_indices[0].repeat_interleave(dim), (edge_indices.shape[1], dim))
            target_indices = torch.reshape(edge_indices[1].repeat_interleave(dim), (edge_indices.shape[1], dim))

            # Gather features
            sources_feats = torch.gather(graph_z, 0, source_indices.to(device))
            target_feats = torch.gather(graph_z, 0, target_indices.to(device))

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
        mu, logvar = self.encode(x, edge_attr, edge_index)
        # Sample latent vector (per atom)
        z = self.reparameterize(mu, logvar)
        # Decode latent vector into original molecule
        triu_logits = self.decode(z, batch_index)

        return triu_logits, mu, logvar