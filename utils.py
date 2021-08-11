from torch_geometric.utils import to_dense_adj
import torch
from rdkit import Chem
from config import DEVICE as device
import numpy as np
from config import SUPPORTED_ATOMS, ATOMIC_NUMBERS

def count_parameters(model):
    """
    Counts the number of parameters for a Pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd =  logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div

def slice_graph_targets(graph_id, batch_targets, batch_index):
    """
    Slices out the upper triangular part of an adjacency matrix for
    a single graph from a large adjacency matrix for a full batch.
    --------
    graph_id: The ID of the graph (in the batch index) to slice
    batch_targets: A dense adjacency matrix for the whole batch
    batch_index: The node to graph map for the batch
    """
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph targets
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
    triu_indices = torch.triu_indices(graph_targets.shape[0], graph_targets.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    return graph_targets[triu_mask]

def slice_graph_predictions(triu_logits, graph_triu_size, start_point):
    """
    Slices out the corresponding section from a list of batch triu values.
    Given a start point and the size of a graph's triu, simply slices
    the section from the batch list.
    -------
    triu_logits: A batch of triu predictions of different graphs
    graph_triu_size: Size of the triu of the graph to slice
    start_point: Index of the first node of this graph
    """
    graph_logits_triu = torch.squeeze(
                    triu_logits[start_point:start_point + graph_triu_size]
                    )  
    return graph_logits_triu

def slice_node_features(graph_id, node_features, batch_index):
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph targets
    graph_node_features = node_features[graph_mask]
    return graph_node_features

def slice_edge_type_from_edge_feats(edge_feats):
    """
    This function only works for the MolGraphConvFeaturizer used in the dataset.
    It slices the one-hot encoded edge type from the edge feature matrix.
    The first 4 values stand for ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]. 
    """
    edge_types_one_hot = edge_feats[:, :4]
    edge_types = edge_types_one_hot.nonzero(as_tuple=False)
    # Start index at 1, zero will be no edge
    edge_types[:, 1] = edge_types[:, 1] + 1
    return edge_types


def slice_atom_type_from_node_features(node_features):
    """
    This function only works for the MolGraphConvFeaturizer used in the dataset.
    It slices the one-hot encoded atom type from the node feature matrix.
    Unknown atom types will be decoded with -1.
    """
    supported_atoms = SUPPORTED_ATOMS
    atomic_numbers =  ATOMIC_NUMBERS

    # Slice first X entries from the node feature matrix
    atom_types_one_hot = node_features[:, :len(supported_atoms)]
    # Map the index to the atomic number
    atom_numbers_dummy = torch.Tensor(atomic_numbers).repeat(atom_types_one_hot.shape[0], 1)
    atom_types = torch.masked_select(atom_numbers_dummy, atom_types_one_hot.bool())
    return atom_types

def check_triu_graph_reconstruction(graph_predictions_triu, graph_targets_triu, node_features, num_nodes=None):
    """
    Checks if the triu adjacency matrix prediction matches the ground-truth of the graph 
    """
    # Apply softmax to get class prediction 
    # TODO: Move that to the network?
    preds = torch.softmax(graph_predictions_triu, dim=1).max(dim=1).indices
    # Reshape the targets
    labels = graph_targets_triu.view(-1)
    # Check if the predictions and the groundtruth match
    if labels.shape[0] == sum(torch.eq(preds, labels)):
        pos_edges = sum(~torch.eq(labels, 0))
        atom_types = slice_atom_type_from_node_features(node_features)
        # Check if this molecule contains valid atoms
        if -1 in atom_types:
            print(f"Successfully reconstructed with {pos_edges} pos. edges but unsupported nodes.")
        else:
            smiles, mol = graph_representation_to_molecule(atom_types, preds, num_nodes)
            print(f"Successfully reconstructed with {pos_edges} pos. edges and {num_nodes} nodes.")
        return True
    return False    

def gvae_loss(triu_logits, edge_index, edge_types, mu, logvar, batch_index, kl_beta):
    """
    Calculates a weighted ELBO loss for a batch of graphs for the graph
    variational autoencoder model.
    """
    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))

    # Add edge types to adjacency targets
    batch_targets[edge_index[0], edge_index[1]] = edge_types[:, 1].float()

    # Reconstruction loss per graph
    batch_recon_loss = []
    batch_node_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
            # Get upper triangular targets for this graph from the whole batch
            graph_targets_triu = slice_graph_targets(graph_id, 
                                                    batch_targets, 
                                                    batch_index)

            # Get upper triangular predictions for this graph from the whole batch
            graph_predictions_triu = slice_graph_predictions(triu_logits, 
                                                            graph_targets_triu.shape[0], 
                                                            batch_node_counter)
            
            # Update counter to the index of the next graph
            batch_node_counter = batch_node_counter + graph_targets_triu.shape[0]

            # Calculate edge-weighted binary cross entropy
            num_elements = graph_targets_triu.shape[0]
            uniques, per_class_counts = torch.unique(graph_targets_triu, return_counts=True)
            weight = num_elements/per_class_counts.float()

            # Add missing edge type if not in graph (always same shape)
            missing_elements = np.setdiff1d([0,1,2,3,4], uniques)
            if missing_elements.shape[0] > 0:
                for i in missing_elements:
                    weight = torch.cat([weight[:i], torch.Tensor([0]), weight[i:]], 0)

            ce = torch.nn.CrossEntropyLoss(weight=weight).to(device)
            graph_recon_loss = ce(graph_predictions_triu, graph_targets_triu.view(-1).long())
            batch_recon_loss.append(graph_recon_loss)   
    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / num_graphs
    
    # KL Divergence
    kl_divergence = kl_loss(mu, logvar)

    return batch_recon_loss , kl_divergence #+ kl_beta * kl_divergence


def reconstruction_accuracy(triu_logits, edge_index, edge_types, batch_index, node_features):
    # Convert edge index to adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    # Add edge types to adjacency targets
    batch_targets[edge_index[0], edge_index[1]] = edge_types[:, 1].float()
    # Store target trius
    batch_targets_triu = []
    # Iterate over batch and collect each of the trius
    batch_node_counter = 0
    num_recon = 0
    for graph_id in torch.unique(batch_index):
            # Get triu parts for this graph
            graph_targets_triu = slice_graph_targets(graph_id, 
                                                    batch_targets, 
                                                    batch_index)
            graph_predictions_triu = slice_graph_predictions(triu_logits, 
                                                            graph_targets_triu.shape[0], 
                                                            batch_node_counter)

            # Update counter to the index of the next graph
            batch_node_counter = batch_node_counter + graph_targets_triu.shape[0]

            # Slice node features of this batch
            graph_node_features = slice_node_features(graph_id, node_features, batch_index)

            # Check if graph is successfully reconstructed
            num_nodes = sum(torch.eq(batch_index, graph_id))
            recon = check_triu_graph_reconstruction(graph_predictions_triu, 
                                                    graph_targets_triu, 
                                                    graph_node_features, num_nodes) 
            num_recon = num_recon + int(recon)

            # Add targets to triu list
            batch_targets_triu.append(graph_targets_triu)
    # Calculate accuracy between predictions and labels
    batch_targets_triu = torch.cat(batch_targets_triu).detach().cpu()
    triu_discrete = torch.softmax(triu_logits, dim=1).max(dim=1).indices
    acc = torch.true_divide(torch.sum(batch_targets_triu==triu_discrete), batch_targets_triu.shape[0]) 
    return acc.detach().cpu().numpy(), num_recon


def triu_to_dense(triu_values, num_nodes):
    dense_adj = torch.zeros((num_nodes, num_nodes)).to(device).int()
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
    dense_adj[triu_indices[0], triu_indices[1]] = triu_values.int()
    dense_adj[tril_indices[0], tril_indices[1]] = triu_values.int()
    return dense_adj

def graph_representation_to_molecule(node_types, adjacency_triu, num_nodes):
    # Create empty mol
    mol = Chem.RWMol()

    # Add atoms to mol and store their index
    node_to_idx = {}
    for i in range(len(node_types)):
        a = Chem.Atom(int(node_types[i]))
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    
    # Add edges to mol
    adjacency_matrix = triu_to_dense(adjacency_triu, num_nodes)
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue

            # add bonds
            if bond == 0:
                continue
            else:
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
    # Convert RWMol to mol and Smiles
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)

    # Sanitize molecule (make sure it is valid)
    try:
        Chem.SanitizeMol(mol)
    except:
        print("Sanitization failed for this molecule.")

    # TODO: Visualize and save
    print(smiles)

    return smiles, mol