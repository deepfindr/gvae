from torch_geometric.utils import to_dense_adj
import torch
from rdkit import Chem
from rdkit import RDLogger 
from config import DEVICE as device
from config import (SUPPORTED_ATOMS, SUPPORTED_EDGES, MAX_MOLECULE_SIZE, ATOMIC_NUMBERS,
                    DISABLE_RDKIT_WARNINGS)

# Disable rdkit warnings
if DISABLE_RDKIT_WARNINGS:
    RDLogger.DisableLog('rdApp.*')  

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

def slice_graph_targets(graph_id, edge_targets, node_targets, batch_index):
    """
    Slices out the upper triangular part of an adjacency matrix for
    a single graph from a large adjacency matrix for a full batch.
    For the node features the corresponding section in the batch is sliced out.
    --------
    graph_id: The ID of the graph (in the batch index) to slice
    edge_targets: A dense adjacency matrix for the whole batch
    node_targets: A tensor of node labels for the whole batch
    batch_index: The node to graph map for the batch
    """
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph edge targets
    graph_edge_targets = edge_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
    size = graph_edge_targets.shape[0]
    triu_indices = torch.triu_indices(size, size, offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    graph_edge_targets = graph_edge_targets[triu_mask]
    # Slice node targets
    graph_node_targets = node_targets[graph_mask]
    return graph_edge_targets, graph_node_targets

def slice_graph_predictions(triu_logits, node_logits, graph_triu_size, triu_start_point, graph_size, node_start_point):
    """
    Slices out the corresponding section from a list of batch triu values.
    Given a start point and the size of a graph's triu, simply slices
    the section from the batch list.
    -------
    triu_logits: A batch of triu predictions of different graphs
    node_logits: A batch of node predictions with fixed size MAX_GRAPH_SIZE
    graph_triu_size: Size of the triu of the graph to slice
    triu_start_point: Index of the first node of this graph in the triu batch
    graph_size: Max graph size
    node_start_point: Index of the first node of this graph in the nodes batch
    """
    # Slice edge logits
    graph_logits_triu = torch.squeeze(
                    triu_logits[triu_start_point:triu_start_point + graph_triu_size]
                    )  
    # Slice node logits
    graph_node_logits = torch.squeeze(
                    node_logits[node_start_point:node_start_point + graph_size]
                    ) 
    return graph_logits_triu, graph_node_logits


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


def slice_atom_type_from_node_feats(node_features, as_index=False):
    """
    This function only works for the MolGraphConvFeaturizer used in the dataset.
    It slices the one-hot encoded atom type from the node feature matrix.
    Unknown atom types are not considered and not expected in the datset.
    """
    supported_atoms = SUPPORTED_ATOMS
    atomic_numbers =  ATOMIC_NUMBERS

    # Slice first X entries from the node feature matrix
    atom_types_one_hot = node_features[:, :len(supported_atoms)]
    if not as_index:
        # Map the index to the atomic number
        atom_numbers_dummy = torch.Tensor(atomic_numbers).repeat(atom_types_one_hot.shape[0], 1)
        atom_types = torch.masked_select(atom_numbers_dummy, atom_types_one_hot.bool())
    else:
        atom_types = torch.argmax(atom_types_one_hot, dim=1)
    return atom_types

def to_one_hot(x, options):
    """
    Converts a tensor of values to a one-hot vector
    based on the entries in options.
    """
    return torch.nn.functional.one_hot(x.long(), len(options))

def squared_difference(input, target):
    return (input - target) ** 2


def triu_to_dense(triu_values, num_nodes):
    """
    Converts a triangular upper part of a matrix as flat vector
    to a squared adjacency matrix with a specific size (num_nodes).
    """
    dense_adj = torch.zeros((num_nodes, num_nodes)).to(device).float()
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
    dense_adj[triu_indices[0], triu_indices[1]] = triu_values
    dense_adj[tril_indices[0], tril_indices[1]] = triu_values
    return dense_adj


def triu_to_3d_dense(triu_values, num_nodes, depth=len(SUPPORTED_EDGES)):
    """
    Converts the triangular upper part of a matrix
    for several dimensions into a 3d tensor.
    """
    # Create placeholder for 3d matrix
    adj_matrix_3d = torch.empty((num_nodes, num_nodes, depth), dtype=torch.float, device=device)
    for edge_type in range(len(SUPPORTED_EDGES)):
        adj_mat_edge_type = triu_to_dense(triu_values[:, edge_type].float(), num_nodes)
        adj_matrix_3d[:, :, edge_type] = adj_mat_edge_type
    return adj_matrix_3d

def calculate_node_edge_pair_loss(node_tar, edge_tar, node_pred, edge_pred):
    """
    Calculates a loss based on the sum of node-edge pairs.
    node_tar:  [nodes, supported atoms]
    node_pred: [max nodes, supported atoms + 1]
    edge_tar:  [triu values for target nodes, supported edges]
    edge_pred: [triu values for predicted nodes, supported edges]

    """
    # Recover full 3d adjacency matrix for edge predictions
    edge_pred_3d = triu_to_3d_dense(edge_pred, node_pred.shape[0]) # [num nodes, num nodes, edge types]

    # Recover full 3d adjacency matrix for edge targets
    edge_tar_3d = triu_to_3d_dense(edge_tar, node_tar.shape[0]) # [num nodes, num nodes, edge types]
    
    # --- The two output matrices tell us how many edges are connected with each of the atom types
    # Multiply each of the edge types with the atom types for the predictions
    node_edge_preds = torch.empty((MAX_MOLECULE_SIZE, len(SUPPORTED_ATOMS), len(SUPPORTED_EDGES)), dtype=torch.float, device=device)
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_preds[:, :, edge] = torch.matmul(edge_pred_3d[:, :, edge], node_pred[:, :9])

    # Multiply each of the edge types with the atom types for the targets
    node_edge_tar = torch.empty((node_tar.shape[0], len(SUPPORTED_ATOMS), len(SUPPORTED_EDGES)), dtype=torch.float, device=device)
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_tar[:, :, edge] = torch.matmul(edge_tar_3d[:, :, edge], node_tar.float())
    
    # Reduce to matrix with [num atom types, num edge types]
    node_edge_pred_matrix = torch.sum(node_edge_preds, dim=0)
    node_edge_tar_matrix = torch.sum(node_edge_tar, dim=0)

    if torch.equal(node_edge_pred_matrix.int(), node_edge_tar_matrix.int()):
        print("Reconstructed node-edge pairs: ", node_edge_pred_matrix.int())
    
    node_edge_loss = torch.mean(sum(squared_difference(node_edge_pred_matrix, node_edge_tar_matrix.float())))

    # Calculate node-edge-node for preds
    node_edge_node_preds = torch.empty((MAX_MOLECULE_SIZE, MAX_MOLECULE_SIZE, len(SUPPORTED_EDGES)), dtype=torch.float, device=device)
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_node_preds[:, :, edge] = torch.matmul(node_edge_preds[:, :, edge], node_pred[:, :9].t())

    # Calculate node-edge-node for targets
    node_edge_node_tar = torch.empty((node_tar.shape[0], node_tar.shape[0], len(SUPPORTED_EDGES)), dtype=torch.float, device=device)
    for edge in range(len(SUPPORTED_EDGES)):
        node_edge_node_tar[:, :, edge] = torch.matmul(node_edge_tar[:, :, edge], node_tar.float().t())

    # Node edge node loss
    node_edge_node_loss = sum(squared_difference(torch.sum(node_edge_node_preds, [0,1]), 
                                                 torch.sum(node_edge_node_tar, [0,1])))

    # TODO: Improve loss
    return node_edge_loss #  * node_edge_node_loss


def approximate_recon_loss(node_targets, node_preds, triu_targets, triu_preds):
    """
    See: https://github.com/seokhokang/graphvae_approx/
    TODO: Improve loss function 
    """
    # Convert targets to one hot
    onehot_node_targets = to_one_hot(node_targets, SUPPORTED_ATOMS ) #+ ["None"]
    onehot_triu_targets = to_one_hot(triu_targets, ["None"] + SUPPORTED_EDGES)

    # Reshape node predictions
    node_matrix_shape = (MAX_MOLECULE_SIZE, (len(SUPPORTED_ATOMS) + 1)) 
    node_preds_matrix = node_preds.reshape(node_matrix_shape)

    # Reshape triu predictions 
    edge_matrix_shape = (int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1))/2), len(SUPPORTED_EDGES) + 1) 
    triu_preds_matrix = triu_preds.reshape(edge_matrix_shape)

    # Apply sum on labels per (node/edge) type and discard "none" types
    node_preds_reduced = torch.sum(node_preds_matrix[:, :9], 0)
    node_targets_reduced = torch.sum(onehot_node_targets, 0)
    triu_preds_reduced = torch.sum(triu_preds_matrix[:, 1:], 0)
    triu_targets_reduced = torch.sum(onehot_triu_targets[:, 1:], 0)
    
    # Calculate node-sum loss and edge-sum loss
    node_loss = sum(squared_difference(node_preds_reduced, node_targets_reduced.float()))
    edge_loss = sum(squared_difference(triu_preds_reduced, triu_targets_reduced.float()))

    # Calculate node-edge-sum loss
    # Forces the model to properly arrange the matrices
    node_edge_loss = calculate_node_edge_pair_loss(onehot_node_targets, 
                                      onehot_triu_targets, 
                                      node_preds_matrix, 
                                      triu_preds_matrix)

    approx_loss =   node_loss  + edge_loss + node_edge_loss

    if all(node_targets_reduced == node_preds_reduced.int()) and \
        all(triu_targets_reduced == triu_preds_reduced.int()):
        print("Reconstructed all edges: ", node_targets_reduced)
        print("and all nodes: ", node_targets_reduced)
    return approx_loss


def gvae_loss(triu_logits, node_logits, edge_index, edge_types, node_types, \
              mu, logvar, batch_index, kl_beta):
    """
    Calculates the loss for the graph variational autoencoder,
    consiting of a node loss, an edge loss and the KL divergence.
    """
    # Convert target edge index to dense adjacency matrix
    batch_edge_targets = torch.squeeze(to_dense_adj(edge_index))

    # Add edge types to adjacency targets
    batch_edge_targets[edge_index[0], edge_index[1]] = edge_types[:, 1].float()

    # For this model we always have the same (fixed) output dimension
    graph_size = MAX_MOLECULE_SIZE*(len(SUPPORTED_ATOMS) + 1) 
    graph_triu_size = int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1)) / 2) * (len(SUPPORTED_EDGES) + 1)

    # Reconstruction loss per graph
    batch_recon_loss = []
    triu_indices_counter = 0
    graph_size_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):   
            # Get upper triangular targets for this graph from the whole batch
            triu_targets, node_targets = slice_graph_targets(graph_id, 
                                                            batch_edge_targets, 
                                                            node_types,
                                                            batch_index)

            # Get upper triangular predictions for this graph from the whole batch
            triu_preds, node_preds = slice_graph_predictions(triu_logits, 
                                                            node_logits, 
                                                            graph_triu_size, 
                                                            triu_indices_counter, 
                                                            graph_size, 
                                                            graph_size_counter)

            # Update counter to the index of the next (upper-triu) graph
            triu_indices_counter = triu_indices_counter + graph_triu_size
            graph_size_counter = graph_size_counter + graph_size

            # Calculate losses
            recon_loss = approximate_recon_loss(node_targets, 
                                                node_preds, 
                                                triu_targets, 
                                                triu_preds)
            batch_recon_loss.append(recon_loss)   

    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = torch.true_divide(sum(batch_recon_loss),  num_graphs)
    
    # KL Divergence
    kl_divergence = kl_loss(mu, logvar)

    return batch_recon_loss + kl_beta * kl_divergence, kl_divergence



def graph_representation_to_molecule(node_types, adjacency_triu):
    """
    Converts the predicted graph to a molecule and validates it
    using RDKit.
    """
    # Create empty mol
    mol = Chem.RWMol()

    # Add atoms to mol and store their index
    node_to_idx = {}
    for i in range(len(node_types)):
        a = Chem.Atom(int(node_types[i]))
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    
    # Add edges to mol
    num_nodes = len(node_types)
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
        smiles = None

    # TODO: Visualize and save (use deepchem smiles_to_image)
    return smiles, mol
