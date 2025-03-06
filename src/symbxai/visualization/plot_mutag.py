import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.display import display
import numpy as np
from matplotlib.lines import Line2D
import torch
import os 

def get_molecule_plot_info(data, dataset_name='MUTAG',info_list=[]):
    result = {}

    if dataset_name == 'Mutagenicity':
        node_label_dict = {0:'C',1:'O',2:'Cl',3:'H',4:'N',5:'F',6:'Br',7:'S',8:'P',9:'I',10:'Na',11:'K',12:'Li',13:'Ca'}
        bond_type_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
    elif dataset_name == 'MUTAG':            
        node_label_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
        bond_type_dict = {0: Chem.rdchem.BondType.AROMATIC, 1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
    elif dataset_name[:4] == 'BASF':
        from src.data.load import load_dataset
        dataset = load_dataset(dataset_name)
        node_label_dict = {i: dataset.all_atomic_nbs[i] for i in range(len(dataset.all_atomic_nbs))}
        bond_type_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.AROMATIC, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
    elif dataset_name[:22] == 'P27338_P29274_balanced':
        from src.data.basf import BASFDataset
        dataset = BASFDataset(file_name = 'data/P27338_P29274_balanced_df.tsv')
        node_label_dict = {i: dataset.all_atomic_nbs[i] for i in range(len(dataset.all_atomic_nbs))}
        bond_type_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.AROMATIC, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
    atoms = [node_label_dict[i] for i in data.x.argmax(axis=1).tolist()]

    if 'molecule' in info_list or 'pos' in info_list:
        molecule = Chem.RWMol()

        for atom in atoms:
            molecule.AddAtom(Chem.Atom(atom))
        for bond, bond_type in zip(data.edge_index.t().tolist(), data.edge_attr.argmax(axis=1).tolist()):
            if bond[0] < bond[1]:
                molecule.AddBond(int(bond[0]), int(bond[1]), bond_type_dict[bond_type])
        
        sanitize = True
        if sanitize:
            ### Remove H with no bond
            rdedmol = Chem.EditableMol(molecule)
            to_delete = []
            for idx, atom in enumerate(molecule.GetAtoms()):
                if len(atom.GetBonds()) == 0:
                    to_delete.append(idx)
            for atom in reversed(to_delete):        
                rdedmol.RemoveAtom(atom)

            molecule = rdedmol.GetMol()

            for atom in molecule.GetAtoms():
                if atom.GetSymbol() == 'N' and sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) == 4:
                    atom.SetFormalCharge(1)
                if atom.GetSymbol() == 'O' and sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) == 3:
                    atom.SetFormalCharge(1)
                if atom.GetSymbol() == 'O' and sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) == 1:
                    atom.SetFormalCharge(-1)

        result['molecule'] = molecule

    if 'pos' in info_list:
        AllChem.Compute2DCoords(molecule)
        # compute 2D positions
        pos = []
        n_nodes = molecule.GetNumAtoms()
        for i in range(n_nodes):
            conformer_pos = molecule.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x, conformer_pos.y])

        pos = np.array(pos)
        result['pos'] = pos
    
    if 'node_label_dict' in info_list:
        result['node_label_dict'] = node_label_dict
        
    if 'bond_type_dict' in info_list:
        result['bond_type_dict'] = bond_type_dict

    return result

def plot_molecule(data, dataset_name='MUTAG',filename=None):
    molecule = get_molecule_plot_info(data, dataset_name=dataset_name,info_list=['molecule'])['molecule']
    img = Draw.MolToImage(molecule)
    display(img)
    if filename is not None:
        img.save(filename)

def shrink(rx, ry, factor=11):
    """This function is used to make the walks smooth."""

    rx = np.array(rx)
    ry = np.array(ry)

    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    last_node = rx.shape[0] - 1
    concat_list_x = [np.linspace(rx[0], rx[0], 5)]
    concat_list_y = [np.linspace(ry[0], ry[0], 5)]
    for j in range(last_node):
        concat_list_x.append(np.linspace(rx[j], rx[j + 1], 5))
        concat_list_y.append(np.linspace(ry[j], ry[j + 1], 5))
    concat_list_x.append(np.linspace(rx[last_node], rx[last_node], 5))
    concat_list_y.append(np.linspace(ry[last_node], ry[last_node], 5))

    rx = np.concatenate(concat_list_x)
    ry = np.concatenate(concat_list_y)

    filt = np.exp(-np.linspace(-2, 2, factor) ** 2)
    filt = filt / filt.sum()

    rx = np.convolve(rx, filt, mode='valid')
    ry = np.convolve(ry, filt, mode='valid')

    return rx, ry

def plot_molecule_with_relevance(data, relevances, rel_level='walk', dataset_name='MUTAG', color_factor=1, legend=True, filename=None, **kwargs):
    factor = color_factor
    shrinking_factor = 11
    fig_width = 10 if 'fig_width' not in kwargs else kwargs['fig_width']

    plot_info = get_molecule_plot_info(data, 
                                       dataset_name=dataset_name,
                                       info_list=['pos', 'node_label_dict', 'bond_type_dict'])
    
    pos = plot_info['pos']
    node_label_dict = plot_info['node_label_dict']
    bond_type_dict = plot_info['bond_type_dict']

    pos_size = pos.max(axis=0) - pos.min(axis=0)
    fig_height = (fig_width / pos_size[0]) * pos_size[1]
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1)

    def _iterate_over_all_walks(ax, relevances, selected_color=None):
        # visualization settings
        selfloopwidth = 0.25
        linewidth = 13.
        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(relevances):
            # get walk color
            if selected_color is None:
                color = 'b' if relevance < 0 else 'r'
            # get opacity
            alpha = abs(relevance * factor)
            alpha = min(alpha, 1.)

            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            # plot g loops
            for i in range(len(rx) - 1):
                if rx[i] == rx[i + 1] and ry[i] == ry[i + 1]:
                    rx_tmp = rx[i] + selfloopwidth * np.cos(np.linspace(0, 2 * np.pi, 128))
                    ry_tmp = ry[i] + selfloopwidth * np.sin(np.linspace(0, 2 * np.pi, 128))
                    ax.plot(rx_tmp, ry_tmp, color=color, alpha=alpha, lw=linewidth, zorder=1.)
            # plot walks
            rx, ry = shrink(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.)
        return ax
 
    if relevances is not None and rel_level == 'walk':
        ax = _iterate_over_all_walks(ax, relevances, selected_color=None)

    G = nx.from_edgelist(data.edge_index.T)
    G.remove_edges_from(nx.selfloop_edges(G))

    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", alpha=0, node_size=500)
    collection.set_zorder(2.)

    # plot bonds
    edge_bond = zip(data.edge_index.T, data.edge_attr.argmax(axis=1).tolist())
    edges_in_4_bonds = [[], [], [], []]

    for edge, bond in edge_bond:
        if bond_type_dict[bond] == Chem.rdchem.BondType.SINGLE: edges_in_4_bonds[0].append(edge)
        if bond_type_dict[bond] == Chem.rdchem.BondType.AROMATIC: edges_in_4_bonds[1].append(edge)
        if bond_type_dict[bond] == Chem.rdchem.BondType.DOUBLE: edges_in_4_bonds[2].append(edge)
        if bond_type_dict[bond] == Chem.rdchem.BondType.TRIPLE: edges_in_4_bonds[3].append(edge)

    bond_styles = ['-', '-', '-', '-']
    bond_width = [2, 2, 2, 2]
    bond_color = ['k','r','b','g']

    lines = []
    for i in range(4):
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges_in_4_bonds[i],
                                width=bond_width[i], 
                                edge_color=bond_color[i], 
                                style=bond_styles[i], 
                                alpha=None, 
                                ax=ax)
        lines.append(Line2D([0], [0], linestyle=bond_styles[i], color=bond_color[i], lw=bond_width[i]))
    if legend:
        ax.legend(lines, ['single bond', 'aromatic bond', 'double bond', 'triple bond'], )
    ax.axis('off')

    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    atoms = data.x.argmax(axis=1).tolist()
    atoms = [node_label_dict[atom] for atom in atoms]
    nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(atoms)}, font_size=30)
    plt.axis('off')
    

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=600, format='png',bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# funcs
def get_substructure_match(molecule, substructure_smarts):
    substructure = Chem.MolFromSmarts(substructure_smarts)
    if molecule.HasSubstructMatch(substructure):
        substructure_match = molecule.GetSubstructMatch(substructure)
        return substructure_match
    else:
        return None

def plot_substructure_match(molecule, substructure_match, data=None):
    walk_rel = [[[node,node], 1] for node in substructure_match]
    if data is not None: plot_molecule_with_relevance(data, walk_rel, rel_level='walk', dataset_name='Mutagenicity', color_factor=1, legend=True, filename=None, fig_width=5)
    display(Draw.MolToImage(molecule, highlightAtoms=substructure_match))