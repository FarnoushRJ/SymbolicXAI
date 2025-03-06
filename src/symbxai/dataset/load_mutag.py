import torch_geometric.datasets as datasets
from rdkit import Chem

# from .basf import BASFDataset

def load_dataset(dataset_name):
    if dataset_name in ['MUTAG', 'Mutagenicity']:
        dataset = datasets.TUDataset(name=dataset_name, root='data/')
    # elif dataset_name[:4] == 'BASF':
    #     uniprot = dataset_name.split('_')[1]
    #     dataset = BASFDataset(uniprot=uniprot, nbs=-1, transform=None)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
    return dataset

def get_substructure_match(molecule, substructure_smarts):
    # params = Chem.SmartsParserParams()
    # params.mergeHs = True
    substructure = Chem.MolFromSmarts(substructure_smarts)

    if molecule.HasSubstructMatch(substructure):
        substructure_match = molecule.GetSubstructMatch(substructure)
        return substructure_match
    else:
        return None

