from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import numpy as np


def make_atomic_scaffold_from_smiles(smiles):
    """
    Create an atomic scaffold from SMILES and return it in MOL format.

    Args:
        smiles (str): The SMILES representation of a molecule.

    Returns:
        Chem.Mol or None: The atomic scaffold in MOL format, or None if an error occurs.
    """
    # Create a molecule object from the input SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    # Remove stereochemistry information from the molecule
    Chem.rdmolops.RemoveStereochemistry(mol)
    
    # Get the atomic scaffold for the molecule
    atomic_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    try:
        # Attempt to sanitize the atomic scaffold
        Chem.SanitizeMol(atomic_scaffold)
    except ValueError:
        # If sanitization fails, return None
        return None

    # Return the atomic scaffold in MOL format
    return atomic_scaffold


def make_atomic_scaffold(mol):
    """
    Create an atomic scaffold from a molecule and return it in MOL format.

    Args:
        mol (Chem.Mol): A molecule in RDKit's Chem.Mol format.

    Returns:
        Chem.Mol or None: The atomic scaffold in MOL format, or None if an error occurs.
    """
    # Remove stereochemistry information from the molecule
    Chem.rdmolops.RemoveStereochemistry(mol)
    
    # Get the atomic scaffold for the molecule
    atomic_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    try:
        # Attempt to sanitize the atomic scaffold
        Chem.SanitizeMol(atomic_scaffold)
    except ValueError:
        # If sanitization fails, return None
        return None

    # Return the atomic scaffold in MOL format
    return atomic_scaffold

def smiles_counts(smiles):
    """
    Count the occurrence of each unique SMILES string and return the counts as a list of tuples.

    Args:
        smiles (list): A list of SMILES strings.

    Returns:
        list: A list of tuples, where each tuple contains a unique SMILES string and its count.
    """
    # Create an empty dictionary to store the counts
    count_dict = {}

    # Iterate through the list of SMILES strings
    for s in smiles:
        # Increment the count for the SMILES string in the dictionary
        count_dict[s] = count_dict.get(s, 0) + 1

    # Convert the dictionary items to a list of tuples and sort by count in descending order
    sorted_counts = sorted(list(count_dict.items()), key=lambda x: x[1], reverse=True)

    return sorted_counts

def get_atomic_scaffold_and_smiles_counts(df, smiles_col):
    """
    Calculate atomic scaffolds for SMILES strings in a DataFrame column and count their occurrences.

    Args:
        df (pandas.DataFrame): The DataFrame containing SMILES strings.
        smiles_col (str): The name of the column in the DataFrame containing SMILES strings.

    Returns:
        list: A list of atomic scaffolds as RDKit Mol objects.
        list: A list of tuples, where each tuple contains an atomic scaffold SMILES string and its count.
    """
    # Extract SMILES strings from the specified column and convert them to RDKit Mol objects
    mols = [Chem.MolFromSmiles(x) for x in df[smiles_col].tolist()]

    # Calculate atomic scaffolds for the Mol objects
    atomic_scaffolds = [make_atomic_scaffold(mol) for mol in mols]

    # Filter out None values (invalid scaffolds)
    valid_atomic_scaffolds = [s for s in atomic_scaffolds if s]

    # Count the occurrences of unique atomic scaffold SMILES strings
    aggregated_atomic_scaffolds = smiles_counts([Chem.MolToSmiles(s) for s in valid_atomic_scaffolds])

    return valid_atomic_scaffolds, aggregated_atomic_scaffolds

def top_different_scaffolds(actives_aggregated_atomic_scaffolds,
                            data_all_aggregated_atomic_scaffolds,
                            data_in_aggregated_atomic_scaffolds):
    """
    Identify the top different scaffolds between active compounds and two datasets.

    Args:
        actives_aggregated_atomic_scaffolds (list): A list of tuples containing active compound atomic scaffold SMILES
            strings and their counts.
        data_all_aggregated_atomic_scaffolds (list): A list of tuples containing atomic scaffold SMILES strings and
            their counts in a reference dataset (data_all).
        data_in_aggregated_atomic_scaffolds (list): A list of tuples containing atomic scaffold SMILES strings and
            their counts in an inactivity dataset (data_in).

    Returns:
        list: A list of tuples containing atomic scaffold SMILES strings and their counts that are unique to data_all.
        list: A list of tuples containing atomic scaffold SMILES strings and their counts that are unique to data_in.
    """
    # Extract unique atomic scaffold SMILES strings from active compounds
    actives_scaffolds_smi = list(set([item[0] for item in actives_aggregated_atomic_scaffolds]))

    # Find atomic scaffolds that are unique to data_all and data_in
    diff_all = list(set(actives_scaffolds_smi) - set(data_all_aggregated_atomic_scaffolds))
    diff_in = list(set(actives_scaffolds_smi) - set(data_in_aggregated_atomic_scaffolds))

    # Filter the active compound scaffolds to get the ones unique to data_all and data_in
    sca_diff_all = [item for item in actives_aggregated_atomic_scaffolds if item[0] in diff_all]
    sca_diff_in = [item for item in actives_aggregated_atomic_scaffolds if item[0] in diff_in]

    # Print the counts of different scaffolds
    print("Unique to data_in:", len(diff_in))
    print("Unique to data_all:", len(diff_all))
    print("Scaffolds unique to data_all:", len(sca_diff_all))
    print("Scaffolds unique to data_in:", len(sca_diff_in))

    return sca_diff_in, sca_diff_all
