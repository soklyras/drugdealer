"""Implementes data utilities"""
from typing import List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm


def filter_genes(gene_expressions: pd.DataFrame,
                 genes: List):
    """Filters genes

    Args:
        gene_expressions (pd.DataFrame): Gene expressions.
        genes (List): 2128 genes
    
    Returns:
        gene_expressions (pd.DataFrame): 
    """
    # Rename column and drop full nan columns
    gene_expressions = gene_expressions.rename(columns={'Unnamed: 0': 'cell_line'})
    gene_expressions.dropna(axis=1, how='all', inplace=True)

    # Find difference between all genes and 2128 genes.
    total_genes = set(gene_expressions.columns.drop('cell_line').to_list())
    genes = set(genes)
    interesection_genes = total_genes & genes
    drop_genes = total_genes.difference(interesection_genes)

    # Filter
    gene_expressions.drop(drop_genes, axis=1, inplace=True)

    return gene_expressions


def filter_cell_lines(training_set: pd.DataFrame,
                      test_set: pd.DataFrame,
                      gene_expressions: pd.DataFrame):
    """Filters cell lines that we have info for.

    Args:
        training_set (pd.DataFrame): Training set.
        test_set (pd.DataFrame): Test set.
        gene_expressions (_type_): Gene expressions.

    Returns:
        training_set_filtered: Filtered training set.
        test_set_filtered: Filtered test set.
    """

    # Find intersection between the cell_lines in the training set and in the gene
    # expression data set. We will not use cell_lines we don't have expressions.
    cell_lines_training = pd.unique(training_set['cell_line']).tolist()
    cell_lines_test = pd.unique(test_set['cell_line']).tolist()
    cell_lines_genes = set(pd.unique(gene_expressions['cell_line']).tolist())
    intersect_cell_training = list(set(cell_lines_training) & set(cell_lines_genes))
    intersect_cell_test = list(set(cell_lines_test) & set(cell_lines_genes))

    # Filter the training and test set
    training_set_filtered = training_set.loc[training_set['cell_line'].isin(intersect_cell_training)]
    test_set_filtered = test_set.loc[test_set['cell_line'].isin(intersect_cell_test)]
    
    return training_set_filtered, test_set_filtered


def filter_drugs(drug_representations: Dict,
                 training_set: pd.DataFrame,
                 test_set: pd.DataFrame):
    """Filters drugs with no representation.

    Args:
        drug_representations (Dict): Smiles for each drug.
        training_set (pd.DataFrame): Training set.
        test_set (pd.DataFrame): Test set.
    """
    # Filter out the drugs with no representation
    training_drugs = pd.unique(training_set['drug']).tolist()
    test_drugs = pd.unique(test_set['drug']).tolist()

    # Find drugs with representation
    drugs_with_represenation = []
    drugs_representations = {}
    for k in drug_representations:
        drugs_with_represenation.append(k['drug'])
        drugs_representations[k['drug']] = k['structure']

    # Filter training and test set
    drugs_intersection_training = list(set(training_drugs)
                                       & set(drugs_with_represenation))
    training_set_refiltered = training_set.loc[
        training_set['drug'].isin(drugs_intersection_training)]

    drugs_intersection_test = list(set(test_drugs)
                                   & set(drugs_with_represenation))
    test_set_refiltered = test_set.loc[
        test_set['drug'].isin(drugs_intersection_test)]
    return training_set_refiltered, test_set_refiltered


def to_dict(dataframe: pd.DataFrame,
            labels: pd.DataFrame):
    """Storing data to dictionaries.

    Args:
        dataframe (pd.DataFrame): DataFrame with data.
    """
    dictionary = {}
    for idx, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        dictionary[idx] = {}
        dictionary[idx]['molecular_structure'] = np.squeeze(row['molecular_structure'])
        row.drop('molecular_structure', inplace=True)
        dictionary[idx]['gene_expression'] = row
        dictionary[idx]['ic50'] = labels[idx]
    del dataframe
    return dictionary


def normalize_column(column):
    """Normalizes images.

    Args:
        images (np.ndarray): Image data

    Returns:
        _type_: Normalized image
    """
    normalized_images = [(elem - elem.min()) / (elem.max() - elem.min()) for elem in column]
    return normalized_images
