"""Implements dataprocess"""
import os
import pickle
import warnings
import logging

import pandas as pd
import deepchem as dc
from sklearn.model_selection import train_test_split

from drugdealer.utils.read_utils import read_smi
from .utils.process import filter_cell_lines, \
    filter_genes, filter_drugs, to_dict


warnings.filterwarnings("ignore")


def  run_dataprocess(params: dict,
                     logger: logging.Logger):
    """Path parameters for the data.

    Args:
        params (dict): Dictionary with parameters.
        logger (logger): Logger.
    """

    # Load data
    data_dir = params['data_dir']
    processed_dir = params['processed_dir']
    path_genes = os.path.join(data_dir, params['genes'])
    path_train_ic50 = os.path.join(data_dir, params['train'])
    path_test_ic50 = os.path.join(data_dir, params['test'])
    path_gene_expression = os.path.join(data_dir, params['gene_expressions'])
    path_smiles = os.path.join(data_dir, params['smiles'])
    training_dict_path = os.path.join(processed_dir, params['train_dict'])
    validation_dict_path = os.path.join(processed_dir, params['val_dict'])
    test_dict_path = os.path.join(processed_dir, params['test_dict'])

    # Load data
    genes = pd.read_pickle(path_genes)
    training_set = pd.read_csv(path_train_ic50).drop(['Unnamed: 0'],
                                                     axis=1)
    test_set = pd.read_csv(path_test_ic50).drop(['Unnamed: 0'],
                                                axis=1)
    
    gene_expressions = pd.read_csv(path_gene_expression)
    gene_expressions = gene_expressions.rename(columns={'Unnamed: 0': 'cell_line'})
    
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    # Convert columns to float32 based on the specified mapping
    dtype_mapping = {col: 'float32' for col in gene_expressions.columns if col != 'cell_line'}
    gene_expressions = gene_expressions.astype(dtype_mapping)

    drug_expression = read_smi(path_smiles)
    with open(path_genes, 'rb') as f:
        genes = pickle.load(f)

    # Filter genes and cell lines
    logger.info('Filtering genes and cell lines...')
    training_set_filtered, test_set_filtered = \
        filter_cell_lines(training_set=training_set,
                          test_set=test_set,
                          gene_expressions=gene_expressions)
    
    gene_expressions_filtered = filter_genes(gene_expressions=gene_expressions,
                                             genes=genes)

    # Filter out drugs without representation
    logger.info('Filtering drugs without representations...')
    training_set_refiltered, test_set_refiltered = \
        filter_drugs(drug_representations=drug_expression,
                     training_set=training_set_filtered,
                     test_set=test_set_filtered)
    
    logger.info('Preparing data for training...')
    # Make dictionary with smiles
    drugs_with_represenation = []
    drugs_representations = {}
    for k in drug_expression:
        drugs_with_represenation.append(k['drug'])
        drugs_representations[k['drug']] = k['structure']
        
    # Apply get_molecular structure
    training_set_refiltered['molecular_structure'] = training_set_refiltered['drug'].apply(
        lambda x: drugs_representations.get(x))
    test_set_refiltered['molecular_structure'] = test_set_refiltered['drug'].apply(
        lambda x: drugs_representations.get(x))

    # Merge with the expressions
    training_features = training_set_refiltered.merge(
        right=gene_expressions_filtered, on='cell_line', how='left')
    test_features = test_set_refiltered.merge(
        right=gene_expressions_filtered, on='cell_line', how='left')
    
    # Split train to train and validation
    training_features, validation_features = train_test_split(training_features, test_size=0.2)

    # Seperate features and labels and fill nans with zeros
    X_train = training_features.drop(['drug', 'cell_line', 'IC50'], axis=1)
    X_train = X_train.fillna(value=0)
    y_train = training_features['IC50']
    X_val = validation_features.drop(['drug', 'cell_line', 'IC50'], axis=1)
    X_val = X_val.fillna(value=0)
    y_val = validation_features['IC50']
    X_test = test_features.drop(['drug', 'cell_line', 'IC50'], axis=1)
    X_test = X_test.fillna(value=0)
    y_test = test_features['IC50']

    # Featurizing training smiles
    logger.info('Featurizing training smiles into images.\n'
                'It might take 2-3 minutes...')
    featurizer = dc.feat.SmilesToImage(img_size=76).featurize
    X_train['molecular_structure'] = X_train['molecular_structure'].apply(
        featurizer)
    
    # Store train data to dictionaries
    logger.info('Storing training dataframe rows into dictionary...')
    training_dict = to_dict(X_train,y_train)

    # Save files
    logger.info('Saving into .pkl files...')
    with open(training_dict_path, 'wb') as file:
        pickle.dump(training_dict, file, protocol=5)

    # Featurizing validation smiles
    logger.info('Featurizing validation smiles into images.\n'
                'It might take 1-2 minutes...')
    featurizer = dc.feat.SmilesToImage(img_size=76).featurize
    X_val['molecular_structure'] = X_val['molecular_structure'].apply(
        featurizer)

    # Store validation data to dictionaries
    logger.info('Storing validation dataframe rows into dictionary...')
    validation_dict = to_dict(X_val,y_val)

    # Save files
    logger.info('Saving into .pkl files...')
    with open(validation_dict_path, 'wb') as file:
        pickle.dump(validation_dict, file, protocol=5)

    # Featurizing test smiles
    logger.info('Featurizing test smiles into images.\n'
                'It might take 1-2 minutes minutes...')
    X_test['molecular_structure'] = X_test['molecular_structure'].apply(
        featurizer)

    # Store test data to dictionaries
    logger.info('Storing test dataframe rows into dictionary...') 
    test_dict = to_dict(X_test, y_test)
    
    # Save files
    logger.info('Saving into .pkl files...')
    with open(test_dict_path, 'wb') as file:
        pickle.dump(test_dict, file, protocol=5)
