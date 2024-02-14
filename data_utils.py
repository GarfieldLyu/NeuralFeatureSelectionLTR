
from collections import Counter
import itertools
from sklearn.datasets import load_svmlight_file
from typing import List, Dict, Tuple, Any
from pathlib import Path
from typing import Dict
import torch
import lightgbm as lgb
import numpy as np
import random
import h5py
import pickle
import json
from tqdm import tqdm


def normalize(x: torch.tensor):
    """ Normalize input data, for Web30k. """
    return torch.log1p(1 + torch.abs(x)) * torch.sign(x)

def add_noise(data: torch.tensor, mean: float=0.0, std: float=0.01):
    return data + torch.randn_like(data)*std + mean


class MSLTR(object):
    """Parse LTR ranking dataset class"""
    
    def __init__(self, DIRECTORY, DATASET, FOLD_NAME, DATA_TYPE) -> None:
        self.directory = Path(DIRECTORY) 
        self.dataset = DATASET
        self.fold_name = FOLD_NAME    
        if self.dataset not in ['MQ2008', 'Web30k', 'Yahoo']:
            raise ValueError(f'Invalid dataset {self.dataset}.')
    
        self.get_data(DATA_TYPE)

    def get_data(self, data_type: str='train') -> None:   # mode = [train, valid, test]
        data_dir = self.directory / 'Datasets' / self.dataset / self.fold_name / f'{data_type}.txt'
        self.features, self.labels, self.qids = self.txt_to_np(str(data_dir))
        self.unique_queries = np.unique(self.qids)

    def select_by_qid(self, qid: int) -> Tuple[np.array, np.array]:
        #instances = [(data['feature'], data['label']) for data in self.data_dict if data['qid']==qid]
        idx = np.where(self.qids==qid)[0]
        X = self.features[idx]
        Y = self.labels[idx]
        return X, Y

    def txt_to_np(self, fname) -> Tuple[np.array]:
        """ Parse .txt file to numpy using sklearn directly, without parsing by lines."""
        X, Y, queries = load_svmlight_file(fname, query_id=True)   # X is sparse
        return X.toarray(), Y, queries



class HDFSequence(lgb.Sequence):
    """Inherit from lightgbm Sequence library, used for Dataset Input to lightgbm models only. """
    def __init__(self, dataset_h5, batch_size: int):
        self.data = dataset_h5
        self.batch_size = batch_size
    
    def __getitem__(self, idx):
        return self.data[idx] 

    def __len__(self):
        return len(self.data)  


def load_h5_lgb(input_dir: Path, data_type: str, batch_size: int, reference=None):
    """ Load h5py dataset object to lightgbm Dataset, as input to lightgbm model only"""
    f_data = input_dir / f'{data_type}.h5'
    q_data = input_dir / f'{data_type}_query_group.pkl'
    fp = h5py.File(f_data, 'r')
    with open(q_data, 'rb')as f:
        group = pickle.load(f)
    data = lgb.Dataset(HDFSequence(fp['X'], batch_size), label=fp['Y'][:], group=group, reference=reference)
    return data


def MS_to_h5(directory: str, dataset: str, fold_name: str, data_type: str) -> None:
    """ Create h5 data object for specified dataset, save to h5py object.
        Args: 
            - directoy: project directory
            - dataset: MQ2008, Web30k, Yahoo
            - fold_name: Fold1 
            - data_type: train, valid, test 
    """
    DATA = MSLTR(directory, dataset, fold_name, data_type)
    Feature, Label = DATA.features, DATA.labels
    num_items, feature_size = Feature.shape
    print(f'Instance size: {num_items}, feature size: {feature_size}')

    out_dir = DATA.directory / 'Datasets' / DATA.dataset / DATA.fold_name / f'{data_type}.h5'

    with h5py.File(out_dir, 'w')as fp:
       fp.create_dataset('X', (num_items, feature_size), data = Feature)
       fp.create_dataset('Y', (num_items, 1), data = Label)

    group = list(Counter(DATA.qids).values())
    query_group = out_dir.parent / f'{data_type}_query_group.pkl'

    with open(query_group, 'wb')as f:
        pickle.dump(group, f)
        
    print(f'Saved h5 data to {out_dir}, query group data to {query_group}.')
   

def creat_pairs(directory: str, dataset: str, fold_name: str, data_type: str='train', max_size: int=20, seed: int=123) -> None:
    """ Create h5 pairwise training object for specified dataset, save to h5py object.
        Args: 
            - directoy: project directory
            - dataset: MQ2008, Web30k, Yahoo
            - fold_name: Fold1 
            - data_type: train, valid, test
        Save:
            - Triples of (qid, high_idx, low_idx).
    """
    
    Triples = []
    DATA = MSLTR(directory, dataset, fold_name, data_type)
    for query in tqdm(DATA.unique_queries):
        start_idx = np.where(DATA.qids == query)[0][0]  # start idx of query 
        _, Y = DATA.select_by_qid(query)
        unique_Y = np.unique(Y)  # already sorted from low to high.
        if unique_Y.size < 2:  # all negative/identical labels, skip. 
            continue
        else:
            pairs = itertools.combinations(unique_Y, 2)
            for (low, high) in pairs:
                low_idx = np.where(Y==low)[0]
                high_idx = np.where(Y==high)[0]
                pairs_idx = [(query, i+start_idx, j+start_idx) for i in high_idx for j in low_idx]
                pair_size = len(pairs_idx)
                sample_size = min(pair_size, max_size)
                random.seed(seed)
                Triples.extend(random.sample(pairs_idx, sample_size))
    Triples = np.array(Triples)
    print(f'Generated {Triples.shape[0]} (query, positive negative) pairs.')

    out_dir = DATA.directory / 'Datasets' / DATA.dataset / DATA.fold_name / f'{data_type}_pairs.h5'
    with h5py.File(out_dir, 'w')as fp:
       fp.create_dataset('pairs', Triples.shape, data = Triples)
    print(f'Saved h5 data to {out_dir}.')


def select_by_gbdt(feature_path: str, ratio: float=0.3) -> np.array:
    """Select the top ratio% most importance features, return an array of [0, 1] elements."""
    print(f'read gbdt feature path from {feature_path}...')
    with open(feature_path, 'r')as f:
        FI = json.load(f)
    order_sort = np.argsort(-np.array(FI))  # sorting in decreasing order.
    mask = np.zeros_like(order_sort)
    picked = order_sort[:int(ratio*len(order_sort))]
    mask[picked] = 1
    return mask


def generate_mask(mask: str, inp_dim: int, ratio: float, \
                  trained_dir:Path=Path('~/path-to-your-trained-models'), \
                  mask_file:str='feature_importances.json') -> torch.tensor:
    """ Generate global mask applies to the whole dataset.
        Args:
            mask: which mask to apply. options: ['None', 'random', 'split', 'rfe', 'cae', 'g_l2x', l2x', 'lassonet', 'tabnet', 'instancefg', 'invase', 'fisher', 'laplacian']
            inp_dim: input feature dimension. 46 for MQ2008, 136 for Web30k, 699 for Yahoo.
            ratio: ratio of features to keep.
            trained_dir: trained model directory, used for reading feature importance.
            mask_file: file name for mask method for saving the selected features.

    """
    if isinstance(trained_dir, str):
        trained_dir = Path(trained_dir)
    if mask == 'None':
        return torch.ones(inp_dim)
    elif mask == 'random':
        MASK = torch.zeros(inp_dim)
        choice = random.sample(range(inp_dim), int(ratio*inp_dim))
        MASK[choice] = 1
        return MASK
    elif mask == 'split':
        # sampled from lambdamart split feature importance
        MASK = torch.tensor(select_by_gbdt(trained_dir / 'FI_split.json', ratio))
        return MASK
    elif mask == 'rfe':
        feature_path = trained_dir / f'Lambdamart_feature_{ratio}.json'
        with open(feature_path, 'r')as f:
            feature_idx = json.load(f)
        feature_idx = torch.tensor(feature_idx)
    
    elif mask.lower() in ['g_l2x', 'cae', 'l2x', 'lassonet', 'tabnet', 'instancefg', 'invase']:
        feature_path = trained_dir / mask_file 
        with open(feature_path, 'r')as f:
            feature_idx = json.load(f)
        feature_idx = feature_idx[:int(inp_dim*ratio)]
        print(f'genenrating global feature mask from {feature_path}.')
    
    elif mask in ['fisher', 'laplacian']:
        feature_path = trained_dir / f'{mask}/feature_importances.json'
        with open(feature_path, 'r')as f:
            feature_idx = json.load(f)
        feature_idx = torch.tensor(feature_idx[:int(inp_dim*ratio)])
        print(f'genenrating global feature mask from {feature_path}.')

    else:
        raise ValueError(f'Invalid mask method: {mask}.')
    
    MASK = torch.zeros(inp_dim)
    MASK[feature_idx] = 1
    return MASK









    
