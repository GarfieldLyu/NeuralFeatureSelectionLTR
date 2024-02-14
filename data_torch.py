import torch
import h5py
import pickle
from torch.utils import data
from data_utils import generate_mask, normalize, add_noise
from typing import Dict, Tuple, Iterable, Any 
from sklearn.datasets import load_svmlight_file
from pathlib import Path

""" Convert data to torch dataset, including listwise and pairwise, pointwise is not necessary as it can use listwise directly.
    After downloading the dataset, we need to preprocess the dataset and save it to h5 format, see preprocess.py, then we can use this class to load the data.

"""


class ListwiseTrainingset(data.Dataset):
    """ A trainingset for listwise training. Read each X, Y pair from pre-saved h5 file."""
    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.normalize = args['normalize']  # normalize input features?, only for Web30k, we use log1p normalization.
        self.noise = args['noise']          # add noise to input features? this didn't help improving the performance, so we didn't use it.
        self.y_scale = 1                    # default value, no normalization for y labels.
        if args['normalize_y']:
            self.y_scale = args['y_scale']  # normalize y labels between 0-1 for pointwise training.
        self.data_file = args['data_h5']    # data file in h5 format, preprocessed by preprocess.py

        with h5py.File(self.data_file, 'r')as fp:
            self.length = len(fp['Y'])
        print(f'Training length: {self.length}' )

        # apply global mask to each input features? 
        self.mask = generate_mask(args['global_mask'], \
                                  args['input_dim'], \
                                  args['global_mask_ratio'], \
                                  args['trained_fold'], \
                                  args['mask_file'])    

    def __getitem__(self, index):           
        with h5py.File(self.data_file, 'r')as fp:
            Xs = torch.tensor(fp['X'][index]).double() 
            Ys = torch.tensor(fp['Y'][index][0]).double()   # doc_len
        if self.normalize:
            Xs = normalize(Xs)

        if self.noise:
            Xs = add_noise(Xs)
        Xs *= self.mask 
        Ys /= self.y_scale
        return Xs, Ys

    def __len__(self) -> int:
        return self.length
    
    def collate_fn(self, inputs: Iterable[Tuple[Any, Any]]):
        return inputs   


class PairwiseTrainingset(data.Dataset):
    """ A trainingset for pairwise training. Read all pairs from h5 file."""
    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.normalize = args['normalize']
        self.noise = args['noise']
        self.pairs_file = Path(args['pairs_h5'])

        with h5py.File(self.pairs_file, 'r')as fp:
            self.length = len(fp['pairs'])
        print(f'Training length: {self.length}' )

        self.mask = generate_mask(args['global_mask'], \
                                  args['input_dim'], \
                                  args['global_mask_ratio'], \
                                  args['trained_fold'], \
                                  args['mask_file'])
        
        self.data_file = self.pairs_file.parent / 'train.txt'
        self.X, self.Y, self.queries = load_svmlight_file(str(self.data_file), query_id=True)

    def __getitem__(self, index):
        with h5py.File(self.pairs_file, 'r')as fp:
            triples = fp['pairs'][index]

        high_idx = triples[1]
        low_idx = triples[2]

        Xs_high = torch.tensor(self.X[high_idx].toarray()).squeeze(0).double()
        Xs_low = torch.tensor(self.X[low_idx].toarray()).squeeze(0).double()
        Y_high = torch.tensor(self.Y[high_idx]).double()
        Y_low = torch.tensor(self.Y[low_idx]).double()

        if self.normalize:
            Xs_high = normalize(Xs_high)
            Xs_low = normalize(Xs_low)
        if self.noise:
            Xs_high = add_noise(Xs_high)
            Xs_low = add_noise(Xs_low)

        Xs_high *= self.mask 
        Xs_low *= self.mask
        return Xs_high, Xs_low, Y_high, Y_low

    def __len__(self) -> int:
        return self.length 


class Validset(data.Dataset):
    """A validate dataset. Read all docs for a given query based on query group ids.
        In the experiment we set the batch_size to 1, so we can read all docs for a given query in one batch.
        So the evaluation e.g., ndcg@k is computed based on query level, not batch level.
    """
    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.normalize = args['normalize']
        self.noise = args['noise']
        self.data_file = args['data_h5']
        self.group_file = args['group_pkl']  # query group file, used to read all docs idx for a given query.

        with open(self.group_file, 'rb')as f:
            self.group = pickle.load(f)

        self.length = len(self.group)      # length==num_query
        print(f'Predicting length: {self.length}' )

        self.mask = generate_mask(args['global_mask'], \
                                  args['input_dim'], \
                                  args['global_mask_ratio'], \
                                  args['trained_fold'], \
                                  args['mask_file'])

    def __getitem__(self, index):  # suppose 1 query per batch, batch_size==1
        start_id = sum(self.group[:index])
        end_id = start_id + self.group[index] 
        with h5py.File(self.data_file, 'r')as fp:
            Xs = torch.tensor([fp['X'][idx] for idx in range(start_id, end_id)]).double()  # doc_len, feature_dim
            Ys = torch.tensor([fp['Y'][idx][0] for idx in range(start_id, end_id)]).double()   # doc_len
        if self.normalize:
            Xs = normalize(Xs)
        if self.noise:
            Xs = add_noise(Xs)
        Xs *= self.mask 
        return Xs, Ys

    def __len__(self) -> int:
        return self.length
    
    def collate_fn(self, inputs: Iterable[Tuple[Any, Any]]):
        return inputs   # do nothing for now, since 1 query per batch


