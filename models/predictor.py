import torch
from base_ranker import BaseRanker
from typing import Any, Dict, List
from networks import DeepSet, SetTransformer
#import metrics

class Predictor(BaseRanker):
    """A simple NN ranking model, without learning feature selection."""
    def __init__(self, hparams: Dict[str, Any]):
        super(Predictor, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        if hparams['predictor'] == 'linear':
            self.net = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers'])
        elif hparams['predictor'] == 'transformer':
            self.net = SetTransformer(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_heads'], hparams['num_layers'])
        else:
            raise ValueError(f"Invalid predictor: {hparams['predictor']}")
        self.save_hyperparameters() # save hyperparameters in hparams.

        
    def forward(self, Input: torch.Tensor):
        return self.net(Input).squeeze(-1)    # B, 1 -> B


