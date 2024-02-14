import metrics
import torch
import pytorch_lightning as pl
from typing import Iterable, List, Any, Dict
import numpy as np


# @Abstract class for basic ranking models.
class BaseRanker(pl.LightningModule):
    """Base class for rankers. Necessary functions to be extended:
        - forward().  make sure the first output must always be the prediction logits.
    """
    def __init__(self, train_mode,  rank_loss: str, 
                    ndcg_truncate: List[int] = [1, 3, 5, 10], early_signal: int=10) -> None:
        """Init function
            Args
            - rank_loss: Listwise loss function for training. By default, softmax crossentropy loss.
            - ndcg_truncate: NDCG at k, by default [1, 3, 5, 10].
            - early_signal: ndcg at k for validation data, used as early stopping signal. By default, ndcg@10
        
        """
        
        super(BaseRanker, self).__init__()
        self.train_mode = train_mode
        if self.train_mode == 'pointwise':
            print(f'pointwise training, using MSE loss.')
            self.rank_loss = torch.nn.MSELoss()

        elif self.train_mode == 'pairwise':
            print(f'pairwise training, using {rank_loss} loss.')
            self.rank_loss = metrics.get_loss(rank_loss)
            
        elif self.train_mode == 'listwise':
            print(f'listwise training, using {rank_loss} loss.')
            self.rank_loss = metrics.get_loss(rank_loss)
        else:
            raise ValueError(f'Invalid train_mode {self.train_mode}.')
        self.ndcg_truncate = ndcg_truncate
        self.early_signal = early_signal


    def training_step(self, batch, batch_idx) ->torch.Tensor:
        """ Train for a batch, make sure batch contains a tuple of (data, label).
            The first output of forward function should always be the predicting logits.
            Args:
                batch (InstanceTrainingBatch): A training batch.
                batch_idx (int): Batch index.
            Returns:
                torch.Tensor: training loss.   
        """
        
        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            high_logits = self(high_data)
            low_logits = self(low_data)

            if isinstance(high_logits, (list, tuple)):
                high_logits = high_logits[0]
                low_logits = low_logits[0]
            loss = self.rank_loss(high_logits, low_logits, high_label, low_label)

        else:
            data_batch, label_batch = batch
            logits = self(data_batch)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            if self.train_mode == 'pointwise':
                loss = self.rank_loss(logits, label_batch)

            elif self.train_mode == 'listwise':
                loss = self.rank_loss(logits, label_batch) / data_batch.shape[0]
              
            else:
                raise ValueError(f'Invalid train_mode {self.train_model}')
            
        self.log("rank_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """ Valid a batch, returns a predicting logits batch. 
            Args:
                A validation batch. Make sure a batch contains all items of a single query.
            Returns:
                #A ndcg@k score, used for early stopping signal, k=10 by default.
                prediction and labels, compute ndcg in the epoch end for efficiency reason.
        """
     
        data_batch, label_batch = batch

        if len(data_batch.shape) > 2:
            data_batch = data_batch.squeeze(0)
            label_batch = label_batch.squeeze(0)
        
        logits = self(data_batch)

        if isinstance(logits, (list, tuple)):
            logits = logits[0].detach().data
        
        return {'preds':logits, 'labels':label_batch}

    def validation_epoch_end(self, outputs: Iterable[torch.Tensor]) -> None:
        """Collect output for all batches and return the averages of ndcg scores at early_signal, 10 by default.
            Args: 
                outputs of all test steps:  Iterable of {'preds': tensor, 'labels': tensor} a step.
            Returns:
               None, the log already saves the validation scores for early stopping.
        """

        valid_ndcg = []
        for output in outputs:
            ndcg = metrics.list_ndcg(output['preds'].cpu().numpy(), output['labels'].cpu().numpy(), self.early_signal)
            valid_ndcg.append(ndcg)

        valid_ndcg_avg = np.mean(valid_ndcg)  # mean ncdg@early_signal on whole dataset.
        self.log(f'validation_ndcg@{self.early_signal}', valid_ndcg_avg)

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """ Test a batch, returns a predicting logits batch
            Args:
                test batch, contains all items of a identical query.
                batch idx
            Return:
                - prediction logits.
                - labels
        """
        data_batch, label_batch = batch
        if len(data_batch.shape) > 2:
            data_batch = data_batch.squeeze(0)
            label_batch = label_batch.squeeze(0)
        
        logits = self(data_batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0].detach().data
        return {'preds':logits, 'labels':label_batch}
    
    def test_epoch_end(self, outputs: Iterable[Dict[str, torch.Tensor]]) -> List[Any]:
        """Collect output for all batches and return the averages of ndcg scores at all truncates.
            Args: 
                outputs of all test steps:  Iterable of {'preds': tensor, 'labels': tensor} a step.
            Returns:
                List: {test_ndcg@k: avg_ndcg}, k= [1, 3, 5 10] by default.
        """

        NDCG = []    # ndcgs for all steps, each ndcgs include all ndcg@k, k = 1, 3, 5, 10. 

        for output in outputs:
            preds, labels = output['preds'].cpu().numpy(), output['labels'].cpu().numpy()
            ndcgs = []
            for k in self.ndcg_truncate:
                ndcg_k = metrics.list_ndcg(preds, labels, k)
                ndcgs.append(ndcg_k)
            NDCG.append(ndcgs)
        
        # compute average ndcg@k
        avg_result = {}
        AVG_NDCG = [np.mean([N[i] for N in NDCG]) for i, _ in enumerate(self.ndcg_truncate)]

        for i, k in enumerate(self.ndcg_truncate):
            key = f'test_ndcg@{k}'
            avg_result[key] = AVG_NDCG[i]
            self.log(f'test_ndcg@{k}', AVG_NDCG[i])

        NDCG.append(avg_result)
        self.test_result = NDCG  # save all ndcgs for all dataset, for analysis.
        return NDCG
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)      
        return optimizer
