
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

warnings.filterwarnings('ignore')
from evaluation import EvalMetric


class Client(object):
    def __init__(self, args, device, criterion, dataloader, model):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth
    
    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-04
        )
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                
                self.model.zero_grad()
                optimizer.zero_grad()
                x_a, x_b, y = batch_data
                
                # missing modality case
                if x_a[0] is not None: x_a = x_a.to(self.device)
                if x_b[0] is not None: x_b = x_b.to(self.device)
                y = y.to(self.device)
                # pdb.set_trace()
                # forward
                outputs = self.model(x_a.float(), x_b.float())
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                
                # backward
                loss = self.criterion(outputs, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                # save results
                if not self.multilabel: 
                    self.eval.append_classification_results(
                        y, 
                        outputs, 
                        loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        loss
                    )
                    
        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
