
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


class ClientFedRS(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model,
        label_dict=None,
        num_class=None
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.label_dict = label_dict
        self.num_class = num_class
        self.get_label_dist()

    def get_label_dist(self):
        cnts = list()
        for c in range(self.num_class):
            if c in self.label_dict:
                cnts.append(self.label_dict[c])
            else:
                cnts.append(0)
        self.cnts = torch.FloatTensor(np.array(cnts))
        self.dist = self.cnts / self.cnts.sum()
        # get distribution
        self.dist = self.dist / self.dist.max()
        alpha = 0.99
        self.dist = self.dist * (1.0 - alpha) + alpha
        self.dist = self.dist.reshape((1, -1))
        
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
            weight_decay=1e-5
        )
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                
                self.model.zero_grad()
                optimizer.zero_grad()
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                
                # forward
                _, x_mm = self.model(
                    x_a.float(), 
                    x_b.float(), 
                    l_a, 
                    l_b
                )
                
                if not self.multilabel:
                    # get outputs
                    outputs = self.model.classifier[:-1](x_mm)
                    # get classifier weight
                    ws = self.model.classifier[-1].weight
                    outputs = self.dist.to(self.device) * outputs.mm(ws.transpose(0, 1))
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                # backward
                loss = self.criterion(outputs, y)

                # backward
                loss.backward()
                
                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    10.0
                )
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
