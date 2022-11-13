import pandas as pd
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import copy, pdb, time, warnings, torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
import collections
warnings.filterwarnings('ignore')


class Client(object):
    def __init__(self, args, device, criterion, dataloader, model):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model = model
        
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

        # prediction and truths
        pred_list, truth_list, top_k_list, loss_list = list(), list(), list(), list()
        
        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-04)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                
                self.model.zero_grad()
                optimizer.zero_grad()

                x_a, x_b, y = batch_data
                
                # missing modality case
                if x_a[0] is not None: x_a = x_a.to(self.device)
                if x_b[0] is not None: x_b = x_b.to(self.device)
                y = y.to(self.device)
                
                # forward
                outputs = self.model(x_a.float(), x_b.float())
                outputs = torch.log_softmax(outputs, dim=1)

                # backward
                loss = self.criterion(outputs, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                # save results
                predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                top_k_predictions = np.argsort(outputs.detach().cpu().numpy(), axis = 1)[:, ::-1][:, :5]

                for idx in range(len(predictions)):
                    pred_list.append(predictions[idx])
                    truth_list.append(y.detach().cpu().numpy()[idx])
                    top_k_list.append(top_k_predictions[idx])
                loss_list.append(loss.item())

        self.result = self.result_summary(truth_list, pred_list, top_k_list, loss_list)

    def result_summary(self, truth_list, pred_list, top_k_list, loss_list):
        result_dict = dict()
        result_dict['acc'] = accuracy_score(truth_list, pred_list)*100
        result_dict['uar'] = recall_score(truth_list, pred_list, average="macro")*100
        result_dict['top5_acc'] = (np.sum(top_k_list == np.array(truth_list).reshape(len(truth_list), 1)) / len(truth_list))*100
        result_dict['conf'] = np.round(confusion_matrix(truth_list, pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(loss_list)
        result_dict["sample"] = len(truth_list)
        return result_dict
