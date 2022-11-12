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


def pred_summary(y_true, y_pred):
    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    return result_dict


def result_summary(step_outputs):
    loss_list, y_true, y_pred = [], [], []
    for step in range(len(step_outputs)):
        for idx in range(len(step_outputs[step]['pred'])):
            y_true.append(step_outputs[step]['truth'][idx])
            y_pred.append(step_outputs[step]['pred'][idx])
        loss_list.append(step_outputs[step]['loss'])

    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    result_dict['loss'] = np.mean(loss_list)
    result_dict['num_samples'] = len(y_pred)
    return result_dict, y_pred, y_true


def effective_sample(label_list, num_class):
    beta = 0.9999
    class_dict = collections.Counter(label_list)
    
    if len(class_dict) < num_class:
        minimum_val = 1
    else:
        minimum = min(class_dict, key=class_dict.get)
        minimum_val = class_dict[minimum]
    
    cls_num_list = []
    for i in range(num_class):
        if i in class_dict: cls_num_list.append(class_dict[i])
        else: cls_num_list.append(minimum_val)
        
    effective_num = 1.0 - np.power(0.999, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    return per_cls_weights


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
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                
                # forward
                outputs = self.model(x_a.float(), x_b.float())
                outputs = torch.log_softmax(outputs, dim=1)

                # backward
                loss = self.criterion(outputs, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
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
        result_dict['acc'] = accuracy_score(truth_list, pred_list)
        result_dict['uar'] = recall_score(truth_list, pred_list, average="macro")
        result_dict['top5_acc'] = np.sum(top_k_list == np.array(truth_list).reshape(len(truth_list), 1)) / len(truth_list)
        result_dict['conf'] = np.round(confusion_matrix(truth_list, pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(loss_list)
        result_dict["sample"] = len(truth_list)
        return result_dict

    def inference(self):
        self.model.eval()
        step_outputs = []
        
        for batch_idx, batch_data in enumerate(self.dataloader):
            
            if 'bert_' in self.config['feature_type']:
                x1, x2, y = batch_data
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                logits = self.model(x1.float(), x2.float())
            elif 'bert' in self.config['feature_type'] or 'raw_' in self.config['feature_type'] or self.config['feature_type'] == 'mel_spec':
                x, y, l = batch_data
                # _, indices = torch.sort(l, descending=True)
                # x, l, y = x[indices], l[indices], y[indices]
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                logits = self.model(x.float(), l)
            else:
                x, y = batch_data
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x.float())
            
            loss = self.criterion(logits, y)
            
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
            truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
            step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict, y_pred, y_true = result_summary(step_outputs)
        
        self.result = result_dict
        self.test_true = y_true
        self.test_pred = y_pred
