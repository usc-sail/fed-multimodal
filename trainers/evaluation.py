import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
warnings.filterwarnings('ignore')


class EvalMetric(object):
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.pred_list = list()
        self.truth_list = list()
        self.top_k_list = list()
        self.loss_list = list()
        
    def append_classification_results(
        self, 
        labels,
        outputs,
        loss
    ):
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        top_k_predictions = np.argsort(outputs.detach().cpu().numpy(), axis = 1)[:, ::-1][:, :5]
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
            self.top_k_list.append(top_k_predictions[idx])
        self.loss_list.append(loss.item())
        
        
    def append_multilabel_results(
        self, 
        labels,
        outputs,
        loss
    ):
        predictions = np.array(nn.Sigmoid()(outputs).detach().cpu().numpy() > 0.5, dtype=int)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        self.loss_list.append(loss.item())
        
    def classification_summary(
        self, 
        return_auc: bool=False
    ):
        result_dict = dict()
        result_dict['acc'] = accuracy_score(self.truth_list, self.pred_list)*100
        result_dict['uar'] = recall_score(self.truth_list, self.pred_list, average="macro")*100
        result_dict['top5_acc'] = (np.sum(self.top_k_list == np.array(self.truth_list).reshape(len(self.truth_list), 1)) / len(self.truth_list))*100
        result_dict['conf'] = np.round(confusion_matrix(self.truth_list, self.pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["sample"] = len(self.truth_list)
        result_dict['f1'] = f1_score(self.truth_list, self.pred_list, average='macro')*100
        if return_auc: result_dict['auc'] = roc_auc_score(self.truth_list, self.pred_list)*100
        return result_dict

    def multilabel_summary(self):
        num_recordings, num_classes = np.shape(np.array(self.truth_list))
        A = self.compute_confusion_matrices(np.array(self.truth_list), np.array(self.pred_list))
        f_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if 2 * tp + fp + fn:
                f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                f_measure[k] = float('nan')

        result_dict = dict()
        result_dict['acc'] = self.multilabel_acc(np.array(self.truth_list), np.array(self.pred_list))*100
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict['macro_f'] = np.nanmean(f_measure)*100
        result_dict["sample"] = len(self.truth_list)
        return result_dict
    
    # Compute recording-wise accuracy.
    def multilabel_acc(
        self, 
        truth_list, 
        pred_list
    ):
        num_recordings, num_classes = np.shape(np.array(truth_list))
        num_correct_recordings = 0
        for i in range(num_recordings):
            if np.all(truth_list[i, :]==pred_list[i, :]):
                num_correct_recordings += 1
        return float(num_correct_recordings) / float(num_recordings)
    
    
    def compute_confusion_matrices(
        self, 
        labels, 
        outputs, 
        normalize=False
    ):
        # Compute a binary confusion matrix for each class k:
        #
        #     [TN_k FN_k]
        #     [FP_k TP_k]
        #
        # If the normalize variable is set to true, then normalize the contributions
        # to the confusion matrix by the number of labels per recording.
        num_recordings, num_classes = np.shape(labels)
        A = np.zeros((num_classes, 2, 2))
        
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
        return A