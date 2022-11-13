import logging
import pandas as pd
import numpy as np
import collections
import copy, pdb, time, warnings, torch

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils import data
from copy import deepcopy

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(levelname)-3s ==> %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Server(object):
    def __init__(self, args, model, device, criterion):
        self.args = args
        self.global_model = model
        self.device = device
        self.result_dict = dict()
        self.criterion = criterion
        self.model_setting_str = self.get_model_setting()
        
    def initialize_log(self, fold_idx=1):
        # log saving path
        self.fold_idx = fold_idx
        self.log_path = Path(self.args.data_dir).joinpath('log', self.args.dataset, self.model_setting_str, f'fold{fold_idx}', 'raw_log')
        self.result_path = Path(self.args.data_dir).joinpath('log', self.args.dataset, self.model_setting_str, f'fold{fold_idx}')
        Path.mkdir(self.log_path, parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(str(self.log_path), filename_suffix=f'_{self.model_setting_str}')
        
    def get_model_setting(self):
        # Return model setting
        model_setting_str = 'alpha'+str(self.args.alpha).replace('.', '')
        model_setting_str += '_le'+str(self.args.local_epochs)
        model_setting_str += '_lr' + str(self.args.learning_rate).replace('.', '')
        model_setting_str += '_bs'+str(self.args.batch_size)
        model_setting_str += '_sr'+str(self.args.sample_rate).replace('.', '')
        if self.args.missing_modality == True:
            model_setting_str += '_mm'+str(self.args.missing_modailty_rate).replace('.', '')
        if self.args.label_nosiy == True:
            model_setting_str += '_ln'+str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label == True:
            model_setting_str += '_ml'+str(self.args.missing_label_rate).replace('.', '')
        return model_setting_str
    
    def sample_clients(self, num_of_clients, sample_rate=0.1):
        # Sample clients per round
        self.clients_list = list()
        for epoch in range(int(self.args.num_epochs)):
            np.random.seed(epoch)
            idxs_clients = np.random.choice(range(num_of_clients), int(sample_rate * num_of_clients), replace=False)
            self.clients_list.append(idxs_clients)

    def initialize_epoch_updates(self, epoch):
        # Initialize updates
        self.epoch = epoch
        self.model_updates = list()
        self.num_samples_list = list()
        self.result_dict[self.epoch] = dict()
        self.result_dict[self.epoch]['train'] = list()
        self.result_dict[self.epoch]['dev'] = list()
        self.result_dict[self.epoch]['test'] = list()
        
    def get_parameters(self):
        # Return model parameters
        return self.global_model.state_dict()
    
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
    
    def update(self, learning_rate, label_list, fed_setting):
        if fed_setting == 'fed_avg': self.update_weights(learning_rate, label_list)
        else: self.update_gradients(learning_rate, label_list)
    
    def inference(self, dataloader):
        self.global_model.eval()

        # prediction and truths
        pred_list, truth_list, top_k_list, loss_list = list(), list(), list(), list()
        for batch_idx, batch_data in enumerate(dataloader):
                
            self.global_model.zero_grad()
            x_a, x_b, y = batch_data
            x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
            
            # forward
            outputs = self.global_model(x_a.float(), x_b.float())
            outputs = torch.log_softmax(outputs, dim=1)
            loss = self.criterion(outputs, y)
            
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            top_k_predictions = np.argsort(outputs.detach().cpu().numpy(), axis = 1)[:, ::-1][:, :5]

            loss_list.append(loss.item())
            for idx in range(len(predictions)):
                pred_list.append(predictions[idx])
                truth_list.append(y.detach().cpu().numpy()[idx])
                top_k_list.append(top_k_predictions[idx])
        self.result = self.result_summary(truth_list, pred_list, top_k_list, loss_list)

    def result_summary(self, truth_list, pred_list, top_k_list, loss_list):
        # save result summary
        result_dict = dict()
        result_dict['acc'] = accuracy_score(truth_list, pred_list)*100
        result_dict['uar'] = recall_score(truth_list, pred_list, average="macro")*100
        result_dict['top5_acc'] = np.sum(top_k_list == np.array(truth_list).reshape(len(truth_list), 1)) / len(truth_list)*100
        result_dict['conf'] = np.round(confusion_matrix(truth_list, pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(loss_list)
        result_dict["sample"] = len(truth_list)
        return result_dict

    def log_result(self, data_split):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            uar = np.mean([data['uar'] for data in self.result_dict[self.epoch][data_split]])
            top5_acc = np.mean([data['top5_acc'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            uar = self.result_dict[self.epoch][data_split]['uar']
            top5_acc = self.result_dict[self.epoch][data_split]['top5_acc']

        # loggin console
        if data_split == 'train': logging.info(f'Current Round: {self.epoch}')
        logging.info(f'{data_split} set, Loss: {loss:.3f}, Acc: {acc:.2f}, Top-5 Acc: {top5_acc:.2f}')

        # loggin to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'UAR/{data_split}', uar, self.epoch)
        self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)

    def save_result(self):
        f = open(str(self.model_result_path.joinpath('results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

    def save_train_updates(self, model_updates, num_sample, result):
        self.model_updates.append(model_updates)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)

    def log_epoch_result(self, metric='acc'):
        if self.epoch == 0:
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']

        if self.result_dict[self.epoch]['dev'][metric] > self.best_dev_dict[metric]:
            # Save best model and training history
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            torch.save(deepcopy(self.global_model.state_dict()), str(self.result_path.joinpath('model.pt')))
        
        # log dev results
        best_dev_acc = self.best_dev_dict['acc']
        best_dev_uar = self.best_dev_dict['uar']
        best_dev_top5_acc = self.best_dev_dict['top5_acc']

        # log test results
        best_test_acc = self.best_test_dict['acc']
        best_test_uar = self.best_test_dict['uar']
        best_test_top5_acc = self.best_test_dict['top5_acc']
        
        # logging
        logging.info(f'Best epoch {self.best_epoch}')
        logging.info(f'Best dev acc {best_dev_acc:.2f}%, top-5 acc {best_dev_top5_acc:.2f}%')
        logging.info(f'Best test rec {best_test_acc:.2f}%, top-5 acc {best_test_top5_acc:.2f}%')

    def summarize_results(self):
        row_df = pd.DataFrame(index=[f'fold{self.fold_idx}'])
        row_df['acc']  = self.best_test_dict['acc']
        row_df['top5_acc'] = self.best_test_dict['top5_acc']
        row_df['uar']  = self.best_test_dict['uar']
        return row_df

    def average_weights(self):
        """
        Returns the average of the weights.
        """
        total_num_samples = np.sum(self.num_samples_list)
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            w_avg[key] = self.model_updates[0][key]*(self.num_samples_list[0]/total_num_samples)
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                w_avg[key] += torch.div(self.model_updates[i][key]*self.num_samples_list[i], total_num_samples)
        self.global_model.load_state_dict(copy.deepcopy(w_avg))

