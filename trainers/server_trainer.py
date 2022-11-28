import json
import numpy as np
import collections
import pandas as pd
import copy, pdb, time, warnings, torch

from torch import nn
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score

from evaluation import EvalMetric

# logging format
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Server(object):
    def __init__(
        self, 
        args, 
        model, 
        device,
        criterion,
        client_ids
    ):
        self.args = args
        self.global_model = model
        self.device = device
        self.result_dict = dict()
        self.criterion = criterion
        self.model_setting_str = self.get_model_setting()
        self.client_ids = client_ids
        self.multilabel = True if args.dataset == 'ptb-xl' else False

        if self.args.fed_alg == 'scaffold':
            self.server_control = self.init_control(model)
            self.set_control_device(self.server_control, True)
            self.client_controls = {
                client_id: self.init_control(model) for client_id in self.client_ids
            }

    def set_client_control(
        self,
        client_id,
        client_control
    ):
        self.client_controls[client_id] = client_control
        
    def initialize_log(
        self, 
        fold_idx: int=1
    ):
        # log saving path
        self.fold_idx = fold_idx
        self.log_path = Path(self.args.data_dir).joinpath(
            'log', 
            self.args.fed_alg,
            self.args.dataset, 
            self.model_setting_str, 
            f'fold{fold_idx}', 
            'raw_log'
        )
        self.result_path = Path(self.args.data_dir).joinpath(
            'log', 
            self.args.fed_alg,
            self.args.dataset, 
            self.model_setting_str, 
            f'fold{fold_idx}'
        )
        Path.mkdir(self.log_path, parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(
            str(self.log_path), 
            filename_suffix=f'_{self.model_setting_str}'
        )
        
        self.best_test_dict = list()
        
    def get_model_setting(self):
        # Return model setting
        if self.args.dataset in ['mit10', 'mit51', 'ucf101', 'crema_d']:
            model_setting_str = f'{self.args.audio_feat}_{self.args.video_feat}'
            model_setting_str += '_alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['uci-har']:
            model_setting_str = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            model_setting_str += '_alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['extrasensory', 'ku-har']:
            model_setting_str = f'{self.args.acc_feat}_{self.args.gyro_feat}'
        elif self.args.dataset in ['extrasensory_watch']:
            model_setting_str = f'{self.args.acc_feat}_{self.args.watch_feat}'
        elif self.args.dataset in ['ptb-xl']:
            model_setting_str = 'i_to_avf_v1_to_v6'
        elif self.args.dataset in ['meld']:
            model_setting_str = f'{self.args.audio_feat}_{self.args.text_feat}'
        else:
            raise ValueError(f'Data set not support {self.args.dataset}')
        # training settings: local epochs, learning rate, batch size, client sample rate
        model_setting_str += '_hid'+str(self.args.hid_size)
        model_setting_str += '_le'+str(self.args.local_epochs)
        model_setting_str += '_lr' + str(self.args.learning_rate).replace('.', '')
        model_setting_str += '_bs'+str(self.args.batch_size)
        model_setting_str += '_sr'+str(self.args.sample_rate).replace('.', '')
        model_setting_str += '_ep'+str(self.args.num_epochs)
        if self.args.att: model_setting_str += f'_{self.args.att_name}'
        
        # FL simulations: missing modality, label noise, missing labels
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
            idxs_clients = np.random.choice(
                range(num_of_clients), 
                int(sample_rate * num_of_clients), 
                replace=False
            )
            self.clients_list.append(idxs_clients)

    def initialize_epoch_updates(self, epoch):
        # Initialize updates
        self.epoch = epoch
        self.model_updates = list()
        self.num_samples_list = list()
        self.delta_controls = list()
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

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.global_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters]) / 1000
        logging.info(f'Number of Parameters: {num_params} K')
        return num_params
    
    def inference(self, dataloader):
        self.global_model.eval()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        for batch_idx, batch_data in enumerate(dataloader):
                
            self.global_model.zero_grad()
            x_a, x_b, l_a, l_b, y = batch_data
            x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
            l_a, l_b = l_a.to(self.device), l_b.to(self.device)
            
            # forward
            outputs, _ = self.global_model(
                x_a.float(), 
                x_b.float(), 
                l_a, 
                l_b
            )
            if not self.multilabel: 
                outputs = torch.log_softmax(outputs, dim=1)
            loss = self.criterion(outputs, y)
            
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

    def log_classification_result(
            self, 
            data_split: str, 
            metric: str='uar'
        ):
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
        if metric == 'acc':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Acc: {acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
        else:
            logging.info(f'{data_split} set, Loss: {loss:.3f}, UAR: {uar:.2f}%, Top-1 Acc: {acc:.2f}%')

        # loggin to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'UAR/{data_split}', uar, self.epoch)
        self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)
        
    def log_multilabel_result(
        self, 
        data_split: str, 
        metric: str='macro_f'
    ):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            macro_f = np.mean([data['macro_f'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            macro_f = self.result_dict[self.epoch][data_split]['macro_f']

        # logging to console
        if data_split == 'train': 
            logging.info(
                f'Current Round: {self.epoch}'
            )
        logging.info(
            f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {macro_f:.2f}%, Top-1 Acc: {acc:.2f}%'
        )

        # logging to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'Macro-F1/{data_split}', macro_f, self.epoch)

    def save_result(
        self, 
        file_path
    ):
        jsonString = json.dumps(
            self.result_dict, 
            indent=4
        )
        jsonFile = open(str(file_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def save_train_updates(
        self, 
        model_updates: dict, 
        num_sample: int, 
        result: dict,
        delta_control=None
    ):
        self.model_updates.append(model_updates)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)
        self.delta_controls.append(delta_control)

    def log_epoch_result(
        self, 
        metric: str='acc'
    ):
        if len(self.best_test_dict) == 0:
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']

        if self.result_dict[self.epoch]['dev'][metric] > self.best_dev_dict[metric]:
            # Save best model and training history
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            torch.save(
                deepcopy(self.global_model.state_dict()), 
                str(self.result_path.joinpath('model.pt'))
            )
        
        # log dev results
        best_dev_acc = self.best_dev_dict['acc']
        if not self.multilabel:
            best_dev_uar = self.best_dev_dict['uar']
            best_dev_top5_acc = self.best_dev_dict['top5_acc']
        else:
            best_dev_macro_f1 = self.best_dev_dict['macro_f']

        # log test results
        best_test_acc = self.best_test_dict['acc']
        if not self.multilabel:
            best_test_uar = self.best_test_dict['uar']
            best_test_top5_acc = self.best_test_dict['top5_acc']
        else:
            best_test_macro_f1 = self.best_test_dict['macro_f']
            
        # logging
        logging.info(f'Best epoch {self.best_epoch}')
        if metric == 'acc':
            logging.info(f'Best dev Top-1 Acc {best_dev_acc:.2f}%, Top-5 Acc {best_dev_top5_acc:.2f}%')
            logging.info(f'Best test Top-1 Acc {best_test_acc:.2f}%, Top-5 Acc {best_test_top5_acc:.2f}%')
        elif metric == 'macro_f':
            logging.info(f'Best dev Macro-F1 {best_dev_macro_f1:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test Macro-F1 {best_test_macro_f1:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        else:
            logging.info(f'Best dev UAR {best_dev_uar:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test UAR {best_test_uar:.2f}%, Top-1 Acc {best_test_acc:.2f}%')

    def summarize_results(self):
        row_df = pd.DataFrame(index=[f'fold{self.fold_idx}'])
        row_df['acc']  = self.best_test_dict['acc']
        if not self.multilabel:
            row_df['top5_acc'] = self.best_test_dict['top5_acc']
            row_df['uar'] = self.best_test_dict['uar']
        else:
            row_df['macro_f'] = self.best_test_dict['macro_f']
        return row_df

    def summarize_dict_results(self):
        result = dict()
        result['acc'] = self.best_test_dict['acc']
        if not self.multilabel:
            result['top5_acc'] = self.best_test_dict['top5_acc']
            result['uar'] = self.best_test_dict['uar']
        else:
            result['macro_f'] = self.best_test_dict['macro_f']
        return result

    def average_weights(self):
        """
        Returns the average of the weights.
        """
        # there are no samples, return
        if len(self.num_samples_list) == 0: 
            return
        total_num_samples = np.sum(self.num_samples_list)
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            w_avg[key] = self.model_updates[0][key]*(self.num_samples_list[0]/total_num_samples)
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                w_avg[key] += torch.div(self.model_updates[i][key]*self.num_samples_list[i], total_num_samples)
        self.global_model.load_state_dict(copy.deepcopy(w_avg))

        # update global control if algorithm is scaffold
        if self.args.fed_alg == 'scaffold':
            self.update_server_control()

    def update_server_control(self):
        # update server control
        total_num_samples = np.sum(self.num_samples_list)
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)

        for key in delta_avg.keys():
            delta_avg[key] = self.delta_controls[0][key]*(self.num_samples_list[0]/total_num_samples)

        for key in delta_avg.keys():
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(self.delta_controls[idx][key]*self.num_samples_list[idx], total_num_samples)
            new_control[key] = new_control[key] - delta_avg[key]
        self.server_control = copy.deepcopy(new_control)

    def set_save_json_file(
        self, 
        file_path
    ):
        self.save_json_path = file_path


    def save_json_file(
        self, 
        data_dict, 
        data_path
    ):
        jsonString = json.dumps(
            data_dict, 
            indent=4
        )
        jsonFile = open(str(data_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    # scaffold init contral
    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control

    def set_control_device(
        self, 
        control, 
        device=True
    ):
        for name in control.keys():
            if device == True:
                control[name] = control[name].to(self.device)
            else:
                control[name] = control[name].cpu()
