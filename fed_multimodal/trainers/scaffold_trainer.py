
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import ScaffoldOptimizer, NovaOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientScaffold(object):
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

    def set_control(
        self, 
        server_control, 
        client_control
    ):
        self.server_control = server_control
        self.client_control = client_control
        self.set_control_device(self.client_control, True)
    
    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # optimizer
        optimizer = ScaffoldOptimizer(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=1e-05,
            momentum=0.9
        )

        # last global model
        last_global_model = copy.deepcopy(self.model)
        # get total training batch number
        n_total_bs = int(self.args.local_epochs * len(self.dataloader))
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
                self.model.zero_grad()
                optimizer.zero_grad()
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                
                # forward
                outputs, _ = self.model(
                    x_a.float(), 
                    x_b.float(), 
                    l_a, 
                    l_b
                )

                if not self.multilabel: 
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

                optimizer.step(
                    server_control=copy.deepcopy(self.server_control),
                    client_control=copy.deepcopy(self.client_control)
                )
                
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
        # get deltas
        delta_model = self.get_delta_model(
            last_global_model, 
            self.model
        )
        
        # update contral variables
        client_control, delta_control = self.update_local_control(
            delta_model=delta_model,
            server_control=self.server_control,
            client_control=self.client_control,
            steps=n_total_bs,
            lr=self.args.learning_rate,
        )

        # save control variables
        self.client_control = copy.deepcopy(client_control)
        self.delta_control = copy.deepcopy(delta_control)

        # set to cpu devices
        self.set_control_device(self.client_control, False)

        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()


    def get_delta_model(self, model0, model1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    def update_local_control(
        self, 
        delta_model, 
        server_control,
        client_control, 
        steps, 
        lr
    ):
        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_model.keys():
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci
        return new_control, delta_control

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
    