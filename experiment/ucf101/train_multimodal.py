import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from copy import deepcopy
from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'dataloader'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'trainers'))

from dataload_manager import dataload_manager
from mm_models import audio_video_classifier
from client_trainer import Client
from server_trainer import Server


# define feature len mapping
feature_len_dict = {'mobilenet_v2': 1280, 'mfcc': 80}


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='ucf101')
    parser.add_argument('--audio_feat', default='mfcc')
    parser.add_argument('--video_feat', default='mobilenet_v2')
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--num_epochs', default=300)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--fed_alg', default='fed_avg')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--data_dir', default='/media/data/projects/speech-privacy/fed-multimodal/')
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    args = parser.parse_args()

    # data manager
    dm = dataload_manager(args)
    
    # find device
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    save_result_df = pd.DataFrame()

    # We perform 3 fold experiments
    for fold_idx in range(3):
        # load all data
        train_audio = dm.load_audio_feat(fold_idx=fold_idx+1, split_type='train')
        dev_audio = dm.load_audio_feat(fold_idx=fold_idx+1, split_type='dev')
        test_audio = dm.load_audio_feat(fold_idx=fold_idx+1, split_type='test')
        train_video = dm.load_video_feat(fold_idx=fold_idx+1, split_type='train')
        dev_video = dm.load_video_feat(fold_idx=fold_idx+1, split_type='dev')
        test_video = dm.load_video_feat(fold_idx=fold_idx+1, split_type='test')
        
        dataloader_dict = dict()
        for client_id in train_audio:
            dataloader_dict[client_id] = dm.set_dataloader(train_audio[client_id], train_video[client_id], shuffle=True)
        dataloader_dict['dev'] = dm.set_dataloader(dev_audio, dev_video, shuffle=False)
        dataloader_dict['test'] = dm.set_dataloader(test_audio, test_video, shuffle=False)
        
        # number of clients
        num_of_clients, client_ids = len(train_audio), list(train_audio.keys())

        # set seeds
        set_seed(8)

        # loss function
        criterion = nn.NLLLoss().to(device)

        # Define the model
        global_model = audio_video_classifier(num_classes=51, 
                                              audio_input_dim=feature_len_dict["mfcc"], 
                                              video_input_dim=feature_len_dict["mobilenet_v2"])
        
        global_model = global_model.to(device)

        # initialize server
        server = Server(args, global_model, device, criterion)
        server.initialize_log(fold_idx+1)
        server.sample_clients(num_of_clients, sample_rate=0.2)
        row_df = pd.DataFrame(index=['fold'+str(int(fold_idx+1))])
        
        # set seeds again
        set_seed(8)

        # Training steps
        for epoch in range(int(args.num_epochs)):
            
            # define list varibles that saves the weights, loss, num_sample, etc.
            server.initialize_epoch_updates(epoch)
            
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                # Local training
                dataloader = dataloader_dict[client_id]
                client = Client(args, device, criterion, dataloader, copy.deepcopy(server.global_model))
                client.update_weights()
                # server append updates
                server.model_updates.append(copy.deepcopy(client.get_parameters()))
                server.num_samples_list.append(client.result['sample'])
                server.result_dict[epoch]['train'] = client.result
                del client
            # pdb.set_trace()
            
            # 2.3 load new global weights
            server.average_weights()

            # 3. Calculate avg validation accuracy/uar over all selected users at every epoch
            print('--------------------------------------------------------------------------------------')
            with torch.no_grad():
                server.inference(dataloader_dict['dev'])
                server.result_dict[epoch]['dev'] = server.result
                server.log_result(data_split='dev')

                # 4. Perform the test on holdout set
                server.inference(dataloader_dict['test'])
                server.result_dict[epoch]['test'] = server.result
                server.log_result(data_split='test')
            
            print('--------------------------------------------------------------------------------------')
            server.log_epoch_result(metric='acc')
            print('--------------------------------------------------------------------------------------')

        # Performance save code
        row_df = server.summarize_results()
        save_result_df = pd.concat([save_result_df, row_df])
        
    # Calculate the average of the 5-fold experiments
    row_df = pd.DataFrame(index=['average'])
    row_df['acc'] = np.mean(save_result_df['acc'])
    row_df['top5_acc'] = np.mean(save_result_df['top5_acc'])
    row_df['uar'] = np.mean(save_result_df['uar'])
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(Path(args.data_dir).joinpath('log', args.dataset, server.model_setting_str).joinpath('result.csv')))
