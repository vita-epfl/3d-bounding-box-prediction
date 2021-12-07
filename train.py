import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import matplotlib.pyplot as plt

import DataLoader
import network
import utils

       
class args():
    def __init__(self):
        #self.jaad_dataset = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/preprocessed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.jaad_dataset = '/home/yju/JTA/preprocessed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'train'
        self.from_file    = True #read dataset from csv file or reprocess data
        self.save         = True
        #self.file         = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/jta_train_16_16.csv'
        self.file         = '/home/yju/JTA/jta_train_16_16.csv'
        #self.save_path    = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/jta_train_16_16.csv'
        self.save_path    = '/home/yju/JTA/jta_train_16_16.csv'
        #self.model_path    = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/models/pv_lstm.pkl'
        self.model_path    = '/home/yju/JTA/models/pv_lstm_trained_capacity1024.pkl'
        self.loader_workers = 12
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        self.batch_size     = 128 # 32, 64
        self.n_epochs       = 150
        self.hidden_size    = 1024
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 16
        self.stride = 16
        self.skip   = 1
        self.task   = 'bounding_box'
        self.use_scenes = False      
        self.lr = 0.001 # inc lr
        self.save_subset = False
        self.subset = 1000

        
        
def training(args, train, val):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                     threshold = 1e-8, verbose=True)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    train_s_scores = []
    val_s_scores   = []
    
    print('='*100)
    print('Training ...')

    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size))

    for epoch in range(args.n_epochs):
        start = time.time()
        
        avg_epoch_train_s_loss = 0
        avg_epoch_val_s_loss   = 0
        
        ade  = 0
        fde  = 0
        aiou = 0
        fiou = 0
        
        counter = 0
        for idx, (obs_s, target_s, obs_p, target_p) in enumerate(train):
            counter += 1
            obs_s    = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_p    = obs_p.to(device='cuda')
            target_p = target_p.to(device='cuda')
            
            net.zero_grad()
            speed_preds = net(speed=obs_s, pos=obs_p) #[100,16,6]
            speed_loss  = mse(speed_preds, target_s)/100
    
            loss = speed_loss
            loss.backward()
            optimizer.step()
            
            avg_epoch_train_s_loss += float(speed_loss)
            
        avg_epoch_train_s_loss /= counter
        train_s_scores.append(avg_epoch_train_s_loss)
        
        counter=0
        state_preds = []
        state_targets = []
        for idx, (obs_s, target_s, obs_p, target_p) in enumerate(val):
            counter+=1
            obs_s    = obs_s.to(device='cuda')
            target_s = target_s.to(device='cuda')
            obs_p    = obs_p.to(device='cuda')
            target_p = target_p.to(device='cuda')
            
            with torch.no_grad():
                speed_preds = net(speed=obs_s, pos=obs_p)
                speed_loss    = mse(speed_preds, target_s)/100
                
                avg_epoch_val_s_loss += float(speed_loss)
                
                preds_p = utils.speed2pos(speed_preds, obs_p, is_3D=True)
                ade += float(utils.ADE_c(preds_p, target_p, is_3D=True))
                fde += float(utils.FDE_c(preds_p, target_p, is_3D=True))
                aiou += float(utils.AIOU(preds_p, target_p, is_3D=True))
                fiou += float(utils.FIOU(preds_p, target_p, is_3D=True))
                
                # state_preds.extend(crossing_preds)
                # state_targets.extend(target_c)
                #intent_preds.extend(intentions)
                #intent_targets.extend(label_c)
            
            avg_epoch_val_s_loss += float(speed_loss)
            
        avg_epoch_val_s_loss /= counter
        val_s_scores.append(avg_epoch_val_s_loss)
        
        ade  /= counter
        fde  /= counter     
        aiou /= counter #  check if viou
        fiou /= counter
        
        print('e:', epoch, '| ts: %.4f'% avg_epoch_train_s_loss, 
              # '| tc: %.4f'% avg_epoch_train_c_loss, 
              '| vs: %.4f'% avg_epoch_val_s_loss, 
              # '| vc: %.4f'% avg_epoch_val_c_loss, 
              '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
              '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, 
              #'| state_acc: %.4f'% avg_acc, 
              # '| intention_acc: %.4f'% intent_acc, 
              '| t:%.4f'%(time.time()-start))
    
    print('='*100) 
    print('Saving ...')
    torch.save(net.state_dict(), args.model_path)
    print('Done !')
    
    return train_s_scores, val_s_scores
    
if __name__ == '__main__':
    args = args()

    net = network.PV_LSTM(args).to(args.device)

    args.dtype = 'train'
    train = DataLoader.data_loader(args)
    print(train.dataset.data.shape)

    args.dtype = 'val'
    args.save_path = args.save_path.replace('train', 'val')
    args.file = args.file.replace('train', 'val')
    val = DataLoader.data_loader(args)
    print(val.dataset.data.shape)
    
    train_s_scores, val_s_scores = training(args, train, val)
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(train_s_scores, 'b')
    plt.plot(val_s_scores, 'r')
    plt.title('Training losses '+ str(args.lr))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'])
    plt.show()
    plt.savefig('/home/yju/yju_testing/training_loss_lr0.001_e150.png', dpi=100)
    
