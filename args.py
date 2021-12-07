# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:30:02 2021

@author: Celinna
"""

class args():
    def __init__(self):
        #self.jaad_dataset = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/preprocessed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.jaad_dataset = '/home/yju/JTA/preprocessed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'train'
        self.from_file    = False #read dataset from csv file or reprocess data
        self.save         = True
       # self.file         = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/jta_train_16_16_100.csv'
        self.file         = '/home/yju/JTA/jta_train_60_60.csv'
       # self.save_path    = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/jta_train_16_16_32.csv'
        self.save_path    = '/home/yju/JTA/jta_train_60_60.csv'
       # self.model_path    = 'C:/Users/Celinna/Desktop/ROBMSC3/VITA/JTA/models/pv_lstm_test_1000.pkl'
        self.model_path    = '/home/yju/JTA/models/multitask_pv_lstm_trained.pkl'
        self.loader_workers = 12
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        self.batch_size     = 128
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.input  = 60
        self.output = 60
        self.stride = 60
        self.skip   = 1
        self.task   = 'bounding_box'
        self.use_scenes = False    
        self.lr = 0.0001
        self.save_subset = False
        self.subset = 2*self.input
      
