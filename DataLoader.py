import torch
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time
#import import_ipynb
import utils
import args


class myJTA(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        
        if(args.from_file):
            sequence_centric = pd.read_csv(args.file)
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
            
            if args.save_subset:
                sequence_centric = sequence_centric.head(args.subset)
                
                if args.save:
                    sequence_centric.to_csv(args.save_path, index=False)
            
            self.data = sequence_centric.copy().reset_index(drop=True)
            
        else:
            #read data
            print('Processing data ...')
            sequence_centric = pd.DataFrame()
            for file in glob.glob(os.path.join(args.jaad_dataset,args.dtype,"*")):
                df = pd.read_csv(file)
                if not df.empty:
                    print(file)
                    #reset index
                    df = df.reset_index(drop = True)
                    
                    df['bounding_box'] = df[['x', 'y', 'z', 'w', 'h', 'd']].apply(lambda row: \
                                    [round(row.x,4), round(row.y,4), round(row.z,4),\
                                     round(row.w,4), round(row.h,4), round(row.d,4)], axis=1)
                    
                    bb = df.groupby(['ID'])['bounding_box'].apply(list).reset_index(name='bounding_box')
                    s = df.groupby(['ID'])['scenefolderpath'].apply(list).reset_index(name='scenefolderpath').drop(columns='ID')
                    #f = df.groupby(['ID'])['filename'].apply(list).reset_index(name='filename').drop(columns='ID')
                    f = df.groupby(['ID'])['frame'].apply(list).reset_index(name='frame').drop(columns='ID')
                    d = bb.join(s).join(f)
                    
                    d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < args.input + args.output)].index)
                    d = d.reset_index(drop=True)
                    
                    INPUT = args.input
                    OUTPUT = args.output
                    STRIDE = args.stride
                    bounding_box_o = np.empty((0,INPUT,6))
                    bounding_box_t = np.empty((0,OUTPUT,6))
                    scene_o = np.empty((0,INPUT))
                    file = np.empty((0,INPUT))   
                    ind = np.empty((0,1))
        
                    for i in range(d.shape[0]):
                        ped = d.loc[i]
                        k = 0
                        while (k+INPUT+OUTPUT) <= len(ped.bounding_box):
                            obs_frames = ped.frame[k:k+INPUT]
                            pred_frames = ped.frame[k+INPUT:k+INPUT+OUTPUT]
                            if utils.check_continuity(obs_frames) and utils.check_continuity(pred_frames):
                                ind = np.vstack((ind, ped['ID']))
                                bounding_box_o = np.vstack((bounding_box_o, np.array(ped.bounding_box[k:k+INPUT]).reshape(1,INPUT,6)))
                                bounding_box_t = np.vstack((bounding_box_t, np.array(ped.bounding_box[k+INPUT:k+INPUT+OUTPUT]).reshape(1,OUTPUT,6)))  
                                scene_o = np.vstack((scene_o, np.array(ped.scenefolderpath[k:k+INPUT]).reshape(1,INPUT)))
                                filename = ['%04d'%int(x) +'.jpg' for x in obs_frames]
                                file = np.vstack((file, np.array(filename)))    
                            else:
                                pass
                        
                            k += STRIDE
                    
                    dt = pd.DataFrame({'ID':ind.reshape(-1)})
                    data = pd.DataFrame({'bounding_box':bounding_box_o.reshape(-1, 1, INPUT, 6).tolist(),
                                         'future_bounding_box':bounding_box_t.reshape(-1, 1, OUTPUT, 6).tolist(),
                                         'scenefolderpath':scene_o.reshape(-1,INPUT).tolist(),
                                         'filename':file.reshape(-1,INPUT).tolist()
                                         })
                    data.bounding_box = data.bounding_box.apply(lambda x: x[0])
                    data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
                    data = dt.join(data)
                    
                    sequence_centric = sequence_centric.append(data, ignore_index=True)
            
        if args.save:
            sequence_centric.to_csv(args.save_path, index=False)
            
        self.data = sequence_centric.copy().reset_index(drop=True)
            
        self.args = args
        self.dtype = args.dtype
        print(args.dtype, "set loaded")
        print('*'*30)
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        seq = self.data.iloc[index]
        outputs = []

        observed = torch.tensor(np.array(seq.bounding_box))
        future = torch.tensor(np.array(seq.future_bounding_box))
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.args.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))
        
        if 'intention' in self.args.task:
            true_cross = torch.tensor([seq.crossing_true[i] for i in range(0,self.args.output,self.args.skip)])
            true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
            true_cross = torch.cat((true_non_cross.unsqueeze(1), true_cross.unsqueeze(1)), dim=1)
            cross_label = torch.tensor(seq.label)
            outputs.append(true_cross.type(torch.float32))
            outputs.append(cross_label.type(torch.float32))
              
        if self.args.use_scenes:     
            scene_paths = [os.path.join(seq["scenefolderpath"][frame], '%.4d'%seq.ID, seq["filename"][frame]) 
                           for frame in range(0,self.args.input,self.args.skip)]
        
            for i in range(len(scene_paths)):
                scene_paths[i] = scene_paths[i].replace('haziq-data', 'smailait-data').replace('scene', 'resized_scenes')

            scenes = torch.tensor([])
            for i, path in enumerate(scene_paths):
                scene = Image.open(path)
                scene = self.scene_transforms(scene)
                scenes = torch.cat((scenes, scene.unsqueeze(0)))
                
            outputs.insert(0, scenes)

        return tuple(outputs)

    def scene_transforms(self, scene):  
        scene = TF.to_tensor(scene)
        
        return scene
    
    
def data_loader(args):
    dataset = myJTA(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return dataloader



if __name__ == '__main__':
    args = args.args()
             
    test = data_loader(args)
    # data = test.dataset.data 
    shape = test.dataset.data.shape  
    #dtype = test.dataset.data.dtypes
    #print(dtype)
    print(shape)
