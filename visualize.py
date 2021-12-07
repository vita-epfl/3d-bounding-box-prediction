"""
Created on Sat Nov 27 23:06:36 2021

@author: Celinna
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.pyplot import AutoLocator

class visualizeJTA():
    def __init__(self):
        self.K  = np.array([[1158, 0, 960], [0, 1158, 540], [0, 0, 1]]) # int mat
        self.figsize = (12, 8 )
        self.coords = {
            'p1': ['x1','y1','z1'], # left-bottom-front
            'p2': ['x1','y2','z1'], #left-top-front
            'p3': ['x2','y2','z1'], #right-top-front
            'p4': ['x2','y1','z1'], #right-bottom-front
            'p5': ['x1','y1','z2'], #left-bottom-back
            'p6': ['x1','y2','z2'], #left-top-back
            'p7': ['x2','y2','z2'], #right-top-back
            'p8': ['x2','y1','z2'] #right-bottom-back
        }
        
        
    def get_colormap(self, n):
        colors=np.random.rand(n)
        #cmap=plt.cm.RdYlBu_r
        cmap=plt.cm.jet
        c=cmap(colors)
        return c


    def cam2pix_bbox(self, df):
        temp = pd.DataFrame()
        bbox = pd.DataFrame()
        #bbox = df[['frame', 'ID']].copy()

        temp['x1'] = df.x - df.w/2
        temp['y1'] = df.y - df.h/2
        temp['z1'] = df.z - df.d/2
        temp['x2'] = df.x + df.w/2
        temp['y2'] = df.y + df.h/2
        temp['z2'] = df.z + df.d/2

        for point in self.coords.keys():
            coord = self.coords[point]
            x_p = temp[coord[0]] / temp[coord[2]]
            y_p = temp[coord[1]] / temp[coord[2]]
            px = (self.K[0][0] * x_p + self.K[0][2]).astype(int)
            py = (self.K[1][1] * y_p + self.K[1][2]).astype(int)

            #px = np.where(px < 0, 0, px)
            #py = np.where(py < 0, 0, py)

            bbox[point] = list(zip(px, py))

        return bbox

    
    def get_line_coords(self, p1, p2):
        '''
        Get coordinates in required format for line plotting in matplotlib
        '''
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        P = np.vstack((p1,p2))
        P = np.transpose(P)

        return P


    def get_lines(self, temp):
        '''
        Get lines for 3D bbox in 2D plots
        '''
        l1 =    self.get_line_coords(temp.p1, temp.p2)
        l2 =    self.get_line_coords(temp.p2, temp.p3)
        l3 =    self.get_line_coords(temp.p3, temp.p4)
        l4 =    self.get_line_coords(temp.p1, temp.p4)
        l5 =    self.get_line_coords(temp.p5, temp.p6)
        l6 =    self.get_line_coords(temp.p6, temp.p7)
        l7 =    self.get_line_coords(temp.p7, temp.p8)
        l8 =    self.get_line_coords(temp.p8, temp.p5)
        l9 =    self.get_line_coords(temp.p1, temp.p5)
        l10 =   self.get_line_coords(temp.p2, temp.p6)
        l11 =   self.get_line_coords(temp.p3, temp.p7)
        l12 =   self.get_line_coords(temp.p4, temp.p8)

        L = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]

        return L
    
    
    def draw_3Dbbox(self, im, path, df_true, df_pred, frame):
        '''
        Draw obs and pred 3D bbox on 2D image
        '''
        # Create figure and axes
        fig = plt.figure(figsize=self.figsize, dpi=100)

        # Add line
        L_true = self.get_lines(df_true)
        L_pred = self.get_lines(df_pred)

        for i in range(len(L_true)):
            plt.plot(L_true[i][0], L_true[i][1], color='g', linewidth=1.5)
            plt.plot(L_pred[i][0], L_pred[i][1], color='r', linewidth=1.5)
        
        name = join(path, str(frame) + '_temp.jpg') 
        plt.savefig(name, bbox_inches='tight')
        plt.imshow(im)
        plt.show()
        #plt.close(fig)

    
    def get_gif(self, idx, true, pred, pid, gif_name, args, outpath):
        '''
        Create GIF
        '''
        self.size = args.input
        self.outpath = outpath
        
        # Fine relevant obs data
        df_true = true.loc[true.ID == pid]
       
        # Get image paths
        inpath = df_true.scenefolderpath.iloc[idx+1][0]
        
        # Get frame numbers (which is stored in the next line)
        # Current line stores frame numbers of obs, not target
        filenames = df_true.filename.iloc[idx+1]
        filenames = [x.lstrip("0") for x in filenames]
        frames = [int(os.path.splitext(x)[0]) for x in filenames]
        
        # Get true observations (which is stored in the next line)
        df_obs = self.cam2pix_bbox(pd.DataFrame(df_true.bounding_box.iloc[idx+1], columns=['x','y','z','w','h','d']))
        
        # Get predictions
        orig_idx = df_true.index[0]
        df_pred = self.cam2pix_bbox(pd.DataFrame(pred[orig_idx], columns=['x','y','z','w','h','d']))
        
        for i in range(self.size):
            im = Image.open(join(inpath, filenames[i])) #frame number starts at 1
            obs = df_obs.iloc[i]
            preds = df_pred.iloc[i]
            self.draw_3Dbbox(im, self.outpath, obs, preds, frames[i])

        with imageio.get_writer(join(self.outpath, gif_name), mode='I') as writer:
            for k in frames:
                name = str(k) + '_temp.jpg'
                image = imageio.imread(join(self.outpath, name))
                writer.append_data(image)
                os.remove(join(self.outpath, name))
    
    
    def get_3d_vertices(self, df):
        temp = pd.DataFrame()

        #switch order of y and z
        temp['x1'] = df.x - df.w/2
        temp['z1'] = df.y - df.h/2 
        temp['y1'] = df.z - df.d/2
        temp['x2'] = df.x + df.w/2
        temp['z2'] = df.y + df.h/2
        temp['y2'] = df.z + df.d/2

        return temp
    
    def get_axlim(self, ax_min, ax_max, row):
        ax_max[0] = max(row['x2'], ax_max[0])
        ax_max[1] = max(row['y2'], ax_max[1])
        ax_max[2] = max(row['z2'], ax_max[2])
        ax_min[0] = min(row['x1'], ax_min[0])
        ax_min[1] = min(row['y1'], ax_min[1])
        ax_min[2] = min(row['z1'], ax_min[2])

        return ax_min, ax_max


    def set_plot_axes(self, ax, ax_min, ax_max):
        ax.set_xlim(xmin=ax_min[0], xmax=ax_max[0])
        ax.set_ylim(ymin=ax_min[1], ymax=ax_max[1])
        ax.set_zlim(zmin=ax_min[2], zmax=ax_max[2])

        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(AutoLocator())
        ax.zaxis.set_major_locator(AutoLocator())

        ax.set_aspect('auto')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


    def plot_3d(self, df, num_peds, color, view='default'):
        bbox = self.get_3d_vertices(df)

        # draw figure
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(projection='3d')

        if view == 'top':
            ax.view_init(elev=90, azim=180) #birds eye view

        ax_min = np.ones(3)*np.inf
        ax_max = -np.ones(3)*np.inf
        for i in range(num_peds):
            Z = []
            for point in self.coords.keys():
                p = self.coords[point]
                Z.append([bbox[p[0]][i], bbox[p[1]][i], bbox[p[2]][i]])

            Z = np.asarray(Z)
            verts = [[Z[0],Z[1],Z[2],Z[3]], [Z[4],Z[5],Z[6],Z[7]], [Z[0],Z[1],Z[5],Z[4]],
                     [Z[2],Z[3],Z[7],Z[6]], [Z[1],Z[2],Z[6],Z[5]], [Z[4],Z[7],Z[3],Z[0]]]

            ax.add_collection3d(Poly3DCollection(verts, facecolors=color[i], linewidths=1, edgecolors=color[i], alpha=.20))

            ax_min, ax_max = self.get_axlim(ax_min, ax_max, bbox.iloc[i])

        # set plot ax params
        self.set_plot_axes(ax, ax_min, ax_max)
        plt.show()
                