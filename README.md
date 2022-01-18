# 3d-bounding-box-prediction

This was a semester project at VITA Lab. 

## Abstract
Predicting the future location of a pedestrian is key to safe-decision making for autonomous vehicles. 
It is a non-trivial task for self-driving cars because humans can choose complex paths and move at non-uniform speeds. 
Furthermore, the self-driving car should predict a pedestrian's location and intention sufficiently far into the future to have time to react accordingly. 
This can be done by predicting a sequence of intentions for a fixed time horizon. 
The proposed network is a multi-task sequence to sequence LSTM model called the Position-Velocity LSTM (PV-LSTM), as it encodes both the position and the velocity of the 3D pedestrian bounding box
The input to the network is the observed speed and position of the 3D bounding box center. 
The output is the predicted speed of the 3D bounding box center, which can then be converted into its corresponding position. The network consists of LSTM encoder-decoders for position and speed followed by a fully connected layer.

This project uses the [Joint Track Auto (JTA)](https://github.com/fabbrimatteo/JTA-Dataset) and the [NuScenes](https://www.nuscenes.org/) datasets.

## Repository structure
```
|---- 3d-bounding-box-prediction          : Project repository
      |---- exploration                   : Jupyter notebooks for data exploration and visualization
            |---- JTA_exploration.ipynb   
            |---- NuScenes_exploration.ipynb
      |---- preprocess                    : Scripts for preprocessing
            |---- jta_preprocessor.py
            |---- nu_preprocessor.py
            |---- split.py
      |---- utils                         : Scripts containing necessary calculations
            |---- utils.py  
            |---- nuscenes.py
      |---- visualization                 : Scripts for visualizing the results and making GIFs
            |---- visualize.py
      |---- Dataloader.py                 : Script for loading preprocessed data
      |---- network.py                    : Script containing network 
      |---- network_pos_decoder.py        : Script containing network variation that has a position decoder (not used)
      |---- test.py                       : Script for testing
      |---- train.py                      : Script for training 
```

## Proposed network
![](/images/network_diag.png)

## Results
![JTA](/images/test_seq_478_frame177_idx17500.gif)
![NuScenes](/images/test_scene-0593_frame0_idx35.gif)


## Setup
Please install the required dependencies from the <requirements.txt> file.
For Nuscenes, scripts in the folder <nuscenes-devkit/python-sdk/nuscenes> are required.

## Jupyter notebooks
The Jupyter notebooks provided demonstrate how all the code in this repository can be used.

## Data loading notes
The input, output, stride, and skip parameters of the loaded dataset can be set the in the '''args''' class.
To load the datasets, first run the preprocessing scripts, then ```Dataloader.py```.
**Note** Due to the large number of samples of the JTA dataset, the preprocessing script first saves files containing all available samples. This data can then be read by the ```Dataloader.py``` file to get sequences of bounding boxes that are passed to the network.
