# 3d-bounding-box-prediction

This was a semester project at VITA Lab. Predicting the future location of a pedestrian is key to safe-decision making for autonomous vehicles. 
It is a non-trivial task for self-driving cars because humans can choose complex paths and move at non-uniform speeds. 
Furthermore, the self-driving car should predict a pedestrian's location and intention sufficiently far into the future to have time to react accordingly. 
This can be done by predicting a sequence of intentions for a fixed time horizon. 
The proposed network is a multi-task sequence to sequence LSTM model called the Position-Velocity LSTM (PV-LSTM), as it encodes both the position and the velocity of the 3D pedestrian bounding box
The input to the network is the observed speed and position of the 3D bounding box center. 
The output is the predicted speed of the 3D bounding box center, which can then be converted into its corresponding position. The network consists of LSTM encoder-decoders for position and speed followed by a fully connected layer.

## repository structure
```

```

