# Final_Project

#### Folders:
1. "Data" Folder contains the raw data for enzyme activities and flux from Lo-Thong et al, in a .csv format.
2. "Converted Data" Folder contains the data in the form of numpy arrays after preliminary data treatment. The data is converted from the .csv file saved here using data_conversion.py.
3. "NN_Data" Folder contains all the saved numpy arrays and pickle files that were generated using create_NN_data.py. These files are used directly by main_direct.py and main_modified.py.
#### Code Files:
1. "main_direct.py" is a direct implementation of the Keras tutorial for MPNN(Message Passing Neural Networks) created by Alexander Kensert(2021) which can be found here:https://keras.io/examples/graph/mpnn-molecular-graphs/. In case the link is not functioning, a forked version of the code can be found here: https://github.com/Jiazhen-Chen/keras-io/blob/master/examples/graph/mpnn-molecular-graphs.py. The direct implementation is training the tutorial MPNN directly with enzyme and flux data provided in literature with minimal changes.
2. "main_modified.py" is a self-designed MPNN based on the example provided, with a modified readout/prediction layer and removal of all reduntant steps which is intended for treating non-uniform graph structures. This model is created using a reverse-engineering approach, taking apart the tutorial code and making changes as needed.<br /><br />By design, the model should take one pathway graph as input at a time and predict the flux. The weights/biases will be updated after each input. **While the code runs, the correctness of the code cannot be guaranteed.** Ensuring the correct implementation of the design through code is beyond my current capabilities. 
#### Figure Files:
1. "modeldirect.png" and "modelmodified.png" are schematic diagrams regarding how the neural network functions. They are generated from the "main_direct.py" and "main_modified.py" codes and renamed.
2. "Figure-DirectMPNN.png" and "Figure-ModifiedMPNN.png" are representitive of a typical training process. The training process is stochastic since there is a degree of randomness to the optimizer, but the end result after the total number of epochs are similar. The figures are generated by the "main_direct.py" and "main_modified.py" codes and renamed.
#### Final Remarks:
After running the "main_direct.py" and "main_modified.py" programs, the model can be used to predict fluxes using the command "mpnn.predict(dataset_name)" where dataset_name is the input data variable name. For checking the training/validation predictions, replace "dataset_name" with "train_data" and/or "val_data". The values however, are all zeros for both neural networks.
<html>
  <p align=center>
   <img src="https://i.imgflip.com/4aiftp.jpg" height="200">
  </p>
</html>
