# Neural Network Models for GAIT Analysis and Prediction
## 1. GAIT Analysis and Prediction
Most of these codes were used for my Master Thesis at NTUA - Mechanical Engineering. Please read the corresponding parts (chapter 3, 4) of my thesis to understand the use of the Phase Variable and all the preprocessing of the Data.

[Thesis - Use of Neural Networks in the Analysis and Prediction of Human GAIT](USE THESIS URL!!!!)

All the data used in this thesis were a contribution from the Hellenic Society for the Protection and Rehabilitation of Disabled People 
    
[ELEPAP](https://elepap.gr/) 

## 2. The Folders of this Project contain

* Data_CP: Lower-limb kinematic and kinetic data of childred with cerebral palsy and other other neurological diseases

* Data_Normal: Lower-limb kinematic and kinetic data of typically developed children.

* Neural_Network_Codes: Neural Networks (LSTM, CNN) using the phase variable or timestamps approach.

* Predictions: Saved predictions in excel files. All Neural Network models save the predictions in the corresponding excel file.

* Saved_Models: Saved Keras models for each neural network to be used by ROS2 Control node in another [project](https://github.com/Evangelos-Karavas/Exo-suit-control).

* Scaler: Saved Scaler for phase variable and timestamps approach to use later with ROS2 Control node.


## 3. ROS2 Control
All the Neural Network Models created in this project were used for the joint angle control of a lower-limb exoskeleton. It's a not finished simulation for an exoskeleton to be used on children with cerebral palsy.

For more information look at the project [ROS2-Lower-Limb-Exoskeleton-Control ](https://github.com/Evangelos-Karavas/ROS2-Lower-Limb-Exoskeleton-Control)
