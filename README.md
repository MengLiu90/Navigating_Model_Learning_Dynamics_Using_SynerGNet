# Navigating_Model_Learning_Dynamics_Using_SynerGNet
This repository records all the models utilized in the paper, including
* Baseline model
* SAGEConv-configured GNN model
* GATv2Conv-configured GNN model
* GINConv-configured GNN model
* TransformerConv-configured GNN model
* GENConv-configured GNN model (SynerGNet)

## Dependencies
1. pytorch 1.10.0
2. torch_geometric 2.0.2
3. numpy 1.19.2
4. sklearn 0.23.2
5. pandas 1.1.3
6. CUDA 11.1

## Trained models
The trained models for different configurations are recorded in ```./Trained_models/```directory.

## To run the models with your data
### Prepare your data:
1. Prepare a .csv file containing synergy instances, following the format exemplified in ```./Example_data/drugcombs_synergy_data.csv```.
2. Transform your graph data into the .h5 format. Refer to the examples of the .h5 files located in ```./Dataset/DrugCombDB/h5py_synergy_data/``` for guidance.
### Code execution
Run ```python Train.py synergy_file_path h5py_dir_path model_name``` 

```synergy_file_path``` represents the file path to the .csv file containing synergy instances.

```h5py_dir_path``` denotes the directory path where the .h5 format graphs are stored.

```model_name``` specifies the model that you want to run. The available options are ```Baseline_model```, ```SAGEConv```, ```GATv2Conv```, ```GINConv```, ```TransformerConv```, and ```GENConv```.
## Remark
GENConv-configured GNN model was ultimately selected as the final model, designated as SynerGNet.
For further details on SynerGNet and instructions on its usage, please refer to https://github.com/MengLiu90/SynerGNet.
## Synergy Datasets used in this paper
Two synergy datasets were utilized in this study:
1. Original synergy dataset
   
   The synergy data from AZ-DREAM challenge (https://www.synapse.org/#!Synapse:syn4231880/wiki/) 
3. Augmented synergy dataset
   
   Augmented synergy data generated from the AZ-DREAM challenge synergy data. The complete augmented data can be accessed through https://osf.io/kd9e7/.
