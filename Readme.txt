Description of each folder

1. Development
This folder contains all files for the development of the system. It has 5 subfolders.
  - 3D object data
     Contains all stl files needed to perform simulation
  - Conda_Package
    all python libraries needed to run the python script
  - Network
    contains training data, test data, and script for training and test the network
  - Simulation
    contains script for performing simulation and generate synthetic data. The generated
    synthetic data is saved inside the Image folder. The main file for creating synthetic data is 'synthetic_data_generation.py'.
    The rest of them are modules.
All python script modules contain documentation of each function. Please take a look at the particular python file 
to look more in detail what each function does.
	
2. Execution
This folder contains all files required for performing experiments. The controller software is developed by our lab and the command sequence for the controller software is 'CommandSequence.txt'.
All python script modules contain documentation of each function. Please take a look at the particular python file 
to look more in detail what each function does.

The rest of the folders are self-explanatory
