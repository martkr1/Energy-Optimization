# Energy Optimization using K-Spice and Reinforcement learning

## DQN (Deep Q-network) - Defining our ML model
Currently containing a simple definition of a MLP network. The input nodes represent the state of the simulator, i.e. the input vals, ctrl vals and global KPIs provdied in the excel sheet an fed to the network as a flattened tensor. 
The output nodes represents each possible action, i.e. if we have two controllers both only able to do one of the following actions: up, down, stay, the output layer would containt all combinations of these actions for two controllers (up,down,stay)^2 -> 9 nodes. 
The network will output the "probability" of choosing an action and we'll always choose the action corresponding to the node in the output layer with the highest value.

## enviroment - Communicating with the K-spice process simulator
Contains the main functionality for communicating with the K-Spice library. 
 - env_workbook is a jupyter notebook illustarting the functionality of the enviroment class.

Main functionality:
- [x] import_variables(): func to import the K-spice variables we want to use/adjust/evaluate
- [x] state: property - state of simulator, describes through the imported variables
- [x] reward(): was the action good? -currently just an binary abs diff func 
- [x] step(): use a polling func to update variables
- [x] reset(): re-load initial conditions to reset simulator
- [ ] Implement slugging as randomness in env.reset
- [ ] Terminated: functionality such that simulator resets if trip is reached. Trip limit needs to be provided to the simulator (via the excel sheet?)

## workbook - Jupyter notebook as final interface
Combining DQN and enviroment

- new_wb.ipynb - one or two ctrl learning
  
- [x] Select action - function for choosing next action
- [x] Memory
- [x] Optimizer - First draft complete
- [x] Trainer - First draft complete
- [x] Upscale to multible controllers 
