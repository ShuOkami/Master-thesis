# README

## Michael's master's thesis

This github repository includes all the implementation used for the purpose of my master's thesis. <br />
It builds, from a task system, the corresponding MDP and linear program. <br />
Then, we can either perform value iteration from the MDP or solve the linear program using Gurobi's solver in order to find the safe and optimal strategy of the scheduler.

## Dependencies

The present implementation makes use of many librabry and depedencies, find here the list of them, alongside some explanations on the installation:

- Stormpy
- Gurobi

### Stormpy

Stormpy is used in order to both build our model and perform value iteration. <br />
Find here the documentation of stormpy https://moves-rwth.github.io/stormpy/ , giving an installation guide. 

### Gurobi

Gurobi is used on python, using Gurobipy. A simple "pip install gurobipy" or anything equivalent is enough to install that library. <br />
However, in order to run Gurobi's solver on large systems, you need a Gurobi's license. <br />
For that matter, go to https://www.gurobi.com/

## Run the code

In this section, find how to run the code. 

### Taskmodel.py

You do not need to run that python file. It only includes functions used in other files.

### generateLP.py

This file takes as input a task system and from it, build the corresponding MDP and then the linear program. You can also, from this file, perfom value iteration on the MDP or solve the linear program with Gurobi. <br />
But in our case, we seperate these functions in different files. <br />
<br />
The command to run generateLP.py is the following : <br />
```
python3 generateLP.py myTaskSystem
```
where myTaskSystem is a text file following a specific format. <br />
myTaskSystem must be stored in the TasksSet folder. <br />
Once you have ran generateLP.py, a linear program will be writen in the LPFile folder. 

### optimize.py

This file will solve your linear programs. <br />
You can run optimize.py with the following command:
```
python3 optimize.py myLpFile
```
where myLpFile is a linear program written in the LPFile folder, following the .lp format <br />
Once finished, the program will write the solution of the linear program in the LPSol folder.

### stormcheck.py

This script will build the MDP from a task system and will perform Storm's default value iteration <br />
Here is the command in order to run it :
```
python3 stormcheck.py myTaskSystem
```
where myTaskSystem is a text file following a specific format. <br />
