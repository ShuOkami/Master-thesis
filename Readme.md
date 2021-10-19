# README

## Depedencies

The present implementation makes use of many librabry and depedencies, find here the list of them, alongside some explanations on the installation:

- Stormpy
- Gurobi

### Stormpy

Stormpy is used in order to both build our model and perform value iteration. <br />
Find here the documentation of stormpy https://moves-rwth.github.io/stormpy/

### Gurobi

Guroby is used on python, using Gurobipy. A simple "pip install gurobipy" is enough to install that library. <br />
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
where myTaskSystem is a text file stored in LPFile.
