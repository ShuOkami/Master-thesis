import stormpy
import doctest
import time
import sys
import taskmodel
import gurobipy as gp
from gurobipy import GRB

#https://docs.google.com/spreadsheets/d/1dL2a4TVPY6VVq9mJx9eabCzVpXS5uRD_s2UollUrsO4/edit#gid=0



def cleanArg(arg):
	#Clean the argument given in command and generate file name for further function calls

	file_name = arg.strip()
	out_file = file_name.split(".")[0]
	return out_file



def prepareLP(model, M):
	#Build dictionary that will later contains the Linear-Program's strings to write. 
	#M[1] is the function to optimize (Minimize/Maximize)
	#M[2] is the list of all constraints
	#M[3] is the final constraint for statistical equilibrium
	#Each MDP's state represents a constraint

	for i in range(model.nr_states):

		M[2].append("r_" + str(i+1) + ": ")

	return M



def endSol(M):
	#Brings closure to the dictionary before writing the Linear Program

    for i in range(len(M[2])):

        M[2][i] += " = 0\n" #All constraint end by " = 0"

    M[3] += "= 1" #All but last, ending by " = 1"

    return M



def writeLP(M, fileName):
	#Writes the Linear Program according to the dictionary

    m = open(fileName, "w")
    m.write("Minimize\n")
    m.write(M[1]) 				#Objective function
    m.write("\nSubject To\n")
    m.writelines(M[2])			#Constraint
    m.write(M[3]) 				#Statistical equilibrium
    m.write("\nEnd")
    m.close()



def exploreModel(model):
	#Explores the built model in order to generate the objective function alsongside the constraints of the Linear Program
	#Result will end in M

	M = {1 : "", 2 : [], 3 : "r_00: "}
	M = prepareLP(model, M)

	costs = model.reward_models[""].state_action_rewards
	#model.reward_models is a dict of the form : {"model_name" : SparseRewardModel}
	#costs is a list of size [number of unique actions] and represents for each of these unique actions their corresponding cost

	actionId = 1 #Keep track of the current unique action

	for state in model.states: #Run through all the states

		for action in state.actions: #For all action from each state

			if costs[actionId-1] > 0: #If cost is not zero, add action to the objective function
				M[1] += "+ " + str(costs[actionId-1]) + " x" + str(actionId) + " "

			M[2][int(state)] += "+ x" + str(actionId) + " " #For each state we add all outgoing edge in the corresponding constraint
			M[3] += "+ x" + str(actionId) + " " #Add the unique action to the last constraint
			
			for transition in action.transitions: #For all possible outcome from picking the action
				M[2][transition.column] += "- " + str(transition.value()) + " x" + str(actionId) + " " #Add incoming edge to the corresponding constraint

			actionId += 1

	return M



def optimize(fileName):
	#Uses Gurobi to optimize the Linear program built previously
	#And write the solution to LPSol/model.sol

	model = gp.read("LPFile/" + fileName + ".lp")
	model.optimize()

	if model.status == GRB.OPTIMAL: #If model is feasible
		model.write("LPSol/" + fileName + ".sol")



def parseSystem(fileName):
	#Parse the task.txt file to retreive the list of soft tasks

	print("Parsing Tasks system ... ", end="")
	file_name = "TasksSet/" + fileName + ".txt"
	hard_tasks, soft_tasks = taskmodel.get_tasks(file_name) 
	print("Done")
	return hard_tasks, soft_tasks #We only look for soft tasks



def buildMDPs(soft_tasks):
	#Build all individual MDP for each of the soft tasks

	print("Generating individual MDPs ... ", end="")
	task_graphs_list = taskmodel.construct_graphs_soft_tasks(soft_tasks)
	print("Done")
	return task_graphs_list



def generatePrismFile(fileName, task_graphs_list, hard_tasks, soft_tasks):
	#Generate the prim file from the list of MDPs

	print("Writing Prism file ... ", end = "")
	taskmodel.generate_prism_file(task_graphs_list, "PrismFile/" + fileName + '.pm', hard_tasks, soft_tasks)  # .pm extension for mdp
	print("Done")



def buildFinalMDP(fileName):
	#Read the prism file and build the product of all individual MDPs
	#The product is computed by Storm's tool

	print("Building product of MDPs... ", end="")
	doctest.ELLIPSIS_MARKER = '-etc-'
	program = stormpy.parse_prism_program("PrismFile/" + fileName + ".pm")
	prop = "Rmin=?[LRA]"
	properties = stormpy.parse_properties_for_prism_program(prop, program, None)
	model = stormpy.build_model(program, properties)
	print("Done\nNumber of states: {}".format(model.nr_states))
	print("Number of transitions: {}".format(model.nr_transitions))
	return model



def generateLP(model, fileName):
	#Generate LP File

	print("Generating LP... ", end="", flush=True)
	M = exploreModel(model)
	M = endSol(M)
	writeLP(M, "LPFile/" + fileName + ".lp")
	print("Done")



def modelCheck(model, fileName):

	print("Performing model checking... ", end="")
	program = stormpy.parse_prism_program("PrismFile/" + fileName + ".pm")
	prop = "R=? [F \"done\"]"

	properties = stormpy.parse_properties(prop, program, None)
	model = stormpy.build_model(program, properties)
	initial_state = model.initial_states[0]
	result = stormpy.model_checking(model, properties[0])
	print("Result: {}".format(round(result.at(initial_state), 6)))
	print("Done\nResult: {}".format(round(result.at(initial_state), 6)))



def main():

	fileName = cleanArg(sys.argv[1])

	hard_tasks, soft_tasks = parseSystem(fileName) 
	task_graphs_list = buildMDPs(soft_tasks)
	generatePrismFile(fileName, task_graphs_list, hard_tasks, soft_tasks)
	model = buildFinalMDP(fileName)

	start = time.time()

	generateLP(model, fileName)

	end = time.time()
	print("Generation computation time : ", round(end-start, 3), " seconds")


	optimize(fileName) #No Gurobi license on my VM

	start = time.time()

	#modelCheck(model, fileName) #Use Storm's model checker

	end = time.time()
	print("Storm checker computation time : ", round(end-start, 3), " seconds")
	



if __name__== "__main__":
	main()