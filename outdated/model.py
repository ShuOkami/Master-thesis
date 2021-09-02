import sys
sys.path.insert(1, '../')
import task
import os
import networkx as nx
import pygraphviz
from copy import deepcopy
import time
import gurobipy as gp
from gurobipy import GRB
import lpsolve55
from lp_solve import *

def mappingLP(G, soft_tasks):

	actMatrix = [[i,j] for i in range(1,G.number_of_nodes()+1) for j in range(-1,len(soft_tasks))]
	mapping = {}
	num = 1
	for elem in actMatrix:
		key = "x"
		key += (str(num))
		num += 1
		telem = tuple(elem)
		mapping[telem] = key

	return mapping


def Tmap(G, soft_tasks):

	actMatrix = [[i,j] for i in range(1,G.number_of_nodes()+1) for j in range(-1,len(soft_tasks))]
	mapping = {}
	num = 1
	for elem in actMatrix:
		key = "x"
		key += (str(num))
		num += 1
		telem = tuple(elem)
		mapping[key] = telem

	return mapping



def getCostMatrix(G, costs):

	nodes = list(G.nodes(data=True))
	matrix = [[None for i in range(len(nodes))] for j in range(len(nodes))]

	for node in nodes:
		pred = G.predecessors(node[0])
		iterObj = iter(pred)
		flags = node[1]["config"][-1]
		cost = 0
		it = 0
		for flag in flags:
			if flag : cost += costs[it]
			it += 1
		while True:
			try:
				item = next(iterObj)
				data = G.get_edge_data(item-1,node[0]-1)
				if data:
					matrix[item-1][node[0]-1] = cost
			except StopIteration:
				break

	return matrix


def costToEdge(G, costs):

	nodes = list(G.nodes(data=True))
	for node in nodes:
		flags = node[1]["config"][-1]
		cost = 0
		it = 0
		for flag in flags:
			if flag : cost += costs[it]
			it += 1
		for edge in G.out_edges(node[0]):
			newTuple = (G[edge[0]][edge[1]]["label"][0], G[edge[0]][edge[1]]["label"][1], cost)
			G[edge[0]][edge[1]]["label"] = newTuple
			#print(G[edge[0]][edge[1]]["label"])




def cleanList(liste):
	new_list = [] 
	for i in liste : 
	    if i not in new_list: 
	        new_list.append(i) 
	return new_list


def getModel(G, mapping, soft_tasks):

	minimize = []
	prevAction = None
	for i in range(G.number_of_nodes()):
		for k in G.out_edges(i+1):
			#cost = costMatrix[i][k[1]-1]
			data = G.get_edge_data(i+1,k[1])
			action = data["label"][0]
			cost = data["label"][2]
			if cost and action != prevAction:
				minimize.append([cost, (k[0], action)])
				prevAction = action

	minToString1 = ""
	minToString2 = ""
	for elem in minimize:
		minToString1 += str(elem[0])
		minToString2 += str(elem[0]) + " "
		minToString1 += str(elem[1])
		minToString2 += (str(mapping[elem[1]]))
		if elem != minimize[-1]:
			minToString1 += " + "
			minToString2 += " + "

	equations = []
	eq2 = []
	for i in range(G.number_of_nodes()):
		string = ""
		string2 = ""
		terme = []
		terme2 = []
		for k in G.out_edges(i+1):
			data = G.get_edge_data(i+1,k[1])
			action = data["label"][0]
			terme.append((k[0], action))
			terme2.append(mapping[(k[0], action)])



		for elem in cleanList(terme):
			string += str(elem)
			if elem != terme[-1]:
				string += " + "

		#string += " = "

		for elem in cleanList(terme2):
			string2 += str(elem)
			if elem != terme2[-1]:
				string2 += " + "

		#string2 += " = "

		terme = []
		terme2 = []
		for k in G.in_edges(i+1):
			data = G.get_edge_data(k[0], i+1)
			prob = data["label"][1]
			action = data["label"][0]
			terme.append([float(prob), (k[0], action)])
			terme2.append([float(prob), mapping[(k[0], action)]])

		for elem in terme:
			string += " - "
			string += str(elem[0]) + " "
			string += str(elem[1])
			if elem != terme[-1]:
				pass
		string += " = 0"

		equations.append(string)

		for elem in terme2:
			string2 += " - "
			string2 += str(elem[0]) + " "
			string2 += str(elem[1])
			if elem != terme2[-1]:
				pass
		string2 += " = 0"

		eq2.append(string2)

	constraint = []
	string = ""
	for i in range(G.number_of_nodes()):
		for k in G.out_edges(i+1):
			data = G.get_edge_data(i+1,k[1])
			action = data["label"][0]
			constraint.append([(k[0], action)])

	for elem in constraint:
		string += str(elem)
		if elem != constraint[-1]:
			string += " + "
	string += " = 1"

	string2 = ""
	for elem in constraint:
		string2 += str(mapping[(elem[0][0], elem[0][1])])
		if elem != constraint[-1]:
			string2 += " + "
	string2 += " = 1"

	"""
	string2 = ""
	for i in range(1, G.number_of_nodes() * (len(soft_tasks) + 1) + 1):
		string2 += "x" + str(i)
		if i < G.number_of_nodes() * (len(soft_tasks) + 1):
			string2 += " + "
	string2 += " = 1"
	"""

	return(minToString1, minToString2, equations, eq2, string, string2)


def makeNodeLabel(node):
	config = node[1]["config"]
	newConfig = list(deepcopy(config))
	newConfig = newConfig[:-1]
	label = []
	for tasks in newConfig:
		for task in tasks:
			label.append(task)
			i = 0
			for value in task:
				if isinstance(value, list):
					newlist = []
					for tup in value:
						newlist.append((tup[0], float(tup[1])))
					label[-1][i] = newlist
				i += 1

	return label



def main():

	# file_name = raw_input("Enter file name: ")
	file_name = sys.argv[1]
	file_name = file_name.strip()
	file_name = "../TasksSet/" + file_name
	hard_tasks, soft_tasks = task.get_tasks(file_name)  # hard_tasks and soft_tasks are a list of task descriptions: [arrival, exe dist, deadline, period dist, max_exe_time, min_arrive_time] elements

	start = time.time()

	costs = []
	for stask in soft_tasks:
		costs.append(float(stask[-1]))
	G = task.construct_graph(soft_tasks, False)

	end = time.time()

	print("Computed graph in : ", end-start)

	#print(G.adj)

	""""
	mat = []
	for n in G:
		nodeToN = []
		for n2 in G:
			if G.has_edge(n2,n):
				nodeToN.append(n2)
		mat.append([n, nodeToN])
	"""


	start = time.time()
	#costsMatrix = getCostMatrix(G, costs)

	costToEdge(G, costs)
	mapping = mappingLP(G, soft_tasks)
	Tmapping = Tmap(G, soft_tasks)
	system = getModel(G, mapping, soft_tasks)

	end = time.time()

	print("Computed linear program in : ", end-start)

	minimize = system[0]
	minimize2 = system[1]
	equations = system[2]
	eq2 = system[3]
	constraint = system[4]
	c2 = system[5]

	r = open("model.txt", "w")
	r.write("Minimize : \n")
	r.write(minimize)
	r.write("\n\n")
	r.write("s.t.\n")
	for eq in equations:
		r.write(eq)
		r.write("\n")
	r.write("\n")
	r.write(constraint)

	m = open("model.lp", "w")
	m.write("min : ")
	m.write(minimize2)
	m.write(";\n\n")
	it = 1
	for eq in eq2:
		row = "r_" + str(it)
		it+=1
		m.write(row)
		m.write(": ")
		m.write(eq)
		m.write(";\n")
	m.write(c2)
	m.write(";\n")
	m.close()

	m = open("Gmodel.lp", "w")
	m.write("Minimize\n")
	m.write(minimize2)
	m.write("\nSubject To\n")
	it = 1
	for eq in eq2:
		row = "r_" + str(it)
		it+=1
		m.write(row)
		m.write(": ")
		m.write(eq)
		m.write("\n")
	row = "r_" + str(it)
	m.write(row)
	m.write(": ")
	m.write(c2)
	m.write("\nEnd")
	m.close()




	"""
	nodes = list(G.nodes(data=True))
	for node in nodes:
		print(node)
	"""


	os.system("lp_solve -s -time model.lp > output.txt")
	#os.system("lp_solve -time -presolvel -s model.lp > output2.txt")
	#solve("model.lp")


	model = gp.read("Gmodel.lp")
	print("GUROBI with Presolve\n")
	model.optimize()
	print("GUROBI without Presolve\n")
	model = gp.read("Gmodel.lp")
	model.setParam(GRB.Param.Presolve, 0)
	model.optimize()
	#model.printStats()
	#p = model.presolve()
	#p.printStats()

	if model.status == GRB.OPTIMAL:
		model.write('model.sol')

	model = gp.read("Gmodel.lp")
	model.printStats()
	model.setParam(GRB.Param.Presolve, 2)
	start = time.time()
	p = model.presolve()
	end = time.time()
	p.write("presolvedG.lp")
	p.write("presolvedG.mps")
	print("Presolved in : ", end-start)
	p.printStats()
	p.optimize()





	start = time.time()

	file = open("output.txt")
	result = {}
	for line in file.readlines():
		if line[0] == "x":
			elem = line.strip("").split()
			result[elem[0]] = elem[1]

	variables = []
	for key in result.keys():
		variables.append(key)

	assMat = []
	unchecked = deepcopy(variables)
	for variable in variables:
		if variable in unchecked:
			unchecked.remove(variable)
			curr = Tmapping[variable][0]
			ass = [variable]
			for variable2 in unchecked:
				if curr == Tmapping[variable2][0]:
					ass.append(variable2)
					unchecked.remove(variable2)
			assMat.append(ass)


	for ass in assMat:
		probsum = 0 
		for var in ass:
			probsum += float(result[var])
		for var in ass:
			if float(result[var]) > 0:
				result[var] = float(result[var]) / probsum


	for edge in G.edges:
		action = G.get_edge_data(edge[0], edge[1])["label"][0]
		key = (edge[0], action)
		prob = result[mapping[key]]
		G[edge[0]][edge[1]]["label"] = (action, prob)
		if float(prob) > 0:
			G[edge[0]][edge[1]]["color"] = "red"

	for node in G.nodes(data=True):

		label = makeNodeLabel(node)
		newlabel = []
		for elem in label:
			newlabel.append(elem)
			if elem != label[-1]:
				newlabel.append('\n')

		node[1]["label"] = newlabel


	#print(newG.nodes(data=True))

	A = nx.nx_agraph.to_agraph(G)
	A.write("AGraph.dot")

	os.system("dot -T pdf AGraph.dot -o test.pdf")

	end = time.time()

	#print("Generated graph PDF in : ", end-start)





if __name__== "__main__":
    main()