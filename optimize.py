import time
import gurobipy as gp
from gurobipy import GRB
import sys

def main():

	start = time.time()

	model = gp.read("LPFile/" + sys.argv[1] + ".lp")
	#model.setParam(GRB.Param.Presolve, 0) #Unable Presolve
	model.optimize()

	if model.status == GRB.OPTIMAL:
		model.write("LPSol/" + sys.argv[1] + ".sol")

	end = time.time()

	print("\nTotal computation time : ", end-start, " seconds")



if __name__== "__main__":
	main()