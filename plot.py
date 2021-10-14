import matplotlib.pyplot as plt
import numpy as np


def main():


	size = 		[19,191,192,231,286,322,476,1283,1347,2343,3690,3790,6446,11893,12297,14738,17787,32866,40900,58292,94428,96264,146159,385429,752792]
	stormtime = [0.001,0.007,0.002,0.002,0.015,0.052,0.007,0.601,0.074,0.035,0.211,0.198,0.351,2.206,324.11,0.823,3.991,5.946,9713,174.4,15.6,830.9,106,15274,31755]
	lptime = 	[0.006,0.029,0.03,0.03,0.032,0.035,0.036,0.163,0.198,0.637,0.423,0.426,0.726,1.926,1.83,1.858,3.272,7.296,8.546,20.592,74.4,101.898,172,1412,5759]
	lpsolvtime = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.06,0.1,0.46,0.14,0.16,0.26,0.64,0.46,0.61,1.14,2.43,1.72,6.41,33.89,17.34,72.60,186.38,604.34]

	size12 = [19,191,231,286,322,476]
	storm12 = [0.001,0.007,0.002,0.015,0.052,0.007]
	lp12 = [0.006,0.029,0.03,0.032,0.035,0.036]

	size3 = [192,1347,2343,3790,6446,14738]
	storm3 = [0.002,0.074,0.035,0.198,0.351,0.823]
	lp3 = [0.03,0.198,0.637,0.426,0.726,1.858]

	size45 = [1283,3690,11893,12297,17787,32866,40900,58292,94428,96264,146159]
	storm45 = [0.601,0.211,2.206,324.114,3.991,5.946,9713.673,174.415,15.6,830.973,106.25]
	lp45 = [0.163,0.423,1.926,1.83,3.272,7.296,8.546,20.592,74.4,101.898,172]

	sizearr = np.array(size)
	stormarr = np.array(stormtime)
	lparr = np.array(lptime)
	solvarr = np.array(lpsolvtime)

	sizear12 =  np.array(size12)
	sar12 = np.array(storm12)
	lpar12 = np.array(lp12)

	sizear3 =  np.array(size3)
	sar3 = np.array(storm3)
	lpar3 = np.array(lp3)

	sizear45 =  np.array(size45)
	sar45 = np.array(storm45)
	lpar45 = np.array(lp45)

	plt.yscale("log")
	plt.xscale("log")

	plt.plot(sizearr, stormarr, 'bs', sizearr, solvarr, 'g^')
	plt.plot(sizearr, stormarr, label = "Storm's value iteration")
	plt.plot(sizearr, solvarr, label = "Linear programming (solving time only)")

	#plt.plot(sizear12, sar12, 'bs', sizear12, lpar12, 'g^')
	#plt.plot(sizear12, sar12, label = "Storm's value iteration")
	#plt.plot(sizear12, lpar12, label = "Linear programming")


	#plt.plot(sizear3, sar3, 'bs', sizear3, lpar3, 'g^')
	#plt.plot(sizear3, sar3, label = "Storm's value iteration")
	#plt.plot(sizear3, lpar3, label = "Linear programming")

	#plt.plot(sizear45, sar45, 'bs', sizear45, lpar45, 'g^')
	#plt.plot(sizear45, sar45, label = "Storm's value iteration")
	#plt.plot(sizear45, lpar45, label = "Linear programming")


	plt.title("Computation time for minimum expected mean-cost over MDP's size")
	plt.xlabel('Size of the MDP')
	plt.ylabel('Computation time (s)')
	plt.legend()

	plt.show()


if __name__== "__main__":
	main()