import sys
from fractions import Fraction
from decimal import Decimal
from copy import deepcopy

maxOffset = 0

class Task:

	def __init__(self, ID, offset, exe, deadline, period, cost, done=False):
		
		self.ID = ID
		self.offset = offset
		self.exe = exe
		self.deadline = deadline
		self.period = period
		self.cost = cost
		self.done = done


	def __repr__(self):

		return 'Task %s : offset = %s, execution time = %s, deadline = %s, period = %s, cost = %s, done = %s' % (self.ID, self.offset, self.exe, self.deadline, self.period, self.cost, self.done)


	def __eq__(self, other):

		return (self.ID == other.ID and self.exe == other.exe and self.deadline == other.deadline and self.period == other.period and self.cost == other.cost)


	def decrease(self, chosen):

		newExe = deepcopy(self.exe)
		if chosen :
			for i in range(len(self.exe)):
				self.exe[i][0] -= 1
				if self.exe[i][0] == 0 and len(self.exe) == 1: self.done = True

		self.deadline -= 1

		for i in range(len(self.period)):
				self.period[i][0] -= 1


	def isPossiblyDone(self):

		for poss in self.exe:
			if poss[0] == 0:
				return True
		return False



class Node:

	def __init__(self, tasks, step):

		self.nTasks = tasks
		self.actions = []
		self.step = step
		self.generateActions()
		self.link = []

	def __eq__(self, other):

		global maxOffset
		return (self.actions == other.actions and self.nTasks == other.nTasks and self.step > maxOffset and other.step > maxOffset)


	def getTask(self, ID):

		for task in self.nTasks:
			if task.ID == ID:
				return task
		return None


	def generateActions(self):

		actions = []
		for task in self.nTasks:
			if (task.deadline != 0 and not task.done):
				actions.append(task)
		#self.actions.append("idle")

		self.actions = actions


	def schedule(self, scheduledTask):

		possiblyDone = False

		#print("About to be scheduled : %s" % (scheduledTask.ID))
		#self.printTasks()

		for task in self.nTasks:
			if task == scheduledTask:
				task.decrease(True)
				possiblyDone = task.isPossiblyDone()
			else:
				task.decrease(False)

		#print("Completed")
		#self.printTasks()

		return possiblyDone


	def splitOnExe(self, taskID):

		task = self.getTask(taskID)
		lTask = None
		rTask = None
		rExe = []
		self.nTasks.remove(task)
		for exe in task.exe:
			if exe[0] == 0:
				lTask = Task(task.ID, task.offset, [exe], task.deadline, task.period, task.cost, True)
			else:
				rExe.append(exe)
		rTask = Task(task.ID, task.offset, rExe, task.deadline, task.period, task.cost)

		lTasks = deepcopy(self.nTasks)
		rTasks = deepcopy(self.nTasks)
		lTasks.append(lTask)
		rTasks.append(rTask)

		return [Node(lTasks, self.step), Node(rTasks, self.step)]



	def printTasks(self):

		for task in self.nTasks:
			print(task)



class Graph:

	def __init__(self, tasks):

		global maxOffset
		self.tasks = tasks
		self.globalTasks = deepcopy(tasks)
		self.nodes = []
		self.offsets = []
		self.start = None
		self.getOffsets()
		self.constructGraph()
		maxOffset = max(self.offsets)


	def getOffsets(self):

		for i in range(len(self.tasks)):
			self.offsets.append(self.tasks[i].offset)


	def constructGraph(self):

		self.start = self.makeStartState()
		self.nodes.append(self.start)
		self.start.printTasks()

		queue = [self.start]
		nodeNumber = 0
		while len(queue) > 0:
			(head, queue) = popQueue(queue)
			nextStates = self.getNext(head)
			print("\n\nNEW QUEUE ITERATION \n\n")
			for node in nextStates:
				print("\nNode : ", node, " From ", head)
				node.printTasks()
			queue.extend(nextStates)
			nodeNumber += len(nextStates)
		print("NODE NUMBER IS : ", nodeNumber)


	def getNext(self, node):

		newTasks = []
		newNodes = []
		print("\n\nComputing ", node)
		for task in self.tasks:
			if (task.offset - node.step - 1 == 0):
				newTasks.append(task)

		for task in node.actions:
			copyNode = deepcopy(node)

			if copyNode.schedule(task) and len(task.exe) >= 2:						#True if the task is possibly completed
				splitedNodes = copyNode.splitOnExe(task.ID)

				if newTasks:
					for newNode in splitedNodes:
						newNode.nTasks.extend(newTasks)

				newNodes.extend(splitedNodes)

			else:
				copyNode = deepcopy(node)
				copyNode.schedule(task)
				if newTasks:
					copyNode.nTasks.extend(newTasks)

				newNodes.append(copyNode)

		print("newNode : " + str(len(newNodes)))
		for node in newNodes:
			node.step += 1
			node.generateActions()
			print(node)

		return newNodes


	def isPossiblyDone(self, task):

		for poss in task.exe:
			if poss[0] == 0:
				return poss
		return False


	def exists(self, baseNode, newNode):

		for node in self.nodes:
			if newNode == node:
				baseNode.link.append(node)
				return True
		return False


	def makeStartState(self):

		startTasks = []
		for task in self.tasks:
			if (task.offset == 0):
				startTasks.append(task)
		return Node(startTasks, 0)


def popQueue(queue):

    return (queue[0], queue[1:])



def get_tasks(file_name):
	f=open(file_name, "r")

	#readlines reads the individual line into a list
	fl =f.readlines()
	lines=[]
	tasks= []
	hard_tasks= []
	soft_tasks= []
	has_soft_tasks = False

	f.close()

	# for each line, extract the task parameters
	ID = 0
	for x in fl:
		y = x.strip()

		if y[0] == '-':
			hard_tasks = tasks
			tasks = []
			has_soft_tasks = True

		elif y!='' and y[0]!='#':
			lines.append(y)
			task = y.split("|")  # task should have 4 elements

			arrival = int(task[0])

			dist = task[1].split(";")  # distribution on the execution time
			exe = []
			max_exe_time = 0
			for z in dist:
				z = z.strip("[")
				z = z.strip("]")
				z = z.split(",")
				time = int(z[0])

				if time > max_exe_time:  # compute maximum execution time
					max_exe_time = time

				# change to fraction since float arithmetic produces bad precision
				#prob = float(z[1])
				#exe.append((time, prob))
				exe.append([time, Fraction(Decimal(z[1]))])

			deadline = int(task[2])

			dist = task[3].split(";")  # distribution on the period
			period = []
			arrive_time = []
			for z in dist:
				z = z.strip("[")
				z = z.strip("]")
				z = z.split(",")
				time = int(z[0])

				# change to fraction since float arithmetic produces bad precision
				#prob = float(z[1])
				#period.append((time, prob))
				period.append([time, Fraction(Decimal(z[1]))])

				arrive_time.append(time)

			min_arrive_time = min(arrive_time)

			if has_soft_tasks:
				cost = Fraction(Decimal(task[4]))
				tasks.append(Task(ID, arrival, exe, deadline, period, cost))
			else:
				tasks.append([arrival, exe, deadline, period, max_exe_time, min_arrive_time])
			ID += 1

	if has_soft_tasks:
		soft_tasks = tasks
	else:
		hard_tasks = tasks

	return hard_tasks, soft_tasks



def main():

	file_name = sys.argv[1]
	file_name = file_name.strip()
	hard_tasks, soft_tasks = get_tasks(file_name)  # hard_tasks and soft_tasks are a list of task descriptions: [arrival, exe dist, deadline, period dist, max_exe_time, min_arrive_time] elements
	print(soft_tasks)
	G = Graph(soft_tasks)

if __name__== "__main__":
	main()
