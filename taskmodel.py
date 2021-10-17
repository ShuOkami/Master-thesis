import networkx as nx
import time
from fractions import Fraction
from decimal import Decimal
import sys
import gurobipy as gp
from gurobipy import GRB
import os
from copy import deepcopy

#Author : Guha Shibashis
#Slight modifications : Paquet Michael

edgeNumber = 1
M = {1 : "", 2 : [], 3 : "r_00: "}


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
                exe.append((time, Fraction(Decimal(z[1]))))

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
                period.append((time, Fraction(Decimal(z[1]))))

                arrive_time.append(time)

            min_arrive_time = min(arrive_time)

            if has_soft_tasks:
                cost = Fraction(Decimal(task[4]))
                tasks.append([arrival, exe, deadline, period, max_exe_time, min_arrive_time, cost])
            else:
                tasks.append([arrival, exe, deadline, period, max_exe_time, min_arrive_time])

    if has_soft_tasks:
        soft_tasks = tasks
    else:
        hard_tasks = tasks

    return hard_tasks, soft_tasks


# pop the first element of queue
def pop_queue(queue):

    return (queue[0], queue[1:])

# push n elements to the end of queue
def push_queue(queue, n):

    return queue + range(queue[-1]+1, queue[-1]+1+n)


def dec_dist(dist):
    new_dist = []
    for x in dist:
        new_dist.append((x[0]-1, x[1]))

    return new_dist



def modify_params(rct_dist, curr_deadline, iat_dist):
    dec_rct_dist = dec_dist(rct_dist)
    dec_iat_dist = dec_dist(iat_dist)

    if curr_deadline > 0:  # check if deadline is greater than 0
        deadline = curr_deadline - 1
    else:
        deadline = curr_deadline

    return dec_rct_dist, deadline, dec_iat_dist



def normalize_dist(dist):
    # dist is a list with elements of the form (time, prob)
    prob_dist = [x[1] for x in dist]
    sum_dist = sum(prob_dist)

    dist = [(x[0], 1 / sum_dist * x[1]) for x in dist]  # float is not needed since we use fraction
    #dist = [(x[0], float(1/sum_dist) * x[1]) for x in dist]
    # dist = [(x[0], float(x[1]/sum_dist)) for x in dist] # This expression is correct but gives imprecise floating point numbers
    return dist



def add_possible_states_task_j(rct_dist, iat_dist, init_config_j, deadline, dec_iat_dist, hard):

    possible_states_task_j = []

    if iat_dist[0][0] > 1:
        possible_states_task_j.append((1, [rct_dist, deadline, dec_iat_dist], 0))

    elif iat_dist[0][0] == 1:
        if hard:
            if deadline >= rct_dist[-1][0]:  # deadline >= max rct
                possible_states_task_j.append((iat_dist[0][1], init_config_j, 0))  # initial state
        else:
            badflag = 0
            if rct_dist[0][0] > 0:
                badflag = 1
            possible_states_task_j.append((iat_dist[0][1], init_config_j, badflag))  # initial state

        if len(iat_dist) > 1:
            possible_states_task_j.append((1-iat_dist[0][1], [rct_dist, deadline, dec_iat_dist[1:]], 0))

    return possible_states_task_j



def safe(task):
    # task is an element of the form [rct, deadline, iat], where rct is of the form [(t1, p1), ..., (tn, pn)]
    # check if deadline >= max rct
    max_rct = task[0][-1][0]
    deadline = task[1]
    if max_rct > deadline:
        return False

    else:
        return True



def compute_states(raw_states, hard):
    # raw_states is a list where each element is also a list that corresponds to each task and has elements (prob, [rct, deadline, iat], bad_flag)
    # it returns states where each element is of the form [(prob1, [rct1, deadline1, iat1], bad_array1), (prob2,[rct2, deadline2, iat2], bad_array2), ... (probn,[rctn, deadlinen, iatn], bad_arrayn)),
    # that is the set of future states
    # we compute the probabilities of each state, denoted by prob and bad_array_i in state i is an array with an element
    # for each task; the element is 1 if the task does not meet its deadline, else it is 0

    states = []
    if len(raw_states)==1:
        for x in raw_states[0]:
            if hard:
                if safe(x[1]):
                    states.append((x[0], [x[1]], [0]))
            else:
                states.append((x[0], [x[1]], [x[2]]))

        return states

    else:
        # recursive call
        subsets = compute_states(raw_states[1:], hard)
        for x in raw_states[0]:
            if hard:
                if safe(x[1]):
                    for y in subsets:
                        states.append((x[0]*y[0], [x[1]] + y[1], [0]))

            else:  # this will never be executed since we do not compute the product of the soft tasks together
                for y in subsets:
                    states.append((x[0] * y[0], [x[1]] + y[1], [x[2]] + y[2]))

        return states



def get_states_execute_i(all_tasks, i, tasks, hard):
    # all_tasks is a list of [rct dist, deadline, iat dist] elements, one for each task. It denotes the current state
    # i is the task that will be executed (start index is 0)
    # tasks is a list of the task descriptions [arrival, exe dist, deadline, period dist, max_exe_time, min_arrive_time]

    # This function returns a tuple (i, states), where states is the set of all states obtained after executing task i,
    # and the second member of the tuple denotes that task i has been executed to reach these states

    raw_states = []

    for j in range(len(all_tasks)):
        task = all_tasks[j]
        init_config_j = tasks[j][1:4]  # gives rct, deadline, iat in the current config

        # normalize distributions so that the probabilities sum up to 1
        rct_dist = normalize_dist(task[0])
        iat_dist = normalize_dist(task[2])
        possible_states_task_j = []

        # reduce times by 1 due to one clock tick
        dec_rct_dist, deadline, dec_iat_dist = modify_params(rct_dist, task[1], iat_dist)

        # compute remaining computation time distribution for task j that is scheduled
        if j == i:
            if (rct_dist[0][0] > 1 or len(rct_dist) == 1) and iat_dist[0][0] > 1:
                # only one successor for task j
                # the first parameter is the probability that task j evolves to this state
                # the probability is 1 since there is only one successor
                # the last parameter is a bad flag needed only for soft tasks, it is 1 when a new task comes but the previous one is not finished
                possible_states_task_j.append((1, [dec_rct_dist, deadline, dec_iat_dist], 0))

            elif iat_dist[0][0] == 1 and len(iat_dist) == 1:
                if not hard:
                    # soft task executes and remaining computation time > 1
                    if rct_dist[0][0] > 1:
                        possible_states_task_j.append((1, init_config_j, 1))  # initial state

                    else:  # rct_dist[0][0] == 1
                        possible_states_task_j.append((rct_dist[0][1], init_config_j, 0))

                        if len(rct_dist) > 1:  # task not getting finished with probability 1 - rct_dist[0][1]
                            possible_states_task_j.append(((1 - rct_dist[0][1]), init_config_j, 1))

                else:  # hard task
                    # one successor that is the initial config for task j
                    # the probability is 1 since there is only one successor
                    possible_states_task_j.append((1, init_config_j, 0))  # initial state

            else:
                # we reach here if
                # min RCT=1 and len(RCT) > 1 and len(iat) >= 1 (min(iat) > 1 when len(iat)=1) or
                # min(RCT) > 1 and len(iat) > 1 and min(iat) = 1
                if iat_dist[0][0] == 1:  # one of the successor states is the initial state of task i=j
                    # len(iat) > 1 here
                    if rct_dist[0][0] == 1 and len(rct_dist) > 1:
                        possible_states_task_j.append((rct_dist[0][1] * iat_dist[0][1], init_config_j, 0))  # initial state with task finished
                        if hard:
                            possible_states_task_j.append(((1 - rct_dist[0][1]) * iat_dist[0][1], init_config_j, 0))
                        else:
                            possible_states_task_j.append(((1 - rct_dist[0][1]) * iat_dist[0][1], init_config_j, 1))  # initial state with task not finished
                        possible_states_task_j.append((rct_dist[0][1] * (1 - iat_dist[0][1]), [[dec_rct_dist[0]], deadline, dec_iat_dist[1:]], 0))
                        possible_states_task_j.append(((1-rct_dist[0][1]) * (1-iat_dist[0][1]), [dec_rct_dist[1:], deadline, dec_iat_dist[1:]], 0))

                    elif rct_dist[0][0] == 1:  # and len(rct_dist) == 1
                        possible_states_task_j.append((iat_dist[0][1], init_config_j, 0))
                        possible_states_task_j.append((1-iat_dist[0][1], [dec_rct_dist, deadline, dec_iat_dist[1:]], 0))

                    else:  # rct_dist[0][0] > 1
                        if hard:
                            possible_states_task_j.append((iat_dist[0][1], init_config_j, 0))
                        else:
                            possible_states_task_j.append((iat_dist[0][1], init_config_j, 1))
                        possible_states_task_j.append((1-iat_dist[0][1], [dec_rct_dist, deadline, dec_iat_dist[1:]], 0))

                else:  # min(iat) > 1
                    # Here min RCT=1 and len(RCT) > 1
                    possible_states_task_j.append((rct_dist[0][1], [[dec_rct_dist[0]], deadline, dec_iat_dist], 0))
                    possible_states_task_j.append((1-rct_dist[0][1], [dec_rct_dist[1:], deadline, dec_iat_dist], 0))

        else:  # when j!=i is scheduled
            possible_states_task_j = add_possible_states_task_j(rct_dist, iat_dist, init_config_j, deadline,
                                                                dec_iat_dist, hard)
        raw_states.append(possible_states_task_j)

    states = compute_states(raw_states, hard)
    len_tasks = len(tasks)
    states = filter(lambda prob_next_badflag: len(prob_next_badflag[1])==len_tasks, states)

    return i, states



def get_states_execute_none(all_tasks, tasks, hard):
    # This function returns a tuple (-1, states), where states is the set of all states obtained from the current state
    # all_tasks when no task is executed. The slacks can be used by soft tasks

    raw_states = []

    for j in range(len(all_tasks)):
        task = all_tasks[j]
        init_config_j = tasks[j][1:4]  # gives rct, deadline, iat in the current config

        # normalize distributions so that the probabilities sum up to 1
        rct_dist = normalize_dist(task[0])
        iat_dist = normalize_dist(task[2])

        dec_rct_dist, deadline, dec_iat_dist = modify_params(rct_dist, task[1], iat_dist)

        possible_states_task_j = add_possible_states_task_j(rct_dist, iat_dist, init_config_j, deadline, dec_iat_dist, hard)
        raw_states.append(possible_states_task_j)

    states = compute_states(raw_states, hard)
    len_tasks = len(tasks)
    states = filter(lambda prob_next_badflag: len(prob_next_badflag[1])==len_tasks, states)

    return -1, states  # -1 denotes no task has been executed



def get_new_states(all_tasks, tasks, hard):
    # all_tasks is a list of [rct dist, deadline, iat dist] elements, one for each task. It denotes the current state
    # tasks is a list of the task descriptions [arrival, exe dist, deadline, period dist, max_exe_time, min_arrive_time]
    # hard is a flag that is set to True for hard tasks and False for soft tasks

    # This function returns new_states which is a list of elements of the form (i, states), where
    # i -- (start index 0) is the task that is executed, -1 when no task is executed and
    # states -- is the list of states reached from current state all_tasks after executing task i.
    # Every state is a list of the form [rct dist, deadline, iat dist]

    new_states = []
    for i in range(len(all_tasks)):
        # execute task i if min rct > 0

        task_i_rct_dist = all_tasks[i][0]
        task_i_deadline = all_tasks[i][1]
        if task_i_rct_dist[0][0] > 0 and task_i_deadline > 0:
            # get_states_execute_i returns the set of states that are obtained by executing task i from current state all_tasks
            # and are locally known to be safe as guaranteed by function safe, i.e., max rct <= deadline
            new_states.append(get_states_execute_i(all_tasks, i, tasks, hard))

    # add states when no task is executed
    #if len(new_states) == 0:
    new_states.append((get_states_execute_none(all_tasks, tasks, hard)))

    return new_states



def normalize_state(state):

    norm_state = []

    for task in state:
        [rct_dist, deadline, iat_dist] = task
        mod_task = [normalize_dist(rct_dist), deadline, normalize_dist(iat_dist)]
        norm_state.append(mod_task)

    return norm_state



def add_new_states(G, next_states, node_num, new_node_cnt, costFromOrigin):
    # next_states is a list of the form [(0, states1) (1, states2) ... (n,statesn) ... (-1, states)] and
    # each of states1, states2, ... statesn, states is a list of elements the form (prob, state)

    global edgeNumber
    global M

    for j, next_states_exe_j in next_states:

        if costFromOrigin > 0:
            M[1] += "+ " + str(costFromOrigin) + " x" + str(edgeNumber) + " "

        M[2][node_num-1] += " + x" + str(edgeNumber)
        M[3] += "+ x" + str(edgeNumber) + " "

        for prob, state, badflag in next_states_exe_j:
            # normalize the distributions of all tasks in a state
            state = normalize_state(state)

            # check if the state is already present in G, the edge from current_node to state still needs to be added
            nodes_with_same_param = filter(lambda n_d: n_d[1]['config'] == (state, badflag), G.nodes(data=True))

            node_num_same_param  = [x[0] for x in nodes_with_same_param]

            # There should be at most one node with the same param
            len_node_num_same_param = len(node_num_same_param)
            # assert len_node_num_same_param <= 1

            if len_node_num_same_param == 0:  # no existing node with the same config
                # add a new node
                new_node_cnt+=1
                M[2].append("r_" + str(new_node_cnt) + ":")
                G.add_node(new_node_cnt, config=(state, badflag))

                # add an edge from the current node to the new node with the task that is executed and the probability
                M[2][new_node_cnt-1] += " - " + str(float(prob)) + " x" + str(edgeNumber)
                G.add_edge(node_num, new_node_cnt, label=(j, prob, 0, costFromOrigin, edgeNumber))

            else:  # exists a node with the same config
                # add an edge from the current node to the existing node
                M[2][node_num_same_param[0]-1] += " - " + str(float(prob)) + " x" + str(edgeNumber)
                G.add_edge(node_num, node_num_same_param[0], label=(j, prob, 0, costFromOrigin, edgeNumber))

        edgeNumber += 1

    return G, new_node_cnt



def construct_graph(tasks, hard):

    global M
    # Build the graph of product of hard tasks when hard is true, else build the graph for a soft task
    start_time = time.time()
    G = nx.DiGraph()

    # extract the parameters for each task
    state = []
    for x in tasks:
        if x[0] > 0:  # initial arrival time > 0
            #task = [0, 0, x[0]]
            # change to fraction
            #task = [[(0,1)], 0, [(x[0],1)]]
            task = [[(0,Fraction(1,1))], 0, [(x[0],Fraction(1,1))]]

        else:
            task = [x[1], x[2], x[3]]  # [RCT dist, deadline, Period dist]

        state.append(task)

    # construct the initial state
    node_cnt = 1
    G.add_node(node_cnt, config=(state, [0]))
    M[2].append("r_" + str(node_cnt) + ":")

    # construct the transitions
    queue = [node_cnt]

    while len(queue) > 0:
        (queue_head, queue) = pop_queue(queue)
        next_states = get_new_states(G.nodes[queue_head]['config'][0], tasks, hard)
        originFlags = G.nodes[queue_head]['config'][1]
        costFromOrigin = 0
        index = 0
        for flag in originFlags:
            if flag:
                costFromOrigin += float(tasks[index][6])
                index += 1
        if len(next_states) > 0:
            first_elem_to_push = queue_head+1 if not queue else queue[-1]+1
            G, new_node_cnt = add_new_states(G, next_states, queue_head, first_elem_to_push-1, costFromOrigin)

            # push the new nodes to the end of the queue so that they can be explored further to extend G
            queue = queue + list(range(first_elem_to_push, new_node_cnt+1))

    time_reqd = time.time() - start_time
    return G



def get_reachable_nodes(G):

    queue = [1]
    visited = []
    while len(queue) > 0:
        (n, queue) = pop_queue(queue)

        if not n in visited:
            succ_nodes = list(G.successors(n))
            queue = queue + succ_nodes
            visited.append(n)

    return visited



def restrict_to_safe_states(G, len_hard_tasks):

    start_time = time.time()
    labels = range(len_hard_tasks) + [-1]

    # Remove those nodes for which every action may lead to an unsafe node to false
    flag = True
    unsafe_nodes = []

    while flag:
        flag = False

        for n in list(G.nodes()):

            to_remove = True
            for j in labels:
                edges_task_j = filter(lambda n1_n2_d: n1_n2_d[2]['label'][0] == j, G.edges(n, data=True))

                sum_prob_task_j = sum([d['label'][1] for (n1, n2, d) in edges_task_j])

                if sum_prob_task_j == 1:  # for action j all possible nodes are safe
                    to_remove = False

                elif edges_task_j:  # there are some edge for task j but their probabilities do not add up to 1
                    # Here 0 < sum_prob_task_j < 1
                    G.remove_edges_from(edges_task_j)

            if to_remove:
                # remove unsafe node and all the incoming and outgoing edges
                G.remove_node(n)
                flag = True

                # Debug
                unsafe_nodes.append(n)

    # Since some edges were removed, it is possible that some nodes become unreachable
    # Remove the unreachable nodes
    if G.nodes():
        unreachable_nodes = set(G.nodes()) - set(get_reachable_nodes(G))
        G.remove_nodes_from(unreachable_nodes)

        # Debug
        # print("%d Unreachable nodes" % len(unreachable_nodes))
        # print(unreachable_nodes)

    # Debug
    # print("Final list of nodes: ")
    # print(list(G.nodes()))
    # print("%d Unsafe nodes" % len(unsafe_nodes))
    # print(unsafe_nodes)
    time_reqd = time.time() - start_time
    r= open("results.txt", "a+")
    r.write("Num safe states %d, " % len(G.nodes()))
    r.write("Construct safe model: %s seconds \r\n\r\n" % time_reqd)

    return G



def exists_harder(s, states):
    # extract the support of each task from s
    s_tasks_rct = [x[0] for x in s]

    for state in states:

        if state != s:
            state_tasks_rct = [x[0] for x in state]

            s_is_harder = False

            for i in range(len(s_tasks_rct)):  # i is the task number
                if len(s_tasks_rct[i]) == len(state_tasks_rct[i]):

                    if s_tasks_rct[i][0][0] > state_tasks_rct[i][0][0]:
                        s_is_harder = True
                        break

                elif len(s_tasks_rct[i]) > len(state_tasks_rct[i]):
                    s_is_harder = True
                    break

            if not s_is_harder:
                return True

    return False



def compute_antichain(G):
    # Put all states in a dictionary whose keys are the deadlines of each task. We compare every task in two states
    # such that the remaining arrival time distributions are the same for each task in both the states.
    # The remaining arrival time distribution being the same denotes that the deadlines of the tasks are also the same
    # in the two states
    task_dict = {}
    for n in G.nodes():
        # extract the deadlines for each task in each node
        tasks = G.node[n]['config'][0]

        deadlines = [x[1] for x in tasks]

        if task_dict.has_key(str(deadlines)):
            states = task_dict[str(deadlines)]
            states.append(tasks)
            task_dict[str(deadlines)] = states

        else:
            task_dict[str(deadlines)] = [tasks]

    # Compute the set of maximal elements by removing the easy states
    keys = task_dict.keys()
    cnt = 0  # count of anti-chain elements

    for deadlines in keys:
        states = task_dict[str(deadlines)]

        new_states = []
        for s in states:
            # check if there exists a state in states that is harder than s
            if not exists_harder(s, states):
                cnt += 1
                new_states.append(s)

        task_dict[str(deadlines)] = new_states

    print("%d Number of antichain elements " % cnt)
    print("%d Number of elements in G " % len(G.nodes()))

    return task_dict



def construct_graphs_soft_tasks(soft_tasks):

    soft_graphs_list = []
    name = 1
    for task in soft_tasks:
        G = construct_graph([task], False)
        soft_graphs_list.append(G)
        #drawMDP(G, str(name) + ".pdf")
        name += 1

    return soft_graphs_list



def write_intro_mdp(f):

    f.write("mdp\r\n\r\n")


def write_intro(f, num_hard_tasks, num_soft_tasks):
    # Here we write the player definitions at the beginning of the PRISM file

    f.write("smg\r\n\r\n")
    f.write("player p1\r\n")
    actions = "\t"
    for j in range(num_hard_tasks):
        actions = actions+"[hard"+str(j+1)+"]"
        #if j< num_hard_tasks-1:
        actions = actions + ', '

    actions = actions + "[none]"

    if num_soft_tasks > 1:
        actions = actions + ', '

    for j in range(num_soft_tasks):
        actions = actions+", [soft"+str(j+1)+"]"

    f.write(actions+"\r\n")
    f.write("endplayer\r\n\r\n")

    f.write("player p2\r\n")
    actions = "\thard_tasks"
    if num_soft_tasks > 0:
        actions = actions + ', '

    for j in range(num_soft_tasks):
        actions = actions+"soft_task"+str(j+1)
        if j< num_soft_tasks-1:
            actions = actions + ','

    f.write(actions+"\r\n")
    f.write("endplayer\r\n\r\n")



def write_init_params(f, tasks, hard, task_num):

    pre_str = ""
    if not hard:
        pre_str = "s"

    for j in range(len(tasks)):
        if hard:
            [arrival, exe, deadline, period, max_exe_time, min_arrive_time] = tasks[j]

        else:
            [arrival, exe, deadline, period, max_exe_time, min_arrive_time, cost] = tasks[j]

        if arrival == 0:
            init_exe = [i[0] for i in exe]
            d = deadline
            init_period = [i[0] for i in period]

        else:
            init_exe = [0]
            init_exe = init_exe + [-1] * (len(exe)-1)

            d = 0
            init_period = [arrival]
            init_period = init_period + [-1] * (len(period)-1)

        if hard:
            task_num = j

        for e in range(len(exe)):
            if e > 0:
                min_val = -1
            else:
                min_val = 0
            # f.write("\t"+pre_str+"rct"+str(task_num+1)+"_"+str(e+1)+": ["+str(min_val)+".."+str(exe[e][0])+"] init "+str(init_exe[e])+";\r\n")
            f.write("\t"+pre_str+"rct"+str(task_num+1)+"_"+str(e+1)+": ["+str(min_val)+".."+str(max_exe_time)+"] init "+str(init_exe[e])+";\r\n")

        f.write("\t"+pre_str+"d"+str(task_num+1)+": [0.."+str(deadline)+"] init "+str(d)+";\r\n")

        max_period = 0
        for p in range(len(period)):
            if period[p][0] > max_period:
                max_period = period[p][0]

        for p in range(len(period)):
            if p > 0:
                min_val = -1
            else:
                min_val = 0
            # f.write("\t"+pre_str+"p"+str(task_num+1)+"_"+str(p+1)+": ["+str(min_val)+".."+str(period[p][0])+"] init "+str(init_period[p])+";\r\n")
            f.write("\t"+pre_str+"p"+str(task_num+1)+"_"+str(p+1)+": ["+str(min_val)+".."+str(max_period)+"] init "+str(init_period[p])+";\r\n")

        if not hard:
            f.write("\tf"+str(task_num+1)+": [0..1] init 0;\r\n")

        f.write("\r\n")



def get_antecedents(G, n, tasks, hard, task_num):
# This function returns the string on the left side of -> for a transition of the tasks

    len_tasks = len(tasks)  # len_tasks is 1 when called for a soft task
    string_to_write = ""

    if hard:
        pre_str = ""
    else:
        pre_str = "s"

    for j in range(len_tasks):
        [rct_taskj_in_n, deadline_taskj_in_n, iat_taskj_in_n] = G.nodes[n]['config'][0][j]

        if hard:
            [arrival, exe, deadline, period, max_exe_time, min_arrive_time] = tasks[j]
        else:
            [arrival, exe, deadline, period, max_exe_time, min_arrive_time, cost] = tasks[j]

        # Write the antecedent
        if j > 0:
            string_to_write = string_to_write + " & "

        if hard:
            task_num = j

        for ind in range(len(rct_taskj_in_n)):
            string_to_write = string_to_write + pre_str + "rct" + str(task_num + 1) + "_" + str(ind+1) + "=" + str(
                rct_taskj_in_n[ind][0]) + " & "

        unused_exe_inds = list(set(range(len(exe))) - set(range(len(rct_taskj_in_n))))

        # set the remaining exe indices to -1
        for ind in unused_exe_inds:
            string_to_write = string_to_write + pre_str + "rct" + str(task_num + 1) + "_" + str(ind+1) + "=" + str(-1) + " & "

        string_to_write = string_to_write + pre_str + "d" + str(task_num + 1) + "=" + str(deadline_taskj_in_n) + " & "

        for ind in range(len(iat_taskj_in_n)):
            string_to_write = string_to_write + pre_str + "p" + str(task_num + 1) + "_" + str(ind+1) + "=" + str(iat_taskj_in_n[ind][0])
            if ind < len(period)-1:
                string_to_write = string_to_write + " & "

        unused_period_inds = list(set(range(len(period))) - set(range(len(iat_taskj_in_n))))

        # set the remaining period indices to -1
        for ind in unused_period_inds:
            string_to_write = string_to_write  + pre_str + "p" + str(task_num + 1) + "_" + str(ind+1) + "=" + str(-1)
            if ind != unused_period_inds[-1]:
                string_to_write = string_to_write + " & "

    string_to_write = string_to_write + " -> "

    return string_to_write



def get_implied_string_action_j(G, n, j, tasks, hard, task_num):
# This function returns the string on the right side of -> for a transition of the hard tasks

    edges_task_j = list(filter(lambda n1_n2_d: n1_n2_d[2]['label'][0] == j, G.edges(n, data=True)))
    implied_str = ""
    len_tasks = len(tasks)
    if hard:
        pre_str = ""
    else:
        pre_str = "s"

    # loop for all edges that correspond to task j
    for edge in edges_task_j:
        dest = edge[1]
        prob = edge[2]['label'][1]

        if prob != 1:  # it is not needed to write the probability when it is 1
            implied_str = implied_str + str(prob) + " : "

        # build the string for each task k in the destination
        for k in range(len_tasks):
            if k > 0:
                implied_str = implied_str + " & "

            [rct_taskk_in_dest, deadline_taskk_in_dest, iat_taskk_in_dest] = G.nodes[dest]['config'][0][k]

            if hard:
                [arrival, exe, deadline, period, max_exe_time, min_arrive_time] = tasks[k]
            else:
                [arrival, exe, deadline, period, max_exe_time, min_arrive_time, cost] = tasks[k]

            if not hard:
                badflag = G.nodes[dest]['config'][1][k]
                implied_str = implied_str + "(f" + str(task_num + 1) + "'=" + str(badflag) + ") & "

            if hard:
                task_num = k

            for ind in range(len(rct_taskk_in_dest)):
                implied_str = implied_str + "(" + pre_str + "rct" + str(task_num + 1) + "_" + str(ind+1) + "'=" + str(
                    rct_taskk_in_dest[ind][0]) + ") & "

            unused_exe_inds = list(set(range(len(exe))) - set(range(len(rct_taskk_in_dest))))

            # set the remaining exe indices to -1
            for ind in unused_exe_inds:
                implied_str = implied_str + "(" + pre_str + "rct" + str(task_num + 1) + "_" + str(ind + 1) + "'=" + str(-1) + ") & "

            implied_str = implied_str + "(" + pre_str + "d" + str(task_num + 1) + "'=" + str(deadline_taskk_in_dest) + ") &"

            for ind in range(len(iat_taskk_in_dest)):
                implied_str = implied_str + " ("+ pre_str + "p" + str(task_num + 1) + "_" + str(ind+1) + "'=" + str(
                    iat_taskk_in_dest[ind][0]) + ")"
                if ind < len(period)-1:
                    implied_str = implied_str + " &"

            unused_period_inds = list(set(range(len(period))) - set(range(len(iat_taskk_in_dest))))

            # set the remaining period indices to -1
            for ind in unused_period_inds:
                implied_str = implied_str + " ("+ pre_str + "p" + str(task_num + 1) + "_" + str(ind + 1) + "'=" + str(-1) + ")"
                if ind != unused_period_inds[-1]:
                    implied_str = implied_str + " &"

        if edge != list(edges_task_j)[-1]:  # edge is not the last edge in the list edges_task_j
            implied_str = implied_str + " + "

    return implied_str



def write_transitions(f, G, hard_tasks, len_soft_tasks):
    # iterate over the nodes and write the transitions

    for n in G.nodes():

        string_to_write = get_antecedents(G, n, hard_tasks, True, -1)

        # Find the edges with the same action j, that is, hard task j is scheduled from node n
        for j in range(len(hard_tasks)):
            antecedent = "\t[hard" + str(j + 1) + "] " + string_to_write
            implied_str = get_implied_string_action_j(G, n, j, hard_tasks, True, -1)

            if implied_str:  # implied_str is non-empty if hard task j can be scheduled from n
                f.write(antecedent + implied_str + ";\r\n\r\n")

        # Find the edges when no task is scheduled
        implied_str = get_implied_string_action_j(G, n, -1, hard_tasks, True, -1)
        for j in range(len_soft_tasks):

            if implied_str:  # implied_str is non-empty if some soft task can be scheduled from n
                antecedent = "\t[soft" + str(j + 1) + "] " + string_to_write
                f.write(antecedent + implied_str + ";\r\n\r\n")

        # add a none transition
        antecedent = "\t[none] " + string_to_write

        if implied_str:  # implied_str is non-empty if some soft task can be scheduled from n
            f.write(antecedent + implied_str + ";\r\n\r\n")



def write_module_hard_tasks(f, G_hard, hard_tasks, len_soft_tasks):
    # G_hard is the graph of the product of the hard tasks
    # This function writes down the module for the product of the hard tasks

    f.write("module hard_tasks\r\n\r\n")

    write_init_params(f, hard_tasks, True, -1)
    write_transitions(f, G_hard, hard_tasks, len_soft_tasks)

    f.write("\r\nendmodule\r\n\r\n")



def write_transitions_soft(f, G, soft_task, task_num, len_hard_tasks, len_soft_tasks):

    visited_states = []

    for n in G.nodes():
        # G.node[n]['config'] is of the form (state, bad_flag)
        # We ignore bad_flag in individual transitions on the left side when writing the Prism file
        if not G.nodes[n]['config'][0] in visited_states:
            visited_states.append(G.nodes[n]['config'][0])

            string_to_write = get_antecedents(G, n, soft_task, False, task_num)

            # Find the edges with the same action j, that is, soft task j is scheduled from node n
            antecedent = "\t[soft" + str(task_num + 1) + "] " + string_to_write
            implied_str = get_implied_string_action_j(G, n, 0, soft_task, False, task_num)

            if implied_str:  # implied_str is non-empty if soft task j can be scheduled from n
                f.write(antecedent + implied_str + ";\r\n\r\n")

            # Find the edges when no task is scheduled
            implied_str = get_implied_string_action_j(G, n, -1, soft_task, False, task_num)

            if implied_str:  # implied_str is non-empty if some soft task can be scheduled from n
                for j in range(len_hard_tasks):
                    antecedent = "\t[hard" + str(j + 1) + "] " + string_to_write
                    f.write(antecedent + implied_str + ";\r\n\r\n")

                for j in range(len_soft_tasks):
                    if j != task_num:
                        antecedent = "\t[soft" + str(j + 1) + "] " + string_to_write
                        f.write(antecedent + implied_str + ";\r\n\r\n")

                # add a [none] transition
                antecedent = "\t[none] " + string_to_write
                f.write(antecedent + implied_str + ";\r\n\r\n")



def write_module_soft_tasks(f, soft_graphs, hard_tasks, soft_tasks):

    cnt = 1
    for G in soft_graphs:
        f.write("module soft_task" + str(cnt) + "\r\n\r\n")

        write_init_params(f, [soft_tasks[cnt-1]], False, cnt-1)
        write_transitions_soft(f, G, [soft_tasks[cnt-1]], cnt-1, len(hard_tasks), len(soft_tasks))

        f.write("\r\nendmodule\r\n\r\n")
        cnt+=1



def generate_schedule_hard(j, len_hard_tasks):

    string_to_write = "d" + str(j+1) + " > 0"

    for k in range(len_hard_tasks):
        if k != j:
            string_to_write = string_to_write + " & (d" + str(j+1) + " <= d" + str(k+1) + " | d" + str(k+1) + "=0 | rct" + str(k+1) + "_1=0)"

    return string_to_write



def generate_schedule_soft(j, len_hard_tasks, len_soft_tasks):

    string_to_write = "sd" + str(j+1) + " > 0"

    for k in range(len_hard_tasks):
        string_to_write = string_to_write + " & (d" + str(k+1) + "=0 | rct" + str(k+1) + "_1=0)"

    for k in range(len_soft_tasks):
        if k != j:
            string_to_write = string_to_write + " & (sd" + str(j+1) + " <= sd" + str(k+1) + " | sd" + str(k+1) + "=0 | srct" + str(k+1) + "_1=0)"

    return string_to_write



def generate_schedule_none(j, len_hard_tasks, len_soft_tasks):

    string_to_write = ""
    for k in range(len_hard_tasks):
        if k == 0:
            pre_str = ""
        else:
            pre_str = " &"
        string_to_write = string_to_write + pre_str + " (d" + str(k+1) + "=0 | rct" + str(k+1) + "_1=0)"

    if len_hard_tasks > 0:
        pre_str = " &"
    else:
        pre_str = ""

    for k in range(len_soft_tasks):
        string_to_write = string_to_write + pre_str + " (sd" + str(k + 1) + "=0 | srct" + str(k + 1) + "_1=0)"

    return string_to_write



def write_scheduler_EDF(f, len_hard_tasks, len_soft_tasks):

    f.write("module scheduler\r\n\r\n")
    # flag is a dummy variable needed for Prism syntax
    f.write("\tflag : [0..1] init 0;\r\n\r\n")

    for j in range(len_hard_tasks):
        f.write("\t[hard" + str(j + 1) + "] " + generate_schedule_hard(j, len_hard_tasks) + " -> (flag' = flag);\r\n\r\n")

    for j in range(len_soft_tasks):
        f.write("\t[soft" + str(j + 1) + "] " + generate_schedule_soft(j, len_hard_tasks, len_soft_tasks) + " -> (flag' = flag);\r\n\r\n")

    f.write("\t[none] " + generate_schedule_none(j, len_hard_tasks, len_soft_tasks) + " -> (flag' = flag);\r\n\r\n")

    f.write("endmodule\r\n\r\n")



def write_rewards(f, soft_tasks):

    f.write("rewards\r\n\r\n")

    for j in range(len(soft_tasks)):
        f.write("\r\n")
        for k in range(len(soft_tasks) + 1):
            if k == len(soft_tasks):
                f.write("\t[none] f" + str(j+1) + "= 1 : " + str(soft_tasks[j][6]) + ";\r\n")
            else:
                f.write("\t[soft" + str(k+1) + "] f" + str(j+1) + "= 1 : " + str(soft_tasks[j][6]) + ";\r\n")

    f.write("\r\n")

    f.write("endrewards\r\n")



def generate_prism_file(task_graphs_list, out_file, hard_tasks, soft_tasks):

    f = open(out_file, "w+")
    # write_intro(f, len(hard_tasks), len(soft_tasks))
    write_intro_mdp(f)

    task_graphs_list_soft_ind = 0
    if len(hard_tasks) > 0:
        write_module_hard_tasks(f, task_graphs_list[0], hard_tasks, len(soft_tasks))
        task_graphs_list_soft_ind = 1

    if len(soft_tasks) > 0:
        write_module_soft_tasks(f, task_graphs_list[task_graphs_list_soft_ind:], hard_tasks, soft_tasks)

        # EDF on hard tasks followed by EDF on soft tasks
        # write_scheduler_EDF(f, len(hard_tasks), len(soft_tasks))

        write_rewards(f, soft_tasks)

    f.close()



def endSol(M):

    for i in range(len(M[2])):
        M[2][i] += " = 0\n"
    M[3] += "= 1"

    return M


def writeLP(M, fileName):

    m = open(fileName, "w")
    m.write("Minimize\n")
    m.write(M[1])
    m.write("\nSubject To\n")
    m.writelines(M[2])
    m.write(M[3])
    m.write("\nEnd")
    m.close()


def optimize(model):

    model = gp.read(model)
    #model.setParam(GRB.Param.Presolve, 0)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        model.write('model.sol')


def retreiveSol():

    sol = {}
    s = open("model.sol", "r")
    for line in s.readlines():
        if line[0] != "#":
            elem = line.split()
            sol[elem[0]] = elem[1]
    s.close()

    return sol


def solToMDP(G, sol):

    for node in G.nodes(data=True):

        dic = {}

        for edge in G.out_edges(node[0]):
            edgeNum = G.get_edge_data(edge[0], edge[1])["label"][4]
            prob = sol["x" + str(edgeNum)]
            if edgeNum not in dic:
                dic[edgeNum] = float(prob)

        sumProb = sum(dic.values())

        for edge in G.out_edges(node[0]):

            edgeNum = G.get_edge_data(edge[0], edge[1])["label"][4]
            executedTask = G.get_edge_data(edge[0], edge[1])["label"][0]
            prob = float(sol["x" + str(edgeNum)])
            if prob > 0:
                prob = prob / sumProb
                G[edge[0]][edge[1]]["color"] = "red"
            G[edge[0]][edge[1]]["label"] = ("x" + str(edgeNum), float(prob), executedTask)

        label = makeNodeLabel(node)
        newlabel = []
        for elem in label:
            newlabel.append(elem)
            if elem != label[-1]:
                newlabel.append('\n')

        node[1]["label"] = newlabel


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


def drawMDP(G, pdfFile):

    A = nx.nx_agraph.to_agraph(G)
    A.write("AGraph.dot")
    toExecute = "dot -T pdf AGraph.dot -o " + pdfFile
    os.system(toExecute)
    os.remove("AGraph.dot")


def main():

    # file_name = raw_input("Enter file name: ")
    file_name = sys.argv[1]
    file_name = file_name.strip()
    file_name = "TasksSet/" + file_name
    hard_tasks, soft_tasks = get_tasks(file_name)  # hard_tasks and soft_tasks are a list of task descriptions: [arrival, exe dist, deadline, period dist, max_exe_time, min_arrive_time] elements

    """
    G = construct_graph(soft_tasks, False)

    LP = endSol(M) #Brings closure to the LP strings so that it contains the correct LP format (currently : format supported by Gurobi)

    model = sys.argv[2]
    model = "LPFile/" + model

    writeLP(M, model) #Writes the LP file

    optimize(model) #Gurobi Optimization

    sol = retreiveSol() #Retreive solution computed by Gurobi in a dic

    solToMDP(G, sol) #Use the solution to modify MDP's labels

    if len(soft_tasks) < 3:
        pdfFile = sys.argv[3]
        pdfFile = "MDP_PDF/" + pdfFile
        drawMDP(G, pdfFile) #Draw the MDP on a PDF if the set of tasks is not too big (Maximim = 2, otherwise the MDP in unreadable)
    """

    task_graphs_list = construct_graphs_soft_tasks(soft_tasks)
    # Generate the prism file
    file_name = sys.argv[1]
    out_file = file_name.split(".")
    generate_prism_file(task_graphs_list, out_file[0] + '.pm', hard_tasks, soft_tasks)  # .pm extension for mdp


if __name__== "__main__":
    main()
