# coding: utf-8

# # Branch-and Bound to solve Integer Linear Programming problems

# **Authors: HE Chenchen**

# **Import the libraries**
import numpy as np
from scipy.optimize import linprog
# import math

# **Function "isInteger(x)"**
def isInteger(x):
    """
    As for a vector x,check if it contains only integer values:
    return the boolean value True and the None, i.e., [True,None] when the values are all integer
    return False and the index of the noninteger value in x, i.e., [False,index]

    """
    xx = np.array(x)
    dist = np.array(abs(np.rint(xx)-xx))
    for xx in dist:
        if float(xx).is_integer() == False:
            dist[dist == 0] = np.nan
            return (False, np.nanargmin(dist))
    return (True, None)


# **Class "Node"**
class Node:
    """
    This class models a node in the branch and bound algorithm.
    x: the solution in node
    z: the value of node
    status: the status of the node, if this node is availabe

    """
    def __init__(self, x, z, bounds, status):
        self.x = x
        self.z = z
        self.bounds = bounds
        self.status = status

# Definition of the integer LP problem

#==========assignment 8.1 ====================
c = np.array([10, 20])
A = np.array([[5, 8], [1, 0], [0, 1]])
b = np.array([60, 8, 4])


#=========assignment 8.2 ====================
'''
c = np.array([2,3,1,2])
A = np.array([[5,2,1,1],[2,6,10,8],[1,1,1,1],[2,2,3,3],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
b = np.array([15,60,8,16,3,7,5,5])
'''

A_eq = None
b_eq = None
n = len(c)

# two ways to initialize bounds automatically

# bounds=np.full((n,2),(0,None))
bounds = [[0, None] for _ in range(n)]


# **Initialization of the branch-and-bound algorithm**

"""
Initialization of the branch-and-bound algorithm

we have three main input parameters, i.e., node, bestnode, iteration
node: the node we need to proceed
bestnode: is used to store the best solution of the branch-and-bound algorithm
iteration: the iteration of processing node

"""
# Initializing the node
init_res = linprog(-c, A, b, A_eq, b_eq)
x = init_res.x
z = init_res.fun
status = init_res.status
node = Node(x, z, bounds, status)

# Initialize the bestnode
bstnode_status = 0
bestnode = Node([0, 0], 0, bounds, bstnode_status)

# Initialize the iteration
iteration = 0

# Branch-and-Bound recursive processing
def branch_bound(node, bestnode, iteration):
    # decide the node is available or not
    if node.status != 0:
        return (bestnode,iteration)

    # decide update the best solution or not
    if node.z < bestnode.z:
        (isIntegerOrNot,splitIndex) = isInteger(node.x)

        # decide if the solution is integer
        if isIntegerOrNot is True:
            # update the bestnode
            bestnode = node
            # return the bestnode
            return (bestnode, iteration)
        else:
            # split the node to find the bestnode
            # choose a functional variable to split node
            spValue = node.x[splitIndex]
            # one child with conditional constraints x<=lowbound
            lowbound = np.floor(spValue)
            boundsL = np.array(node.bounds)
            boundsL[splitIndex] = [0, lowbound]
            # process the child node and insert them into the <node> to find the bestnode
            resL = linprog(-c, A, b, A_eq, b_eq, boundsL)
            nodeL = Node(resL.x, resL.fun, boundsL, resL.status)
            iteration = iteration+1
            print("Iteration number is: "+str(iteration) + "\nProcessed node: Solution x:{0}, Value:{1}, Bounds:{2}".format(nodeL.x, -nodeL.z, boundsL) + "\nSplit on x_"+str(splitIndex+1))
            if nodeL.z <= bestnode.z:
                # start the recursive processing
                (bestnode, iteration) = branch_bound(nodeL, bestnode, iteration)
            else:
                # one child with conditional constraints x>=upbound
                upbound = np.ceil(spValue)
                boundsR = np.array(node.bounds)
                boundsR[splitIndex] = [upbound, None]
                # process the child node and insert them into the <node> to find the bestnode
                resR = linprog(-c, A, b, A_eq, b_eq, boundsR)
                nodeR = Node(resR.x, resR.fun, boundsR, resR.status)
                iteration = iteration + 1
                print("Iteration number is: "+str(iteration)+ "\nProcessed node: Solution x:{0}, Value:{1}, Bounds:{2}".format(nodeR.x, -nodeR.z, boundsR)+"\nSplit on x_"+str(splitIndex+1))
                if nodeR.z <= bestnode.z:
                    # start the recursive processing
                    (bestnode, iteration) = branch_bound(nodeR, bestnode, iteration)
    else:
        # return the bestnode
        return(bestnode, iteration)
    # return the bestnode
    return (bestnode, iteration)


# **Final messages to the users**

# print("Original node: x: {0}, Value:{1}, Bounds:{2}".format(node.x,node.z,node.bounds))
# print("Optimal solution x:{0}, Value:{1}, Bounds:{2}".format(bestnode.x,-bestnode.z, bestnode.bounds))

(bestnode, iteration) = branch_bound(node, bestnode, iteration)
print("\nAn integer solution is found by improvement!\n")
