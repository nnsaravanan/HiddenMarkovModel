from decimal import Decimal

class Node:
    """Class that stores a node in a Table"""
    def __init__(self,v,p,point=None):
        # stores hmm state, probability and ap pointer to the previous node
        self.v = v
        self.p = p
        self.point = point
        
    # basic conditional statements to check with max function
    def __eq__(self,other):
        return self.p == other.p
    def __gt__(self,other):
        return self.p > other.p
    def __lt__(self,other):
        return self.p < other.p
    
    # useful for debugging with print
    def __str__(self):
        return str((self.v,self.p))
    def __repr__(self):
        return str((self.v,self.p))


def normalize(A):
    """Normalizes all probabilities of nodes in a given list"""

    # calculates the total probabilites of all items in the list
    total = 0
    for node in A:
        total+= node.p
    
    # divides each probability by the total value
    for node in A:
        node.p = node.p/total


def calc_correct(s1,s2):
    """Returns percentage of matching characters in 2 strings to 3 decimal places"""

    # counts the matches and divides by the total 
    tot = ct = 0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
            ct+=1
        tot +=1

    return round(ct/tot*100,3)


def convert_to_string(A,hidden_states):
    """Converts a list of HMM states to a string of Hidden States"""

    temp = list()
    for num in A:
        temp.append(hidden_states[num])

    return "".join(temp)

def decimalize(x):
    """Returns Decimal of a number given"""
    return Decimal(x)