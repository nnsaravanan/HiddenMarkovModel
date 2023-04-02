import numpy as np
from Helpful import Node, decimalize
    
def viterbi(initial_prob, transition_prob, dice_prob, seq):

    """
    Uses the Viterbi algorithm to find the most likely sequence of hidden states that generates a 
    given sequence of observed characters in a Hidden Markov Model.

    Inputs:

        initial_prob (list): A list of the initial probabilities of each hidden state.
        
        transition_prob (2D list): A square matrix representing the transition probabilities between hidden states.

        dice_prob (2D list): An emmission matrix that has a shape of unique emissions * hidden states, representing the probability 
        of each unique emission visible at a hidden state.

        seq (list): A list of the observed characters.

    Outputs:

        final (list): A list representing the most likely sequence of HMM states corresponding to the observed characters in seq.

    """

    # Creates precise decimal values for small decimals in computation 
    initial_prob = np.vectorize(decimalize)(initial_prob)
    transition_prob = np.vectorize(decimalize)(transition_prob)
    dice_prob = np.vectorize(decimalize)(dice_prob)

    # Sets up the table to track the Viterbi algorithm
    hidden_states_ct = len(initial_prob)
    seq_ct = len(seq)
    finished = list()

    # Backward table used for the Viterbi algorithm
    back_table = [[0]*seq_ct for i in range(hidden_states_ct)]

    # Set up the starting values for the Viterbi table
    for i in range(hidden_states_ct):
        start_prob = initial_prob[i] * dice_prob[seq[0]][i]
        temp = Node(i, start_prob)
        back_table[i][0] = temp
        
    # Iterate through the sequence to set up the rest of the elements in the table.
    # Reaches every individual cell to calculate the probability based on the 2 previous cells.
    for i in range(1,seq_ct):
        observed_val = seq[i]
        
        for j in range(hidden_states_ct):
            intersecting_values = list()

            # Find the 2 last cells before the current cell to calculate the next probability
            for k in range(hidden_states_ct):
                prev_cell = back_table[k][i-1]
                prev_prob = prev_cell.p
                new_prob = prev_prob * transition_prob[k][j] * dice_prob[observed_val][j]
                
                # Create a temporary cell which points to the previous cell
                temp_cell = Node(j, new_prob, prev_cell)
                intersecting_values.append(temp_cell)
            
            # Determine the max of the values and append add it to the current cell location
            back_table[j][i] = max(intersecting_values)

    # Finds the value with the highest probability 
    item = max(back_table[0][-1],back_table[1][-1])

    # Follows the pointer all the way until the first element of the list is reached.
    # Assigns HMM values according to the HMM list given.
    while item:
        finished.append(item.v)
        item = item.point

    # Reverses string and returns it    
    return finished[::-1]
