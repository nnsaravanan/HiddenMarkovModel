import numpy as np
from Helpful import Node,normalize,decimalize

def forwards_backwards(initial_prob, transition_prob, dice_prob, seq):
    
    """
    This function performs the Forward-Backward algorithm on a Hidden Markov Model to estimate the most 
    likely sequence of states given a sequence of observations.

    Inputs:

        initial_prob (list): A list of the initial probabilities of each hidden state.

        transition_prob (2D list): A square matrix representing the transition probabilities between hidden states.

        dice_prob (2D list): An emmission matrix that has a shape of unique emissions * hidden states, representing the probability 
        of each unique emission visible at a hidden state.

        seq (list): A list of the observed characters.

    Outputs:

        final (list): A list representing the most likely sequence of HMM states corresponding to the observed characters in seq.

    """

    # creates precise decimal values for small decimals in computation 
    initial_prob = np.vectorize(decimalize)(initial_prob)
    transition_prob = np.vectorize(decimalize)(transition_prob)
    dice_prob = np.vectorize(decimalize)(dice_prob)

    # set up the tables to track the forward-backwards algorithm
    hidden_states_ct = len(initial_prob)
    seq_ct = len(seq)
    finished = list()

    # Initialize the forward and backward tables
    forward_table = [[0]*seq_ct for i in range(hidden_states_ct)]
    backward_table = [[0]*seq_ct for i in range(hidden_states_ct)]

    # sets the initial values of the backwards and forwards table
    for i in range(hidden_states_ct):

        # Set the starting probability values for all the tables according to HMM states
        start_prob_f = initial_prob[i] * dice_prob[seq[0]][i]
        temp_f = Node(i,start_prob_f)

        forward_table[i][0] = temp_f

        # Sets the starting probabilities to 1 according to the backwards algorithm
        temp_b = Node(i,1)
        backward_table[i][-1] = temp_b


    # forwards algorithm which iterates forwards on the sequence
    for i in range(1,seq_ct):
        observed_val_f = seq[i]

        for j in range(hidden_states_ct):

            # Calculate the probability of reaching the current state
            sum_prob_f = 0
            for k in range(hidden_states_ct):
                prev_cell_f = forward_table[k][i-1]
                prev_prob_f = prev_cell_f.p

                sum_prob_f += prev_prob_f * transition_prob[k][j] * dice_prob[observed_val_f][j]

            # Create a node for the current state
            temp_cell_f = Node(j,sum_prob_f)
            forward_table[j][i] = temp_cell_f


    # backwards algorithm which iterates backwards on the sequence
    for i in range(seq_ct-2,-1,-1):
        observed_val_b = seq[i]

        for j in range(hidden_states_ct):

            # Calculate the probability of reaching the current state
            sum_prob_b = 0
            for k in range(hidden_states_ct):
                prev_cell_b = backward_table[k][i+1]
                prev_prob_b = prev_cell_b.p

                sum_prob_b += prev_prob_b * transition_prob[k][j] * dice_prob[observed_val_b][j]
            
            # Create a node for the current state
            temp_cell_b = Node(j,sum_prob_b)
            backward_table[j][i] = temp_cell_b


    # Normalize data for comparisons
    for i in range(hidden_states_ct):
        normalize(forward_table[i])
        normalize(backward_table[i])

    # Create the final value based on the forward-backwards comparisons
    for i in range(seq_ct):

        possible_values = list()

        # Calculate the total probability for each state at the current position
        for j in range(hidden_states_ct):
            total_prob = forward_table[j][i].p * backward_table[j][i].p
            temp = Node(j,total_prob)
            possible_values.append(temp)

        # Choose the state with the maximum probability
        finished.append(max(possible_values).v)

    return finished


            