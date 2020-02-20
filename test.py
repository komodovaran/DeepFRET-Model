import numpy as np
k_states = 3
trans_prob = 0.03

trans_mat = np.empty([k_states, k_states])
trans_mat.fill(trans_prob)
np.fill_diagonal(trans_mat, 1 - trans_prob)

print(trans_mat)

# Make sure that each row/column sums to exactly 1
if trans_prob != 0:
    stay_prob = 1 - trans_prob
    remaining_prob = 1 - trans_mat.sum(axis=0)
    trans_mat[trans_mat == stay_prob] += remaining_prob

print(trans_mat)