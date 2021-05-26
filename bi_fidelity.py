###############################################################################
# Bi-fidelity Approximation                                                   #
# Perform an ID of the input LF matrix and return the required HF samples     #
###############################################################################

import numpy  as np                       
import matplotlib.pyplot as plt
import scipy.linalg.interpolative as sli

# load LF data matrix
file = open('data/LF.csv')
phi_lf = np.loadtxt(file, delimiter=',')

# apply the ID decomposition to the low-fidelity data
A_lf = phi_lf

eps_lf = 1.0E-6 #0.001
k_lf, idx_lf, proj_lf = sli.interp_decomp(A_lf, eps_lf)

print("rank (LF)= ",k_lf)
print("idx (LF)= ",idx_lf.shape)
print("proj (LF)= ",proj_lf.shape)

B_lf = sli.reconstruct_skel_matrix(A_lf, k_lf, idx_lf)
P_lf = sli.reconstruct_interp_matrix(idx_lf, proj_lf)
C_lf = sli.reconstruct_matrix_from_id(B_lf, idx_lf, proj_lf)

print("B_lf = ",B_lf.shape)
print("P_lf = ",P_lf.shape)
print("C_lf = ",C_lf.shape)

print("Frob Error (LF)= ",np.linalg.norm(A_lf-C_lf))
diff_lf = sli.estimate_spectral_norm_diff(A_lf, C_lf)
print("Spectral Norm of Difference (LF)= ",diff_lf)

# plot "selected" HF values
print(idx_lf[:k_lf])
plt.plot(phi_lf[:,idx_lf[:k_lf]])
plt.show()
