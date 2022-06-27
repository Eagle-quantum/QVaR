import numpy as np
from multivariateGCI_mr import MultivariateGCI_mr
from scipy.stats import norm

from qiskit.circuit.library import LinearAmplitudeFunction


def rotation_mapping(p_zeros, F_values, rhos, n_z, z_max, n_factors, decimal_number, k):
    b = ('{0:0%sb}' % n_factors).format(decimal_number)
    realizations = []
    for i in range(n_factors):
        bin = b[n_z*i:n_z*(i+1)]
        realizations.append(int(bin, 2) * 2 * z_max / (2 ** n_z - 1) - z_max)
        # print(realizations[i])
    
    p = norm.cdf(
        (
        norm.ppf(p_zeros[k]) - np.sum([F*realization for F, realization in zip(F_values[k], realizations)])
        ) / np.sqrt(1-rhos[k])
    ) 
    

    return p

def mapping(decimal_number, K):
    b = ('{0:0%sb}' % K).format(decimal_number)
    # print(b)
    losses = [loss for i, loss in enumerate(lgd[::-1]) if b[i]=='1']
    # print(losses)
    total_loss = sum(losses)
    return total_loss

def find_breakpoint(x_eval, K):
    for el in range(0,2**K):
        if mapping(el) <= x_eval:
            if mapping(el+1) >= x_eval:
                return el
    return 0

def get_loss_objective(K, lgd):
    # define linear objective function for expected loss
    breakpoints = list(range(0,2**K))
    slopes = [0]*len(breakpoints)
    offsets = [mapping(el) for el in breakpoints]
    f_min = 0
    f_max = sum(lgd)
    c_approx = 0.01

    objective_e_loss = LinearAmplitudeFunction(
        K,
        slope=slopes, 
        offset=offsets, 
        # max value that can be reached by the qubit register (will not always be reached)
        domain=(0, 2**K-1),  
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints
    )

    return objective_e_loss