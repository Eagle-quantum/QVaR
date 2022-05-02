# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multivariate Gaussian Conditional Independence Models for Credit Risk."""

from typing import List, Union
import numpy as np
from scipy.stats.distributions import norm

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearPauliRotations
from qiskit_finance.circuit.library import NormalDistribution

class MultivariateGCI_mr(QuantumCircuit):
    """Attempt for a multivariate Gaussian Conditional 
    Independence Model for Credit Risk.
    """


    def __init__(
        self,
        n_normal: int,
        normal_max_value: float,
        p_zeros: Union[List[float], np.ndarray],
        rhos: Union[List[float], np.ndarray],
        F_list: List[float],
    ) -> None:
        """
        Args:
            n_normal: Number of qubits to represent the latent normal random variable Z
            normal_max_value: Min/max value to truncate the latent normal random variable Z
            p_zeros: Standard default probabilities for each asset
            rhos: Sensitivities of default probability of assets with respect to latent variable Z
        """
        self.n_normal = n_normal
        self.normal_max_value = normal_max_value
        self.p_zeros = p_zeros
        self.rhos = rhos
        self.sectors = len(F_list[0])
        num_qubits = n_normal*self.sectors + len(p_zeros)

        # get normal (inverse) CDF and pdf (these names are from the paper, therefore ignore
        # pylint)
        def F(x):  # pylint: disable=invalid-name
            return norm.cdf(x)

        def F_inv(x):  # pylint: disable=invalid-name
            return norm.ppf(x)

        def f(x):  # pylint: disable=invalid-name
            return norm.pdf(x)

        # create linear rotations for conditional defaults
        slopes = []
        offsets = []
        for rho, p_zero, ef in zip(rhos, p_zeros, F_list):
            psi = F_inv(p_zero) / np.sqrt(1 - rho) 
            # compute slope / offset
            slope_list = []
            case=[]
            for i in range(self.sectors):
                slope = -ef[i] / np.sqrt(1 - rho) # -np.sqrt(rho)*ef[i] / np.sqrt(1 - rho)
                slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
                slope_list.append(slope)
            
            offset = 2 * np.arcsin(np.sqrt(F(psi)))
            #print(offset)

            # adjust for integer to normal range mapping
            for i in range(self.sectors):
                offset += slope_list[i] * (-normal_max_value)
                slope_list[i] *= 2 * normal_max_value / (2 ** n_normal - 1)

            offsets += [offset]
            slopes += [slope_list]
            
        # create normal distributions        
        normal_distributions = []
        for i in range(self.sectors):
            dist = NormalDistribution(
                        n_normal,
                        0,
                        1, 
                        bounds=(-normal_max_value, normal_max_value)
                    )
            normal_distributions.append(dist)

        # build circuit
        inner = QuantumCircuit(num_qubits, name="P(X)")
        #inner.append(normal_distribution.to_gate(), list(range(n_normal*2)))
        for i, el in enumerate(normal_distributions):
            inner.append(el.to_gate(), list(range(i*n_normal,(i+1)*n_normal)))
        #inner.draw()

        for k, (slope, offset) in enumerate(zip(slopes, offsets)):
            #lry = LinearPauliRotations(n_normal, slope, offset)
            for i in range(self.sectors):
                if i == 0:
                    lry = LinearPauliRotations(n_normal, slope[i], offset)
                else:
                    lry = LinearPauliRotations(n_normal, slope[i], 0) 
                qubits = list(range(i*n_normal,(i+1)*n_normal)) + [n_normal*self.sectors + k]
            
                inner.append(lry.to_gate(), qubits)

        super().__init__(num_qubits, name="P(X)")
        self.append(inner.to_gate(), inner.qubits)