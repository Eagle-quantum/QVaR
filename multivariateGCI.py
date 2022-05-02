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
from qiskit.circuit.library.generalized_gates import mcmt
from qiskit.circuit.library import WeightedAdder


class MultivariateGCI(QuantumCircuit):
    """Attempt for a multivariate Gaussian Conditional 
    Independence Model for Credit Risk.
    """


    def __init__(
        self,
        n_normal: int,
        normal_max_value: float,
        p_zeros: Union[List[float], np.ndarray],
        rhos: Union[List[float], np.ndarray],
        alphas_list: List[float],
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
        num_qubits = n_normal*len(sectors) + len(p_zeros)

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
        for rho, p_zero, alphas in zip(rhos, p_zeros, alphas_list):
            psi = F_inv(p_zero) / np.sqrt(1 - rho) 
            # compute slope / offset
            slope_list = []
            case=[]
            for i in range(len(sectors)):
                slope = -alphas[i] / np.sqrt(1 - rho) # -np.sqrt(rho)*alphas[i] / np.sqrt(1 - rho)
                slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
                slope_list.append(slope)
            
            offset = 2 * np.arcsin(np.sqrt(F(psi)))
            #print(offset)

            # adjust for integer to normal range mapping
            for i in range(len(sectors)):
                offset += slope_list[i] * (-normal_max_value)
                slope_list[i] *= 2 * normal_max_value / (2 ** n_normal - 1)

            offsets += [offset]
            slopes += [slope_list]
            
        # create normal distributions        
        normal_distributions = []
        for i in range(len(sectors)):
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
            for i in range(len(sectors)):
                if i == 0:
                    lry = LinearPauliRotations(n_normal, slope[i], offset)
                else:
                    lry = LinearPauliRotations(n_normal, slope[i], 0) 
                qubits = list(range(i*n_normal,(i+1)*n_normal)) + [n_normal*len(sectors) + k]
            
                inner.append(lry.to_gate(), qubits)

        super().__init__(num_qubits, name="P(X)")
        self.append(inner.to_gate(), inner.qubits)



class MultivariateGCI_woerner(QuantumCircuit):
    """The Gaussian Conditional Independence Model for Credit Risk.
    Reference: https://arxiv.org/abs/1412.1183
    Dependency between individual risk variables and latent variable is approximated linearly.
    """

    def __init__(
        self,
        n_normal: int,
        normal_max_value: float,
        p_zeros: Union[List[float], np.ndarray],
        rhos: Union[List[float], np.ndarray],
        alphas: List[float],
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
        num_qubits = n_normal*len(alphas) + len(p_zeros)

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
        for rho, p_zero in zip(rhos, p_zeros):
            psi = F_inv(p_zero) / np.sqrt(1 - rho)

            # compute slope / offset
            slope = -1 / np.sqrt(1 - rho) # -np.sqrt(rho) / np.sqrt(1 - rho)
            offset = 2 * np.arcsin(np.sqrt(F(psi)))
            slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
            

            # adjust for integer to normal range mapping
            offset += slope * (-normal_max_value) * len(alphas)
            slope *= 2 * normal_max_value / (2 ** n_normal - 1)

            offsets += [offset]
            slopes += [slope]

        # create multivariate normal distribution
        normal_distribution = NormalDistribution(
            list(np.zeros(len(alphas), dtype=int)+n_z),
            list(np.zeros(len(alphas), dtype=int)),
            np.diag(np.array(alphas)**2), 
            bounds=[(-normal_max_value, normal_max_value) for i in range(len(alphas))],
        )

        # create WeightedAdder
        weights = []
        for n in [n_normal]*len(alphas):
            for i in range(n):
                weights += [2**i]
        Y_risk = WeightedAdder(n_normal*len(alphas), weights) 
        #print(Y_risk.num_sum_qubits)

        # build circuit
        non_state_qubits = Y_risk.num_sum_qubits+Y_risk.num_carry_qubits+Y_risk.num_control_qubits
        inner = QuantumCircuit(num_qubits+non_state_qubits, name="P(X)")
        inner.append(normal_distribution.to_gate(), list(range(n_normal*len(alphas))))
        inner.append(Y_risk.to_gate(), list(range(n_normal*len(alphas)+non_state_qubits)))

        for k, (slope, offset) in enumerate(zip(slopes, offsets)):
            lry = LinearPauliRotations(Y_risk.num_sum_qubits, slope, offset)
            qubits = list(range(Y_risk.num_state_qubits, Y_risk.num_state_qubits+Y_risk.num_sum_qubits)) + [Y_risk.num_qubits + k]
            inner.append(lry.to_gate(), qubits)

        super().__init__(num_qubits+non_state_qubits, name="P(X)")
        self.append(inner.to_gate(), inner.qubits)