from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.circuit.library import WeightedAdder
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.test.mock import FakeToronto
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem


def compute_depth(n_z, K, lgd, u, optimization_level = 0, k = 1):
    
    agg = WeightedAdder(n_z + K, [0] * n_z + lgd)

    # define linear objective function
    breakpoints = [0]
    slopes = [1]
    offsets = [0]
    f_min = 0
    f_max = sum(lgd)
    c_approx = 0.25

    objective = LinearAmplitudeFunction(
        agg.num_sum_qubits,
        slope=slopes,
        offset=offsets,
        # max value that can be reached by the qubit register (will not always be reached)
        domain=(0, 2**agg.num_sum_qubits - 1),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )

    # define the registers for convenience and readability
    qr_state = QuantumRegister(u.num_qubits, 'state')
    qr_sum = QuantumRegister(agg.num_sum_qubits, "sum")
    qr_carry = QuantumRegister(agg.num_carry_qubits, "carry")
    qr_control = QuantumRegister(agg.num_control_qubits, "control")
    qr_obj = QuantumRegister(1, 'objective')
    # ar = QuantumRegister(objective.num_ancillas, "work")  # additional qubits

    # define the circuit
    state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, qr_control, name="A")

    # load the random variable
    state_preparation.append(u.to_gate(), qr_state)

    # aggregate
    state_preparation.append(agg.to_gate(), qr_state[:] + qr_sum[:] + qr_carry[:] + qr_control[:])

    # linear objective function
    state_preparation.append(objective.to_gate(), qr_sum[:] + qr_obj[:])

    # uncompute aggregation
    state_preparation.append(agg.to_gate().inverse(), qr_state[:] + qr_sum[:] + qr_carry[:] + qr_control[:])

    backend = FakeToronto()

    epsilon = 0.01
    alpha = 0.05

    n_shots = 2048

    qi_ideal = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=n_shots, 
                            optimization_level=optimization_level, seed_transpiler=42)

    iae = IterativeAmplitudeEstimation(alpha=alpha, epsilon_target=epsilon, quantum_instance=qi_ideal)
    problem = EstimationProblem(state_preparation=state_preparation,
                                objective_qubits=[u.num_qubits],
                                post_processing=objective.post_processing)
    circuit = iae.construct_circuit(problem, k= k)

    tr_ae_circuit = transpile(circuit, optimization_level=optimization_level, backend=backend, seed_transpiler=42)


    return tr_ae_circuit.depth(), tr_ae_circuit.size()