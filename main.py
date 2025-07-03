import argparse
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock, UCCSD
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA


def setup_lih_hamiltonian(bond_length: float):
    """Build the LiH Hamiltonian and return the qubit operator, problem and converter."""
    driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {bond_length}", basis="sto3g")
    problem = ElectronicStructureProblem(driver)
    second_q_ops = problem.second_q_ops()
    electronic_hamiltonian = second_q_ops[0]
    converter = QubitConverter(JordanWignerMapper(), two_qubit_reduction=False)
    qubit_op = converter.convert(electronic_hamiltonian, num_particles=problem.num_particles)
    return qubit_op, problem, converter


def build_uccsd_ansatz(problem, converter):
    """Return a UCCSD ansatz prepared on the Hartree--Fock state."""
    num_particles = problem.num_particles
    num_spin_orbitals = problem.num_spin_orbitals
    hf_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter=converter)
    ansatz = UCCSD(
        qubit_converter=converter,
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initial_state=hf_state,
    )
    return ansatz


def estimate_energy_z_sampling(counts: Dict[str, int], qubit_op) -> float:
    """Compute the Z-sampling energy estimate from basis counts."""
    total = sum(counts.values())
    energy = 0.0
    for bitstring, freq in counts.items():
        p = freq / total
        e_z = 0.0
        for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
            label = pauli.to_label()
            if 'X' in label or 'Y' in label:
                continue
            parity = 1
            for bit, op_char in zip(reversed(bitstring), reversed(label)):
                if op_char == 'Z' and bit == '1':
                    parity *= -1
            e_z += coeff.real * parity
        energy += p * e_z
    return energy


def vqe_convergence(bond_length: float, shots: int, maxiter: int):
    """Run VQE with QWC grouping and Z-basis sampling, returning histories."""
    qubit_op, problem, converter = setup_lih_hamiltonian(bond_length)
    ansatz = build_uccsd_ansatz(problem, converter)

    backend = QasmSimulator()
    qi = QuantumInstance(backend, shots=shots)

    hist_qwc: List[float] = []
    hist_z: List[float] = []

    def cb_qwc(eval_count, params, energy, _stddev):
        hist_qwc.append(energy)

    optimizer = SPSA(maxiter=maxiter)
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=qi, callback=cb_qwc)
    _ = vqe.compute_minimum_eigenvalue(qubit_op)

    def objective_z(theta: np.ndarray) -> float:
        qc = ansatz.bind_parameters(theta)
        qc.measure_all()
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        return estimate_energy_z_sampling(counts, qubit_op)

    def cb_z(eval_count, params, energy, _):
        hist_z.append(energy)

    optimizer_z = SPSA(maxiter=maxiter)
    initial_point = np.zeros(ansatz.num_parameters)
    optimizer_z.optimize(
        num_vars=ansatz.num_parameters,
        objective_function=objective_z,
        initial_point=initial_point,
        callback=cb_z,
    )

    return hist_qwc, hist_z


def plot_convergence(hist_qwc: List[float], hist_z: List[float], hci_ref: float):
    """Plot the convergence histories and reference line."""
    plt.plot(range(len(hist_qwc)), hist_qwc, label="QWC VQE")
    plt.plot(range(len(hist_z)), hist_z, label="Z-sampling VQE")
    plt.axhline(hci_ref, linestyle="--", label="HCI ref")
    plt.xlabel("SPSA Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="LiH VQE: QWC vs Z-sampling")
    parser.add_argument("--bond_length", type=float, default=1.596)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--hci_ref", type=float, default=-7.8823)
    args = parser.parse_args()

    hist_qwc, hist_z = vqe_convergence(args.bond_length, args.shots, args.maxiter)
    plot_convergence(hist_qwc, hist_z, args.hci_ref)


if __name__ == "__main__":
    main()
