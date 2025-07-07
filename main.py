"""LiH VQE simulation using Cirq and OpenFermion.

This module replaces the original Qiskit based implementation with a purely
OpenFermion / Cirq version.  Two energy estimators are compared:

``QWC``  - measuring qubit-wise commuting Pauli strings and building the
            energy from their expectation values.
``Z-sampling`` - sampling the prepared state in the computational basis and
            using only the diagonal part of the Hamiltonian.

The script can be executed directly to plot the convergence of both methods
for the LiH molecule at a chosen bond length.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sympy
import cirq
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.circuits import (
    uccsd_singlet_generator,
    uccsd_singlet_paramsize,
)


def setup_lih_hamiltonian(bond_length: float) -> Tuple[cirq.PauliSum, int, Dict[str, float]]:
    """Construct the LiH Hamiltonian as a ``cirq.PauliSum``.

    Parameters
    ----------
    bond_length:
        Distance between Li and H atoms in Angstrom.

    Returns
    -------
    qubit_op:
        The Hamiltonian mapped to qubits via the Jordan--Wigner transform.
    n_qubits:
        Number of qubits required for the simulation.
    h_diag:
        Dictionary mapping computational basis bitstrings to diagonal energies.
        This is used for the Z-basis sampling energy estimator.
    """

    # Run PySCF to obtain molecular integrals and build the fermionic Hamiltonian
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    molecule = run_pyscf(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )

    fermion_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(fermion_hamiltonian)

    # Map to qubits using the Jordan--Wigner transform
    qubit_op_of = jordan_wigner(fermion_op)
    qubit_op_of.compress()

    # Convert to Cirq PauliSum
    qubit_op = cirq.PauliSum()
    for term, coeff in qubit_op_of.terms.items():
        pauli_map = {}
        for idx, pauli in term:
            if pauli == "X":
                pauli_map[cirq.LineQubit(idx)] = cirq.X
            elif pauli == "Y":
                pauli_map[cirq.LineQubit(idx)] = cirq.Y
            elif pauli == "Z":
                pauli_map[cirq.LineQubit(idx)] = cirq.Z
        if pauli_map:
            pauli_string = cirq.PauliString(pauli_map)
        else:  # identity term
            pauli_string = cirq.PauliString()
        qubit_op += coeff * pauli_string

    n_qubits = molecule.n_qubits

    # Pre-compute diagonal energies for all computational basis states
    qubits = cirq.LineQubit.range(n_qubits)
    h_diag: Dict[str, float] = {}
    for bits in itertools.product("01", repeat=n_qubits):
        bitstring = "".join(bits)
        energy = 0.0
        for term, coeff in qubit_op_of.terms.items():
            if any(p not in ("Z",) for _, p in term):
                continue
            parity = 1
            for idx, pauli in term:
                if pauli == "Z" and bitstring[n_qubits - 1 - idx] == "1":
                    parity *= -1
            energy += coeff.real * parity
        h_diag[bitstring] = energy

    return qubit_op, n_qubits, h_diag


def build_uccsd_ansatz(
    n_qubits: int, n_electrons: int
) -> Tuple[List[sympy.Symbol], cirq.Circuit]:
    """Return a parameterised UCCSD singlet ansatz as a Cirq circuit."""

    n_params = uccsd_singlet_paramsize(n_qubits, n_electrons)
    symbols = list(sympy.symbols(f"theta_0:{n_params}"))
    generator = uccsd_singlet_generator(symbols, n_qubits, n_electrons)
    qubit_gen = jordan_wigner(get_fermion_operator(generator))

    pauli_sum = cirq.PauliSum()
    for term, coeff in qubit_gen.terms.items():
        pauli_map = {}
        for idx, pauli in term:
            if pauli == "X":
                pauli_map[cirq.LineQubit(idx)] = cirq.X
            elif pauli == "Y":
                pauli_map[cirq.LineQubit(idx)] = cirq.Y
            elif pauli == "Z":
                pauli_map[cirq.LineQubit(idx)] = cirq.Z
        if pauli_map:
            pauli_string = cirq.PauliString(pauli_map)
        else:
            pauli_string = cirq.PauliString()
        pauli_sum += coeff * pauli_string

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(cirq.PauliSumExponential(pauli_sum)(*qubits))
    return symbols, circuit


def estimate_energy_qwc(results: Dict[str, np.ndarray], qubit_op: cirq.PauliSum) -> float:
    """Estimate the energy from measurement results of QWC terms."""

    energy = 0.0
    for pauli_string in qubit_op:
        key = pauli_string.__repr__()
        if key not in results:
            continue
        meas = results[key]
        parity = 1 - 2 * meas
        energy += pauli_string.coefficient.real * np.mean(parity)
    return energy


def estimate_energy_z_sampling(counts: Dict[str, int], h_diag: Dict[str, float]) -> float:
    """Return the Z-sampling energy estimate ``Σ_z p(z) * h_diag[z]``."""

    total = float(sum(counts.values()))
    if total == 0:
        return 0.0

    energy = 0.0
    for bitstring, freq in counts.items():
        energy += (freq / total) * h_diag.get(bitstring, 0.0)
    return energy


def spsa_optimize(
    objective,
    initial_point: np.ndarray,
    maxiter: int,
    callback,
) -> np.ndarray:
    """Simple implementation of the SPSA algorithm."""

    params = np.array(initial_point, dtype=float)

    a = 0.2
    c = 0.1
    alpha = 0.602
    gamma = 0.101

    for k in range(maxiter):
        ak = a / (k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        delta = 2 * np.random.randint(0, 2, size=params.size) - 1
        params_plus = params + ck * delta
        params_minus = params - ck * delta
        y_plus = objective(params_plus)
        y_minus = objective(params_minus)
        grad = (y_plus - y_minus) / (2.0 * ck) * delta
        params -= ak * grad
        energy = objective(params)
        callback(k, params, energy)

    return params


def vqe_convergence(bond_length: float, shots: int, maxiter: int) -> Tuple[List[float], List[float]]:
    """Run VQE with both estimators and return energy histories."""

    qubit_op, n_qubits, h_diag = setup_lih_hamiltonian(bond_length)
    n_electrons = 4  # LiH has four electrons in the minimal basis
    symbols, ansatz_template = build_uccsd_ansatz(n_qubits, n_electrons)
    qubits = cirq.LineQubit.range(n_qubits)
    simulator = cirq.Simulator()

    hist_qwc: List[float] = []
    hist_z: List[float] = []

    def objective_qwc(theta: np.ndarray) -> float:
        resolver = {s: t for s, t in zip(symbols, theta)}
        circuit = cirq.resolve_parameters(ansatz_template, resolver)
        results: Dict[str, np.ndarray] = {}
        for pauli_string in qubit_op:
            basis_ops = []
            for q, gate in pauli_string.items():
                if gate == cirq.X:
                    basis_ops.append(cirq.H(q))
                elif gate == cirq.Y:
                    basis_ops.append(cirq.rx(np.pi / 2).on(q))
            measurement_key = pauli_string.__repr__()
            meas_circuit = circuit + basis_ops + cirq.measure(*qubits, key=measurement_key)
            result = simulator.run(meas_circuit, repetitions=shots)
            results[measurement_key] = result.measurements[measurement_key][:, 0]
        return estimate_energy_qwc(results, qubit_op)

    def cb_qwc(iteration: int, params: np.ndarray, energy: float) -> None:
        hist_qwc.append(energy)

    initial_point = np.zeros(len(symbols))
    spsa_optimize(objective_qwc, initial_point, maxiter, cb_qwc)

    def objective_z(theta: np.ndarray) -> float:
        resolver = {s: t for s, t in zip(symbols, theta)}
        circuit = cirq.resolve_parameters(ansatz_template, resolver)
        circuit += cirq.measure(*qubits, key="z")
        result = simulator.run(circuit, repetitions=shots)
        bitstrings = ["".join(str(b) for b in bs) for bs in result.measurements["z"]]
        counts = dict(Counter(bitstrings))
        return estimate_energy_z_sampling(counts, h_diag)

    def cb_z(iteration: int, params: np.ndarray, energy: float) -> None:
        hist_z.append(energy)

    spsa_optimize(objective_z, initial_point, maxiter, cb_z)

    return hist_qwc, hist_z


def plot_convergence(hist_qwc: List[float], hist_z: List[float], hci_ref: float) -> None:
    """Plot the VQE convergence for the two estimators."""

    plt.plot(range(len(hist_qwc)), hist_qwc, label="QWC VQE")
    plt.plot(range(len(hist_z)), hist_z, label="Z-sampling VQE")
    plt.axhline(hci_ref, linestyle="--", label="HCI ref")
    plt.xlabel("SPSA Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
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

