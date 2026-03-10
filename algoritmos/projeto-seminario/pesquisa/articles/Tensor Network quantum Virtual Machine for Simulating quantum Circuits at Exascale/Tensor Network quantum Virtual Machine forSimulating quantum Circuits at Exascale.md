link: https://dl.acm.org/doi/full/10.1145/3547334

---
tags: [Simulator, Simulator Backend, HPC,  framework, MPS, Tensor Network]

---
## What they did?

They have created a enhanced version of TNQVM simulator levaraging heterogeneous HPC by using ExaTN framework for XACC quantum framework.

They aimed to make it flexible and suited for any kind of circuit and work on different classical platforms.

---
## ideas
* ExaTN uses MPI to spread tasks
* They use tensor slicing
* They use pairwise-contractions
* Contractions are appended into a DAG and executed via ExaTN asynchronously 
* ExaTN provides and API to optimize the networks by applying Tensor Network Reconstruction - approximate a network to another (with a simpler structure, usually)
* It can be exact or approximated using MPS
* In general it visits each node of XACC IR, then construct, evaluate and post-process the ExaTN Tensor network
* After creating the TN and submitting it to ExaTN, it tries to find the optimal contraction path minimizing FLOP or any other metrics
* to minimize FLOP it uses recursive graph partitioning(METIS) and heuristics
* one can Use Bayesian Optimization
* ExaTN is often integrated with different machinery for path finding (cuQuantum and CoTenGra)
* number of the edges in the networks is equals to the number of qubits
* For large circuits we can keep the wave function in memory, so we need auxiliary functions to extract observable values
	* For single qubit amplitudes it can calculate he projection of a given state in the final circuit state. If it grows too large for the underlying memory, it can be split into MPI processes.
	* For Halmiltonian evaluation
		* It can append the hamiltonian as tensors at the end and then append the conjugate of the circuit aftwerwards (better for circuits with many qubits)
		* We could also use the whole wave function but split into different nodes (better for circuit with less qubits by deeper ones)
* Light-cone simplification is a technique that simplifies the network by elimitating tensors that are contracted with their conjugates

* MPS is performed via standard sequential contraction and decomposition steps
* single qubit gates are absorbed into the MPS tensors directly
* two qubit gates are computed by:
	* contract the 2 MPS tensors with the rank-4 tensor
	* decompose the resulting tensor back into MPS tensors
	* truncate the dimension
	* update MPS
* MPS tensors are spread across processes 

* It also has density matrix simulation for noisy
*  locally purified matrix product state (LP-MPS) -> a way to factorize the density matrix into MPS
