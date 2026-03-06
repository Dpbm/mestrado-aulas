## Notes

- Tensor networks for simulation are costly.
- MPS has less cost than TN but it's an aproximated version of TN.
- TN always return the exact output.
- Can hold more qubits than Statevec.
- Are a different way to see a quantum circuit.

---

- Quantum Circuits are types of tensor networs.
- Qubits(lines) are contractions and gates(boxes) are the tensors.
- Efficient for circuits with local gates, and those which are sparse.

---

- Statevec is efficent for computying, but grows exponentially in space.

---

- TN represents a circuit as a series of tensor contractions.
- It's trade between space for compute.
- Tensor contraction can become expensive.
- The main goal is to efficiently contract these tensors.

### cuTensorNet algorithmic description

- Is divided in Pathfinder and Execution.
- The pathfinder finds the optimal contraction path.
- The Execution executes the circuit itself.
- The larger the network is, more computational power.

#### Pathfinder

Uses 3 components

1. Simplification: It tries to find parts that can be joined together in a single tensor.
2. Path Computation: Based on graph-partitioning. A recursive approach to split the graph into smaller pieces that form a contraction path.
3. Hyper-optimizer: Loops over the path computation module aiming to keep track the best path and parameters.

The optimizer can also do some reconfigurations to moves the path towards an optimal state. This happens by selecting a few small branches and
applying contractions to them.

The path computation excludes some tensors and also unroll some to exclude the max it can.

### Tensors

- Are a generalization of scalars, vectors, matrices, etc.
- The rank is the number of dimensions. scalars=0, vectors=1, matrices=2

## References

- [https://pennylane.ai/qml/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks](https://pennylane.ai/qml/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks)
- [https://developer.nvidia.com/blog/scaling-quantum-circuit-simulation-with-cutensornet/](https://developer.nvidia.com/blog/scaling-quantum-circuit-simulation-with-cutensornet/)
- [https://quimb.readthedocs.io/en/main/examples/ex_tn_circuit_sample_explore.html](https://quimb.readthedocs.io/en/main/examples/ex_tn_circuit_sample_explore.html)
- [https://www.reddit.com/r/QuantumComputing/comments/alv99j/open_source_simulation_of_quantum_computers_with/](https://www.reddit.com/r/QuantumComputing/comments/alv99j/open_source_simulation_of_quantum_computers_with/)
- [https://aws.amazon.com/about-aws/whats-new/2020/12/amazon-braket-tensor-network-simulator-supports-50-qubit-quantum-circuits/](https://aws.amazon.com/about-aws/whats-new/2020/12/amazon-braket-tensor-network-simulator-supports-50-qubit-quantum-circuits/)
- [https://quantumcomputing.stackexchange.com/questions/30401/how-to-convert-a-quantum-circuit-into-a-tensor-network](https://quantumcomputing.stackexchange.com/questions/30401/how-to-convert-a-quantum-circuit-into-a-tensor-network)
- [https://www.quera.com/glossary/tensor-networks](https://www.quera.com/glossary/tensor-networks)
- [https://quantumcomputing.stackexchange.com/questions/4104/what-can-tensor-networks-mean-for-quantum-computing](https://quantumcomputing.stackexchange.com/questions/4104/what-can-tensor-networks-mean-for-quantum-computing)
- [https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
- [https://www.azoquantum.com/Article.aspx?ArticleID=420](https://www.azoquantum.com/Article.aspx?ArticleID=420)
