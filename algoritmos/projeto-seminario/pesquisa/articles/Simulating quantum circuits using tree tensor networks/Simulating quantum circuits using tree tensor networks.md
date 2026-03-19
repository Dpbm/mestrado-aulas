link: https://quantum-journal.org/papers/q-2023-03-30-964/pdf/

---
tags: [TTN, MPS, comparison, algorithm, simulation]

---
## What they did?

- They compared TTN to MPS
- They created a novel method for TTN circuit simulation
- They also created a method to generate a from an initial state and fixed a bound dimension to restrict the circuit growth 

---
## ideas

- TTN can hold distant connections
- Some circuits can't be represented as a TTN
    - QFT is lattice, so can't be used in a TTN

- The algorithm has 2 phases
    1. Expressing the initial quantum state as a TTN by mapping logical qubits to leaf nodes and identifying an advantageous rooted tree graph
    2. Sequentially applying quantum circuit gates to the tree while preserving its graph structure, interleaved with re-orthonormalization of the tree tensor network.

    To ensure that the tree tensor network can be simulated on a classical computer, we require that each edge dimension is bounded by a chosen constant Dmax . Naturally, this restricts the set of quantum circuits that can be simulated by our approach
---

## Complexity

- TTN has polynomial cost for contraction
- O(log(n)) to traverse in contrast of O(N) in MPS
- the cost of updating a tree might be expensive, but these kind of expensive update are rare, so MPS still worse in this case