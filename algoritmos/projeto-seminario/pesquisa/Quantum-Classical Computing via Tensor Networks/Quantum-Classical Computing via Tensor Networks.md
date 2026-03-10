link: https://arxiv.org/pdf/2410.15080

---
tags: [Framework, Circuit Knitting, speedup, Tensor Network, GPU]

---
## What they did?

They created a framework for circuit knitting leveraging Tensor networks and GPUs.

They achieved a better Post Processing overhead 10⁴x and overall 20.7x speedup.

---
## ideas

* enable the efficient storage and manipulation of high-dimensional data, making them a powerful tool for quantum circuit simulation
* It uses an hybrid of quantum and classical tensors
* The runtime of the framework, takes this hybrid TN and then executes part in QPUs and other parts in GPUs
* Each cut is weighted by a coefficient that's stored into a classical tensor and the sub circuits are transformed into two Quantum tensors.Then, it evaluates the quantum circuits extracting the expectation values and their results are written into another CT. Finally, these CTs are contracted into a single value.
* QTs and CTs can be fused and simplified to raise a better network to be evaluated efficiently  
