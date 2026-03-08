link: https://quantum-journal.org/papers/q-2022-05-09-709/pdf/

---
tags: [parallelism, framework, c++, High Performance, Tasks, Random Circuits Benchmark]

---

## What they did?

They created a Framework that builds a dependency graph for tensor-network-based quantum-circuits, being used to execute efficiently the simulation on heterogeneous nodes using Taskflow.

They pointed 3 main advantages:
1. decomposition of tensor-network contraction into tasks that can be run in parallel
2. shared computation between contractions (compute once and use for multiple slices)
3. use of heterogeneous nodes (GPU+CPU)

The program takes a file with pre calculated path and the raw tensor network. With it, the algorithm calculates the dependency graph and map it into CPU+GPU using TaskFlow scheduler.

The goal is to use this software to find the best combination of contractions and slices that minimizes the overall amount of FLOP (floating point calculations).

They achieved a decreased the amount of GFLOPS by 47% using shared work and a reduction in time through Intel skylanes runs from 4.99 to 2.11s by using shared work. 

The memory usage increases when applying shared work, since the intermediate tensors are hold on memory. Deleting them when they are not required anymore is the best trade-off.

GPUs have better times of contraction.

---
## ideas

- Tensor networks are the fastest algorithms for quantum advantage experiments using Random quantum circuits (RQCs)
- Tensor networks was still slow compared to Sycamore since their simulator was not using all the horsepower provided by their machines
- One solution for that would be creating asynchronous tasks and run them in parallel
- Most common methods for contraction are tree decomposition and graph partitioning
- A quantum circuit can be seem as connections between many different tensors of multiple rankings (always powers of 2)
- must optimize the contraction order
- Now days, people use the CoTenGra to calculate the paths
- must maximize shared work
- shared work can be saved on disk


---

## Complexity
* Contraction of arbitrary structured tensor network is at least P-Hard
* Determination of optimal path is NP-Hard


![[overall-algorithm.png|697]]
In this case, each index in [a,b,c,d,e,f] is ranging from [0,1], so it passes through each combination and evaluates. D is the indexes dimensions, being in the case D=2, since the variables range [0,1]. In total we have D⁶ combinations, being overall complexity of O(D⁶). D in this case represents the numbering space (local).

![[contractions.png]]
Doing that part by part, we evaluate each index separate and the overall complexity is tied to the rank of the maximum tensor rank in each step.  So, using the contractions, we can decreased the complexity to O(D⁴)
![[slicing.png]]
Addressing the contractions for parallel computing, we could fix values for different indexes (in this case e), and calculate in parallel the contractions and then rejoin them at the end.