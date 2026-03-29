link: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.838601/full#s2

---
tags: [Framework, C++, HPC, TN]

---
## What they did?
* a library for manipulating tensor networks in c++
* For general tensor networks with arbitrary structure
* can be used for HPC (GPUs etc)

---
## ideas
* computation and storage is only done when required
* tensor can be seem as a multidimensional array
* the opposite of contraction is decomposition
* mode of a tensor -> tensor dimension or index
* regular contraction is done in between two dimensions
* hyper-contraction is when there're more than 2 

- A tensor network expansion can be constructed
by gradually appending individual tensor networks with their
respective complex coefficients. Numerical evaluation of a tensor
network expansion results in computing the output tensor of
each individual tensor network component, followed by the
accumulation of all computed output tensors which have the
same shape.

- A tensor network operator is a
linear combination of tensor networks in which additionally
the dimensions of the output tensor in each tensor network are
individually assigned to either the ket or the bra tensor spaces

* operations are asynchronous
* use visitor pattern to visite nodes in the DAG
* the DAG executor looks for dependecy free nodes

- The default
GPU tensor contraction algorithm is based on the matrix-
matrix multiplication (e.g., via cuBLAS) accompanied by an
optimized tensor transpose algorithm