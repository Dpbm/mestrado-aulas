link: https://arxiv.org/abs/2007.14822

---
tags: [TN, Framework, Julia, MPS, MPO]

---
## What they did?

- a Framework in Julia for tensor networks manipulation

---
## ideas

- When all of the tensors in the
network are low-order, a tensor network can make it efficient to perform important operations
such as summing two high-order tensors or computing their inner product. These operations
can remain efficient whether the high-order tensor represented implicitly by the network has
hundreds, thousands, or even an infinite number of indices


![[tn.png]]

- Space can be saved by assessing the sparsity and the structure of tensors
- You can define the network indexes without allocating memory for the tensors (to make it efficient) 
![[1.png]]
![[2.png]]
![[3.png]]
![[4.png]]
![[5.png]]

- The DMRG algorithm computes low-energy states of
quantum systems, or in mathematical terms, dominant eigenvectors of very large Hermitian
linear operators.
The main inputs to a DMRG calculation is a Hamiltonian Ĥ and an initial guess Ψ0 (i)
 for its
ground state Ψ0 .

- For contracting two dense tensors, NDTensors currently uses a strategy of permuting and
reshaping the tensors into matrices, so that the contraction maps to a matrix multiplication8 .
The motivation behind this strategy is that BLAS libraries such as Intel MKL offer such
high performance that the extra overhead of permuting the tensors is worthwhile

- The case of block sparse tensor contraction reduces to doing a set of smaller, dense tensor
contractions on various pairs of blocks from the tensors being contracted 9 . Thus it is built
on top of the dense contraction layer of NDTensors, but also offers an excellent opportunity
to exploit parallelism, since contraction of the blocks can be done independently, although
one does have to handle cases where multiple block pairs contribute to the same block of the
resulting tensor
