link: https://arxiv.org/abs/2303.11735

---
tags: [classification, summary]

---

## Classical TN
-  A generalization of vectors, matrices, scalars, etc.
- map to tangent and cotangent spaces.
- can have regular or dual indices depending if it's in tangent or cotangent space.
- the rank is determined by the amount of free indexes.
- Free regular indices can be contracted with free dual indexes by summing them over its dimensions.
- can be represented using the penrose diagram.
- allows to apply local operations individually without evaluating the whole tensor
- tensors can be joined together by contracting on connected indices or decomposed into connected tensors.
- SVD (singular value decomposition) is the most common approach, but we could also use either Polar or Tucker decomposition. They allow you to decompose a tensor into smaller ones.
- refinements can be made by applying symmetries to the TN
## TNs for quantum computing simulation
- are intended to reduce the computational cost of storing and evaluating lowly entangled multi-particle quantum states.
-  have been proposed to parallelize quantum simulations by cutting the system into several weakly entan-gled pieces and approximating the state of all but one piece by TNs
- Every kind can have many different layouts
- the layout determines the maximum entanglement or internal relation
- must choose the correct layout for an individual circuit to avoid using excessive resources, scalling exponentially with the system size


### Projected Entangled Pair States (PEPS)
- Grid Layout
- Usually used for low dimensions
-  PEPS nodes are constructed as composite objects consisting of coupled internal spins. Each spin connects to a neighboring site via an edge and at each node the constituent spins are entangled and truncated, thus the name PEPS.
- Entanglement size is linear
- costly to be contracted
![[PEPS.png]]

### Matrix Product States (MPS)
- Grid layout
- PEPS with 1 dimension
- also called tensor trains
- entanglement size is constant
- can be contracted efficiently

![[MPS.png]]

### Tree Tensor Networ (TTN)
* Hierarchical layout
* single node on top, each node is connected to a parent (except the root)
* Also called hierarchical Tucker decomposition
*  able to catch both local entanglement and long range entanglement between groups of nodes, but not long range entanglement between individual tensor nodes
* entanglement depends on the amount of nodes
* can be contracted efficiently
![[TTN.png]]
### Multi Scale Entanglement Renormalization Ansatz (MERA) 
* isometric TTN
* better entropy scaling
* enhance the hierarchy with layers of unitary nodes connecting neighboring branches,  so called disentanglers reduce entanglement passed on to the next level 
* has a higher computational cost than other layouts due to the loops, but it can capture symmetry and far higher entanglement while still being effi-ciently storable
* entanglement depends on the amount of nodes
* costly to be contracted
![[MERA.png]]


## Optimizations
- can be both by means of reducing the dimensions or reducing a cost function
-  Renormalization methods: They exploit the locality of operators to optimize the TN site by site. Density matrix renormalization group (DMRG), the first method of this kind, was developed to optimize spin chain Hamiltonians efficiently
- Global Gradient Methods:  standard optimization techniques that also apply to TNs renormalization methods have not been established yet for QTNs. Currently, stochastic gradient approximation methods are employed in QML to circumvent the need for costly calculations of total gradients in high dimensional parameter spaces
- Geometric methods: make use of the network’s underlying tensorial geometry. Tools from differential geometry can be used for analyzing the TN on the space of entanglement patterns and optimizing on loss manifolds. This kind of optimization performs well on high dimensional parameter spaces, especially in combination with stochastic gradient descent and auto-differentiation on individual nodes or whole layers