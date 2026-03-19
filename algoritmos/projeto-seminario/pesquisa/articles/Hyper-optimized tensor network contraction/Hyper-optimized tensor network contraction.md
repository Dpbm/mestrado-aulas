link: https://arxiv.org/pdf/2002.01935

---
tags: [Algorithms, optimal, greedy, supremacy, benchmarks]

---
## What they did?
* A framework for efficient network contraction 
* The networks they focused was those with non-standardized format
* Find of optimal contraction paths
* they achieved speedups that grows exponentially with the number of tensors
* They also created a hyper-optimization approach that tweaks its parameters during the contraction to find better ways to contract
* Near optimal performance
* Estimated 10000x performance on sycamore algorithm
* The graph must have no loops


---
## ideas

![[ideas1.png]]![[ideas2.png]]


* the circuit contraction can be seem as a vertex contraction in a graph G
	* Each contraction removes edges between pairs of tensors
	* Inner product is used for relative indexes
	* Outer product is taken for divergent indexes
* The vertex contraction sequence is denoted in a binary tree 
	![[tree1.png]]
	![[tree2.png]]
* In general, the method aims to minimize the vertex or edge congestion
* outer products are rare and most part of the time not beneficial

	Approches
	* exhaustive search: finds the optimal path. to be efficient one could use dynamic programming to build the tree only based on connected subgraphs
	* Line-Graph tree decomposition: is Upper-bounded by the width of the tree of the line graph -1.
		1. get the edges ordered 
		2. find the subgraph of G from the current edge ei
		3. Update G by contracting the tensors in the subgraph until it remains a single vertex. If more than one tensor one could use greedy algorithms or exhaustive search for the small subtree.
		4. repeat it for every ei
		* The most used algorithms for that are:
			1. QUICKBB - uses a depth-first 'branch and bound' search. Good for graphs with a modest amount of edges. 
			2. FlowCutter - top-down approach. Good for larger graphs.
			* They don't take into account weights, so it's good for quantum circuits.
	* Community detection: search for communities within the graph (vertexes that are densely connected internaly but sparse with its complement)
		* contracting communities first leads to a better path
		* an algorithm to detect these communities is Girvan and Newmann's
			* the result of this algorithm is also a contraction path
	* Agglomerative contraction trees: heuristically scores each possible pairwise contraction. Based on these scores, a pair of tensors can be chosen and contracted into a new vertex and the list of scores then updated with any new possible contractions 
		* Score function must be careful to choose scores
			![[score.png]]
			![[score2.png]]
	* Divisive Contraction Trees - A top Down divisive approach
		* split a node is a tree bipartitioning
		* single contraction one at a time
		* recursive approach
		* the contraction path can then be found using other optimizers (like greedy or exhaustive) on the partitions
		* KaHyPar for partitioning (HyperPar)
			![[partitioning.png]]
* To improve performance they added randomness and bayesian optimization to sample better paths. These parameters are able to be tunned
* There are some ways to simplify a network
	* Diagonal reduction
		* reduce t to k-1 dimension
		* ix becomes a hyperedge
		![[diagonal.png]]
	* Rank Simplification
		* absorb rank-1 and -2 with neighbors
		![[rank.png]]
	* Anti-diagonal gauge
		![[gauge.png]]
	* Column reduction
		![[column.png]]
	* Split simplify
		* Uses SVD
		* split tensors into smaller ones aiming to create a more tensors to reduce the cut-weight
		![[split.png]]

---

## Complexity

![[complexity1.png]]
![[complexity2.png]]
![[complexity3.png]]
![[contraction1.png]]


* Hyper-Par was in general the best approach
