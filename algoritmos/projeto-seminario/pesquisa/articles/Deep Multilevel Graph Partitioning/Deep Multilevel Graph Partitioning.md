link: https://arxiv.org/abs/2105.02022

---
tags: [algorithm, HPC, graph partitioning, MGP]

---
## What they did?

* an improvement on MGPs
* They:
	* coarse the graph until it reaches 2C (being C an input) nodes
	* the coarsed graph is partitioned in two blocks
	* uncoarse by keeping in mind that each partition can have around 2C nodes

---
## ideas
* coarsening --> reduce the graph into smaller parts but preserving its structure
* MGP:
	1. Coarse the graph multiple times until it's small (find clusters or matches and contract them)
	2. running partitioning algorithm (now it's fast -> initial partitioning(direct k-way partitioning, recursive bipartitioning, etc.))
	3. uncoarse the result (balacing at each step)
* The problem is that most MGPs stop coarsening when 2K nodes are reached, but when K is large the partitioning still with high cost
* KaMinPar is the best if K is large

---

## Complexity

* Balanced graph partitioning is at least NP-Hard