link: https://arxiv.org/abs/2001.08063

---
tags: [algorithms, Tensor Network, cost function, comparing algorithms, contraction]

---
## What they did?

They experimented genetic algorithms and simulated annealing for tensor network contraction.

---
## ideas
* tensor contractions is a summation of intermediate bonds
* tensor networks are a list of tensors with their indices to be contracted
* Indices that are not in the summation will not be contracted
* Individual summations commute, we can contract everything at once or create intermediate tensors and use them at the end

![[whole.png]]
![[parts.png]]

* we could do the inverse way as well. The order doesn't affect the result, but the time to process.
![[another-order.png]]
* the order the indices are contracted is called contraction sequence
* The cost is a global optimization problem, so greedy algorithm doesn't perform well as simulated annealing
* The advantage of simulated annealing grows as the networks grows as well
---

## Complexity
* Contractions is NP
* naive contraction grows exponentially 
* Just the order of contractions says how difficult it would be, so finding the best path the time downgrades to Polynomial time
* If a tensor Q has higher-rank (number of indices) the required memory and compute power is grater than Q' with lower-rank
* The computational cost is calculated via FLOPS (amount of floating point operations, in this case additions)
![[cost-function.png]]
![[differences.png]]
* The exhaustive approach grows exponential with the number of edges E, O(e^E)
* The greedy algorithm picks a index that could lead to a better cost at that given moment, so in general it's O(E^k-1), being k the number of steps taken simultaneously. in general it has constant time for each step O(1).