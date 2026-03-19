link: https://arxiv.org/abs/cond-mat/0112110

---
tags: [Social, Graphs, Connection, Communities, algorithm]

---
## What they did?

- A new method for identifying communities in networks

---
## ideas

- Old Approach
  1. Calculate the weight Wij for each pair i,j
  2. Order vertices by their weight Wij (from most to least)
  3. add edges connecting each pair

- Their algorithm
    1. calculate the betweeness for all edges
    2. remove the edge with the highest betweeness
    3. recalculate the betweeness
    4. repeat starting from 2. until no edges remain

    the betweeness is calculated using the Newman's algorithm
---

## Complexity

- Newman's algorithm is O(nm) (m edges and n vertices)
- The whole algorithm is O(m^2n)
- However, since the recalculation only must be done for affeteced edges, which are those at most in the same componenet as the removed one, the
algorithm in general acts better than the worst case
- is O(n³) for sparse graphs