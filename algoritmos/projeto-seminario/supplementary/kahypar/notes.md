# Preprocessing

  - pin sparsification
    - identify and contracts vertices with similar neighborhoods reducing the hyperedge size
    - Locality-Sensitive Hashing
      1. all vertices are marked as unclustered
      2. each pass identifies buckets of similar vertices 
        - uses min-hash fingerprints
        - only vertices with the same fingerprint go to the same bucket (all hashes are the same)
        - only a max number of elements can be put inside, after that the bucket becomes inactive
      3. stops as soon as the number of resulting clusters is less than n/2 or the max of passes is reached
      4. each cluster is contracted into a single vertex

  - community detection (Louvain algorithm)
    > phase 1
    1. assign a different community for each node in he network
    2. for each node i, consider the neighbors and calculate the gain of modularity it would
        give by moving i from community i to j 
    3. move i to the community where the gain is max and positive
    4. repeat until not further improvement is reachable

    > phase 2
    5. construct the graph
      6. the link between nodes are the sum of the internal group weights

    repeat phases

----

# Coarsening

  1. at the beginning of each pass a permutation of the vertex set is created 
  2. for each vertex u find a partner v with the highest rating and contract them
    - v is removed from the graph
    - v is only considered if the it doesnt surpass the maximum vertex weight
    - when more than one cadidate is found the one that has not participated of any contractions 
      yet is chosen
    - for large nets the rating function is not evaluated
  3. stop pass when every node was considered and start another pass with a new permutation
    of the remaining vertices
  4. stop all when a criteria is met of no eligible vertex is left


----


# Partitioning

  - RB algorithm
    - if k is a power of 2.
      - you compute a bipartition and then recurse on each block
      - takes log k of these
    - Otherwise
      - partition is done first with two then block a is floor(k/2) and b 
        ceil(k/2)
      - always keeping track of imbalances

  - refined via 2-way localized local search
    - two priority queues (PQs) 
      - keep the next moves for all vertices
      - one for each block 
    1. all queues are empty and inactive and all vertices marked as inactive
    2. activate uncontracted border vertices only
    3. calculate the FM gain for moving the vertex v to another block j and insert it into the j queue 
    4. repeatedly the algorithm queries the active nonempty queues finding the highest gain and moving the respective node to block j
      5.  update neighbors of v by activating them, if were inactive. Otherwise they become internal and are removed from the queue and become inactive.
      6. v becomes marked
      7. Update the gain of neighbors if the net is not locked
      8. continues until a stop rule is reached or no non-empty enabled PQ remains
    9. Reverse the moves and find the lowest cut state that fulfills the balance criteria

  - net is locked when there's at least one marked pin in each of the two blocks

------

#  Refinement

  - k-way local search algorithm
    - it doesnt compute the gains for all blocks 
    - only maintains feasibility

  - flow-based refinement algorithm  (FlowCutter) (some times -> 7)
    1. given partitions V_0 and V_1, decide which vertices can be moved M.
      2. BFS for each V, starting with border indexes
        3. each visited node is added to M
        4. run until a weight constraint is broken
    5. s=diff(V_0, M), t=diff(V_1, M)
    6. compute the maximum flow (giving the source-side and target-side minimum cut)
      7. rebalance if necessary
        8. move incident vertices from the larger side to the smaller one
        9. augment the flow (uses the same procedure to find the max flow)

      max flow procedure:
        - 2 alternating phases
        - repeat until it can be augmented
        - runs a BFS to find the distances between levels of neighbors 
        - runs DFS to find the shortest edge-disjoint augmenting paths

