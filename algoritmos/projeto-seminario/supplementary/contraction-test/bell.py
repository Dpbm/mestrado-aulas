import quimb.tensor as qtn
import numpy as np

circ = qtn.Circuit(2)
circ.apply_gate('H', 0)
circ.apply_gate('CNOT', 0, 1)

print("By default, Quimb creates MPS states")
tn = circ.psi
# tn.draw(
#     color=['H', 'CX'],  
#     show_inds=True,     
#     show_tags=True,
#     layout='neato'
# )


print("To use TTN we could do something like:")
psi = circ.psi.contract()

# first split
T_left, T_right = qtn.tensor_split(
    psi,
    left_inds=['k0'],
    right_inds=['k1'],
    method='svd'
)

# wrap as tensors
tn = qtn.TensorNetwork([
    qtn.Tensor(T_left, inds=['k0', 'b']),
    qtn.Tensor(T_right, inds=['b', 'k1']),
])

# tn.draw()


print("However, it doesn't seem like a tree, so to see it better let's check with more qubits")

tn.draw# 1. Define tree parameters
n_sites = 8      # Number of physical sites (leaves of the tree)
bond_dim = 4     # Bond dimension of internal edges
phys_dim = 2     # Physical dimension (e.g., 2 for qubits)
max_degree = 3   # 1 parent + 2 children = binary tree

# 2. Generate a random Tree Tensor Network (TTN) state
# This creates a network with a tree geometry and random tensor entries.
ttn = qtn.TN_rand_tree(
    n=n_sites,
    D=bond_dim,
    phys_dim=phys_dim,
    max_degree=max_degree,
    site_tag_id='I{}'  # Tags leaves as I0, I1, ...
)

# 3. Basic Properties
print(f"Number of tensors: {ttn.num_tensors}")
print(f"Is it a tree structure? {ttn.istree()}")

# 4. Canonicalize the state
# For trees, we can gauge the bonds so that tensors are isometric 
# towards a 'root'. Here we canonize towards site 'I0'.
ttn.canonize_around_('I0')

# 5. Compute the Norm
# Because it's a tree, the norm is calculated very efficiently via contraction.
norm = ttn.norm()
print(f"Norm of the state: {norm:.6f}")

# 6. Compute a 1-site Reduced Density Matrix (RDM)
# Since we canonized around I0, the RDM for site 0 is just the tensor at I0 
# contracted with its conjugate. Quimb's partial_trace handles this generally.
# rho_0 = ttn.partial_trace([0])
# print("\nReduced Density Matrix for site 0:")
# print(rho_0)

ttn.draw(
    color=['I0', 'I4'],        # Highlight specific site tags
    show_inds='all',           # Show all bond labels
    legend=True,               # Add a legend for the colors
    figsize=(8, 8)             # Adjust the window size
)
