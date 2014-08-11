import numpy as NP
from scipy import linalg as LA

print '\n----------- unit vector -----------'

def calc_unit_vector(v):
	if not isinstance(v, NP.ndarray):
		raise TypeError, "not an numpy array"
	norm =  NP.linalg.norm(v)
	normalized_v = v / norm
	return v, norm, normalized_v

print '\n----------- dimensions -----------'

A = NP.random.randint(0,10,25).reshape(5,5)  # random integer filled 5x5 matrix
B = NP.random.randint(0,10,25).reshape(5,5)
identity = NP.identity(5, dtype = NP.int) 
print A
print A.shape

print '\n----------- multiplication -----------'

print A.dot(identity).dot(B)

print '\n----------- Add a row of 1s -----------'

ones = NP.ones((1,5), dtype=NP.int)

Anew = NP.vstack([ones, A])
print Anew

print '\n----------- Add a column of 1s -----------'

ones = NP.ones((5,1), dtype=NP.int)

Anew = NP.hstack([ones, A])
print Anew

print '\n----------- upper and lower triangular -----------'

A = NP.random.randint(0,10,25).reshape(5,5)  # random integer filled 5x5 matrix

upper_tri = NP.triu(A)
lower_tri = NP.tril(A)

print A
print upper_tri
print lower_tri

# only the ones with sum of index <= k

partial_upper = A[NP.triu_indices(3)]
print partial_upper

print '\n----------- determinant -----------'
print LA.det(A)

print '\n----------- transpose -----------'

print NP.transpose(A)
print A.T

print '\n----------- inverse -----------'

print "inverse of A:\n", LA.inv(A)
print LA.det(LA.inv(A).dot(A))


print '\n----------- Eigenvalues, Eigenvectors -----------'


M = NP.array([[1,2], [1,0]])

M_e_vals, M_e_vecs = LA.eig(M)

v = NP.array([2, 1])

print calc_unit_vector(v)

print M

print M_e_vals
print M_e_vecs


print '\n----------- Singular Value Decomposition -----------'

M,N = A.shape
U,s,Vh = LA.svd(A)
Sig = LA.diagsvd(s,M,N)
print U
print s
print Vh
print Sig

# singular