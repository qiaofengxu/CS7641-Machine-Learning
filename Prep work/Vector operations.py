import numpy as NP
from scipy import linalg as LA

print "\n---------- vector operations with Numpy and Scipy -----------" 


A = NP.random.randint(0,10,25).reshape(5,5)  # random integer filled 5x5 matrix
B = NP.random.randint(0,10,25).reshape(5,5)
ones = NP.ones((5,5), dtype=NP.int)  # ones matrix
print ones
identity = NP.identity(5, dtype = NP.int)  # identity matrix

print A
print type(A)  # verify it's a matrix
print isinstance(A, NP.ndarray)
print B

C = NP.add(ones, identity)  # matrix addition
print C

D = 3 * ones  # scalar multiplication
print D

F = NP.array([[9,5,4,3,7],[3,3,2,9,7],[6,5,3,4,0],[7,3,5,5,5],[2,5,4,7,8]])

print F



Threes = 3 * ones

normalized_threes = Threes / NP.linalg.norm(Threes)

print normalized_threes

def calc_unit_vector(v):
	if not isinstance(v, NP.ndarray):
		raise TypeError, "not an numpy array"
	norm =  NP.linalg.norm(v)
	normalized_v = v / norm
	return v, norm, normalized_v


v = NP.array([1, -2])

norm =  NP.linalg.norm(v)

normalized_v = v / norm
print v, norm, normalized_v

print "\n---------- dot product -----------"

u = NP.array([1, 2])
v = NP.array([1, 1])

print u
print v

print "dot product:", NP.dot(u, v)

print "\n----------- Cauchy Schwarz inequality -----------"

u = NP.array([1, 2])
v = NP.array([1, 1])

print "length u :", calc_unit_vector(u)[1] 
print "length v :", calc_unit_vector(v)[1] 
print "length u+v :", calc_unit_vector(NP.add(u, v))[1] 
print "abs(u dot v):", abs(NP.dot(u, v)), "should be less than..."
print "...length u x length v :", calc_unit_vector(u)[1] * calc_unit_vector(v)[1] 

print "\n----------- Orthogonal Vectors -----------"

# two vectors are perpindicular/orthogonal if the angle between them is 90degrees
# (a dot b) = norm(a)norm(b) cos(theta), cos(90) = 0, so (a dot b) = 0
# but there must be an angle, so a and b can't be a zero vector

u = NP.array([1, 0])
v = NP.array([0, 1])

print u
print v

print "orthogonal dot product:", NP.dot(u, v)

