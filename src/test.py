import dynet as dy
import numpy as np

dy.renew_cg()
value = 0
dimension = 1

mm = dy.Model()

"""
## ==== Creating Expressions from user input / constants.
x = dy.scalarInput(value)

v = dy.vecInput(dimension)
v.set([1,2,3])

dim1=5
dim2 = 4
input = range(20)
input = np.asarray(input).reshape((5,4))
z = dy.inputTensor(input)


# Or directly from a numpy array
z1 = dy.inputTensor([[1,2],[3,4]]) # Row major

## ==== We can take the value of an expression.
# For complex expressions, this will run forward propagation.
print 'z.value()', z.value()
print 'z.npvalue() ', z.npvalue()      # as numpy array
print 'v.vec_value()', v.vec_value()    # as vector, if vector
print 'x.scalar_value()', x.scalar_value() # as scalar, if scalar
print 'x.value()', x.value()        # choose the correct one

"""

## ==== Parameters
# Parameters are things we tune during training.
# Usually a matrix or a vector.

# First we create a parameter collection and add the parameters to it.
m = dy.ParameterCollection()
pW = m.add_parameters((8,8)) # an 8x8 matrix
pb = m.add_parameters(8)

# then we create an Expression out of the parameter collection's parameters
W = dy.parameter(pW)
b = dy.parameter(pb)
e = dy.concatenate([[1,2],[3,4]])