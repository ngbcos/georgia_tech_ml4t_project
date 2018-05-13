
import numpy as np
import gen_data
import math

print np.random.randint(0, 2*math.pi, size=(100,2))

X, Y = gen_data.best4DT(123)

X2, Y2 = gen_data.best4DT(123)

XL1, YL1 = gen_data.best4LinReg(123)

XL2, YL2 = gen_data.best4LinReg(123)

print X
print Y

print np.array_equal(X, X2)
print np.array_equal(Y, Y2)

print np.array_equal(XL1, XL2)
print np.array_equal(YL1, YL2)

print X.shape
print Y.shape

print np.random.randint(0, 2*math.pi, size=(100,2))