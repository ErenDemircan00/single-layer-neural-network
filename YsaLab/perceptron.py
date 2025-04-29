import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def derivate(x):
    return sigmoid(x) * (1- sigmoid(x))


X = numpy.array([
    [0, 0], [0, 1], [1, 0], [1, 1]])

Y = numpy.array([[0],[0],[0],[1]])

lr = 0.02
epoch=1000
w = numpy.random.randn(2,1)
b = numpy.random.randn(1)

for epoch in range(epoch):
    z = numpy.dot(X, w) + b
    y = sigmoid(z)
    error = y-Y

    dz = error * derivate(z)
    dw = numpy.dot(X.T,dz)
    db = numpy.sum(dz)

    w-= lr*dw
    b-= lr*db

print(w)
print(b)