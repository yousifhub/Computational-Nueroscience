# Functions
def sigmoid(x):
    return 1 / (1 + (2.718281828459 ** -x))

def Layer(X, Y, W1, W2, Bias):
    return sigmoid((X * W1) + (Y * W2) + Bias)

# Network
Input1 = .05
Input2 = .10

# Weights Before Hidden Layer
W1, W2 = .15, .20
W3, W4 = .25, .30
# Weights After Hidden Layer
W5, W6 = .40, .45
W7, W8 = .50, .55

B1 = 0.35
B2 = 0.60

IntervalUp = .5
IntervalDown = -.5

# Main
HiddenLayer1 = Layer(Input1, Input2, W1, W2, B1)
HiddenLayer2 = Layer(Input1, Input2, W3, W4, B2)

print("Hidden Layer 1: ", HiddenLayer1)
print("Hidden Layer 2: ", HiddenLayer2)

OutputLayer1 = Layer(HiddenLayer1, HiddenLayer2, W5, W6, B1)
OutputLayer2 = Layer(HiddenLayer1, HiddenLayer2, W7, W8, B2)

print("Output Layer 1: ", OutputLayer1)
print("Output Layer 2: ", OutputLayer2)
