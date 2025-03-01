# Functions
def sigmoid(x):
    return 1 / (1 + (2.718281828459 ** -x))

def deltaSigmoid(x):
    return x * (1 - x)

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

# Parameters
Target1 = 0.01
Target2 = 0.99
LearningRate = 0.5
Epoch = 10000

# Main
for epoch in range(Epoch):
    # Forward pass
    HiddenLayer1 = Layer(Input1, Input2, W1, W2, B1)
    HiddenLayer2 = Layer(Input1, Input2, W3, W4, B2)

    OutputLayer1 = Layer(HiddenLayer1, HiddenLayer2, W5, W6, B1)
    OutputLayer2 = Layer(HiddenLayer1, HiddenLayer2, W7, W8, B2)

    Error1 = Target1 - OutputLayer1
    Error2 = Target2 - OutputLayer2

    # Backpropagation
    DOutputLayer1 = Error1 * deltaSigmoid(OutputLayer1)
    DOutputLayer2 = Error2 * deltaSigmoid(OutputLayer2)

    DHiddenLayer1 = (DOutputLayer1 * W5 + DOutputLayer2 * W7) * deltaSigmoid(HiddenLayer1)
    DHiddenLayer2 = (DOutputLayer1 * W6 + DOutputLayer2 * W8) * deltaSigmoid(HiddenLayer2)

    W5 += LearningRate * DOutputLayer1 * HiddenLayer1
    W6 += LearningRate * DOutputLayer1 * HiddenLayer2
    W7 += LearningRate * DOutputLayer2 * HiddenLayer1
    W8 += LearningRate * DOutputLayer2 * HiddenLayer2

    W1 += LearningRate * DHiddenLayer1 * Input1
    W2 += LearningRate * DHiddenLayer1 * Input2
    W3 += LearningRate * DHiddenLayer2 * Input1
    W4 += LearningRate * DHiddenLayer2 * Input2

    B1 += LearningRate * (DHiddenLayer1 + DOutputLayer1)
    B2 += LearningRate * (DHiddenLayer2 + DOutputLayer2)

# Print final outputs
print("Hidden Layer 1: ", HiddenLayer1)
print("Hidden Layer 2: ", HiddenLayer2)
print("Output Layer 1: ", OutputLayer1)
print("Output Layer 2: ", OutputLayer2)
