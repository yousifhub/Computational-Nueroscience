import math

# Perceptron Functions
def sigmoid(x):
    return 1 / (1 + (2.718281828459 ** -x))

def deltaSigmoid(x):
    return x * (1 - x)

def Layer(X, Y, W1, W2, Bias):
    return sigmoid((X * W1) + (Y * W2) + Bias)

Input1 = .05
Input2 = .10

W1, W2 = .15, .20
W3, W4 = .25, .30

W5, W6 = .40, .45
W7, W8 = .50, .55

B1 = 0.35
B2 = 0.60

Target1 = 0.01
Target2 = 0.99
LearningRate = 0.5
Epoch = 10000

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

print("Perceptron Hidden Layer 1: ", HiddenLayer1)
print("Perceptron Hidden Layer 2: ", HiddenLayer2)
print("Perceptron Output Layer 1: ", OutputLayer1)
print("Perceptron Output Layer 2: ", OutputLayer2)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = [[0.01 * (i + j) for j in range(hidden_size)] for i in range(input_size)]
        self.Whh = [[0.01 * (i + j) for j in range(hidden_size)] for i in range(hidden_size)]
        self.Why = [[0.01 * (i + j) for j in range(output_size)] for i in range(hidden_size)]

        self.bh = [0.0 for _ in range(hidden_size)]
        self.by = [0.0 for _ in range(output_size)]

        self.h = [0.0 for _ in range(hidden_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def tanh(self, x):
        return math.tanh(x)

    # Forward pass
    def forward(self, x):
        h_new = []
        for j in range(self.hidden_size):
            h_sum = self.bh[j]
            for i in range(self.input_size):
                h_sum += x[i] * self.Wxh[i][j]
            for i in range(self.hidden_size):
                h_sum += self.h[i] * self.Whh[i][j]
            h_new.append(self.tanh(h_sum))
        self.h = h_new

        y = []
        for j in range(self.output_size):
            y_sum = self.by[j]
            for i in range(self.hidden_size):
                y_sum += self.h[i] * self.Why[i][j]
            y.append(self.sigmoid(y_sum))
        return y

    # Backpropagation
    def backward(self, x, dy, learning_rate=0.01):
        dWhy = [[0.0 for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        dby = [0.0 for _ in range(self.output_size)]
        dh = [0.0 for _ in range(self.hidden_size)]

        for j in range(self.output_size):
            dby[j] += dy[j]
            for i in range(self.hidden_size):
                dWhy[i][j] += self.h[i] * dy[j]
                dh[i] += self.Why[i][j] * dy[j]
                
        dWhh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        dWxh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        dbh = [0.0 for _ in range(self.hidden_size)]

        dh_next = [0.0 for _ in range(self.hidden_size)]
        for j in range(self.hidden_size):
            dh_total = dh[j] + dh_next[j]
            dtanh = (1 - self.h[j] ** 2) * dh_total
            dbh[j] += dtanh
            for i in range(self.hidden_size):
                dWhh[i][j] += self.h[i] * dtanh
                dh_next[i] += self.Whh[i][j] * dtanh
            for i in range(self.input_size):
                dWxh[i][j] += x[i] * dtanh

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.Why[i][j] -= learning_rate * dWhy[i][j]
        for j in range(self.output_size):
            self.by[j] -= learning_rate * dby[j]
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.Whh[i][j] -= learning_rate * dWhh[i][j]
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.Wxh[i][j] -= learning_rate * dWxh[i][j]
        for j in range(self.hidden_size):
            self.bh[j] -= learning_rate * dbh[j]

rnn = RNN(input_size=3, hidden_size=5, output_size=2)
input_vector = [0.5, 0.1, -0.3]
output = rnn.forward(input_vector)
print("RNN Output:", output)

output_error = [0.1, -0.2]
rnn.backward(input_vector, output_error)