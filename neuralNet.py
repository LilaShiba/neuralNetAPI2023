import numpy as np

class BasicNeuralNet:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_ih = np.random.randn(hidden_size, input_size)
        self.weights_ho = np.random.randn(output_size, hidden_size)
        self.bias_h = np.random.randn(hidden_size, 1)
        self.bias_o = np.random.randn(output_size, 1)

    def preprocess(self, vector):
        '''
        Preprocess a vector to ensure it's in the shape of a column vector.
        '''
        return np.array(vector).reshape(-1, 1)
        
    def train(self, input, output, epochs=5000):
        for epoch in range(epochs):
            for input_vector, target_vector in zip(input, output):
                input_vector = self.preprocess(input_vector)
                target_vector = self.preprocess(target_vector)
                output = self.feed_forward(input_vector)
                self.back_propagate(input_vector, target_vector, output)
    
    def feed_forward(self, x):
        self.hidden_output = self.sigmoid(np.dot(self.weights_ih, x) + self.bias_h)
        self.final_output = self.sigmoid(np.dot(self.weights_ho, self.hidden_output) + self.bias_o)
        return self.final_output
    
    def back_propagate(self, x, y, output, learning_rate=0.1):
        output_errors = y - output
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        self.weights_ho += learning_rate * np.dot((output_errors * self.sigmoid_deviation(output)), self.hidden_output.T)
        self.bias_o += learning_rate * output_errors * self.sigmoid_deviation(output)
        
        self.weights_ih += learning_rate * np.dot((hidden_errors * self.sigmoid_deviation(self.hidden_output)), x.T)
        self.bias_h += learning_rate * hidden_errors * self.sigmoid_deviation(self.hidden_output)

    def predict(self, input_data):
        input_data = np.array(input_data).T
        predictions = [self.feed_forward(self.preprocess(input_data[:, i])).flatten().tolist() for i in range(input_data.shape[1])]
        return predictions
    
    def sigmoid(self, vector):
        return 1 / (1 + np.exp(-vector))
    
    def sigmoid_deviation(self, vector):
        sx = self.sigmoid(vector)
        return sx * (1 - sx)

if __name__ == "__main__":
    input_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    target_data = [[0.4, 0.8], [0.6, 0.4], [0.8, 0.2]]

    nn = BasicNeuralNet(3, 2, 2)
    nn.train(input_data, target_data, epochs=5000)
    predictions = nn.predict(input_data)
    print(predictions)
