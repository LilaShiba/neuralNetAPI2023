# Neural Network API 🧠🤖

Welcome to the Neural Network API! This API allows you to train and predict with a basic neural network using a simple web interface. 🌐

## Getting Started

To use the Neural Network API, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt` in your terminal. Make sure you have Python 3.6 or later installed.

2. Start the Flask server by running `python app.py`. The server will be running at `http://localhost:5000`.

3. Open your web browser and go to `http://localhost:5000`. You will see the web interface for the Neural Network API.

## Training the Neural Network 🚀

To train the neural network, follow these steps:

1. Enter the input data and target data in the respective text areas. Each value should be separated by a comma, and each data sample should be on a new line.

2. Specify the number of epochs and learning rate for the training process.

3. Click the **Train** button. The neural network will be trained with the provided data.

4. Once the training is complete, you will see two charts: the training error chart and the neural network output chart. These charts visualize the training progress and the predicted outputs.

## Predicting with the Neural Network 🔮

To make predictions with the trained neural network, follow these steps:

1. Enter the input data for which you want to make predictions in the text area. Each value should be separated by a comma, and each data sample should be on a new line.

2. Click the **Predict** button. The neural network will generate predictions for the provided input data.

3. The predicted outputs will be displayed below the Predict button.

## Examples 🌟

To help you get started, here are some example input and target data:

Input data:

```csv
0.1, 0.1, 0.3
0.1, 0.5, 0.2
0.3, 0.8, 0.9
```

Target data;

```csv
0.4, 0.8
0.6, 0.4
0.1, 0.2
```

## Output
![img1](imgs/img1.png)

Feel free to modify the input and target data to experiment with different training scenarios.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

