# Transliteration without Attention

This repository contains code for implementing a transliteration model without attention mechanism. The code is implemented in Python using the PyTorch library. The main purpose of this model is to transliterate English words to Hindi.

## Files

- `NMT_v3.ipynb`: Jupyter Notebook containing the code for the transliteration model without attention.
- `Attention_NMT_v3.ipynb`: Another version of the code with attention mechanism implemented. This file is not included in this repository but can be found separately.

## Setup

1. Mount Google Drive: The code assumes that the dataset and other required files are stored in Google Drive. Make sure to mount your Google Drive by running the following code:
```
from google.colab import drive
drive.mount('/content/gdrive')
```

2. Set the working directory: After mounting Google Drive, set the working directory to the appropriate path by running the following code:
```
import os
os.chdir('/content/gdrive/My Drive/Deep Learning CS6910/rnn_test')
```

3. Install dependencies: The code requires several dependencies to be installed. Please make sure to install them by running the necessary commands, such as `pip install`.

## Data

The code expects the transliteration data to be in CSV format with two columns: English words and their corresponding Hindi transliterations. The data should be stored in the following directory structure:

```
./aksharantar_sampled/
    hin/
        hin_train.csv
        hin_valid.csv
        hin_test.csv
    ...
```

Make sure to update the file paths accordingly in the code.

## Usage

The code is divided into several sections, each serving a specific purpose. Here's a brief description of each section:

1. Data Loading: The code loads the transliteration data from CSV files and performs some preprocessing steps.

2. Model Architecture: The code defines the architecture of the transliteration model. The model is implemented as a recurrent neural network (RNN) with options to choose different RNN types (RNN, GRU, LSTM) and enable bidirectional processing.

3. Training: The code provides functions for training the model. You can set various hyperparameters, such as learning rate, number of batches, batch size, etc. The `train_setup` function trains the model and saves the trained model to a file.

4. Inference: The code provides a function for performing inference on the trained model. You can input an English word, and the model will generate the corresponding Hindi transliteration.

5. Evaluation: The code includes a function to calculate the accuracy of the model on a validation dataset. It compares the predicted transliterations with the ground truth and calculates the accuracy.

To use the code, follow these steps:

1. Mount Google Drive and set the working directory as described in the "Setup" section.

2. Ensure that the data files are in the correct directory structure and update the file paths in the code if necessary.

3. Run the code sections sequentially to load the data, define the model, train the model, perform inference, and evaluate the model.

## Note

This version of the code does not include the attention mechanism. If you are interested in using the attention-based model, refer to the `Attention_NMT_v3.ipynb` file, which contains the code with attention mechanism implemented.