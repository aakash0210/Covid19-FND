# COVID-19 Fake News Detection

## Overview
This repository contains a Jupyter Notebook aimed at detecting fake news related to COVID-19. The project involves various tasks such as data preprocessing, model training, and evaluation to create a robust fake news detection system.

## Description of Code Blocks

### 1. Installation of Required Packages
This code block installs the necessary Python packages, including `tweet-preprocessor`, which is used for preprocessing tweets.

### 2. Importing Libraries and Modules
Here we import all the required libraries and modules for data manipulation, preprocessing, and model building. These include pandas, scikit-learn, Keras, matplotlib, and others.

### 3. Mounting Google Drive
This section mounts Google Drive to access data stored in specific folders.

### 4. Data Loading
Defines a function to load data from TSV files and loads the training, validation, and test datasets from the specified file paths.

### 5. Text Preprocessing
Defines a function to preprocess tweets, removing URLs, converting text to lowercase, and removing special characters. Applies this preprocessing to the datasets.

### 6. Data Segregation
Splits the concatenated data into real and fake news based on their labels.

### 7. Word Cloud Generation (Positive and Negative)
Generates word clouds for both real and fake news data to visualize the most common words in the dataset.

### 8. Plotting Word Clouds
Plots the generated word clouds for both real and fake news data.

### 9. Model Evaluation
Loads evaluation data and evaluates model performance using metrics like accuracy, F1 score, and classification report.

### 10. TF-IDF Vectorization
Converts text data into numerical TF-IDF vectors at different levels: word, n-gram, and character.

### 11. Tokenization and Padding
Tokenizes text data and pads sequences to ensure uniform length for input to deep learning models.

### 12. Loading Pre-trained Word Embeddings
Loads pre-trained word embeddings to represent words as dense vectors.

### 13. Loading Embeddings from Pickle
Loads embeddings from a saved pickle file.

### 14. Label Encoding
Encodes categorical labels (real and fake) into numerical format and converts labels into one-hot encoded vectors.

### 15. Training Model
Defines a function to train machine learning and deep learning models. Trains models and evaluates their performance.

### 16. Model Training and Evaluation
Trains machine learning models like XGBoost and MLP (Multi-Layer Perceptron) and evaluates their performance.

### 17. Training MLP Classifier with Various Epochs
Trains an MLP classifier with different numbers of epochs and evaluates its performance.

### 18. Creating CNN Model
Defines a convolutional neural network (CNN) model architecture and trains it. CNNs are commonly used for text classification tasks.

### 19. Creating RNN-LSTM Model
Defines a recurrent neural network with Long Short-Term Memory (LSTM) units and trains it. LSTM networks are effective for sequence data like text.

## Usage
To use this notebook, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python dependencies listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook and execute each code block sequentially.

## Conclusion
This README provides an overview and purpose of the project. Reader can understand the project's structure and functionalities before diving into the code.

