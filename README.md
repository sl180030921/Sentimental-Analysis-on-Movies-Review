# Sentiment Analysis on Movie Reviews
Sentiment analysis on movie reviews is a common natural language processing (NLP) task where the goal is to determine the sentiment or emotion conveyed in the review. This can be categorized as positive, negative, or neutral based on the text's tone.

This project implements sentiment analysis on movie reviews using various machine learning and deep learning models. The dataset is sourced from Kaggle and undergoes extensive preprocessing, tokenization, and feature extraction. Several algorithms, including Logistic Regression, LinearSVC, K-Neighbors, MLP, and CNN, are applied and evaluated for their performance.

---

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Results](#results)
5. [Conclusion](#conclusion)

---

## Features

- **Data Preprocessing**: Clean the raw text by removing noise, special characters, and converting the text to lowercase.
- **Tokenization and Stemming**: Tokenize the text into words, remove stop words, and apply stemming using the Porter Stemmer.
- **Model Implementation**: Implements multiple machine learning models:
  - Logistic Regression
  - LinearSVC
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Networks (CNN)
- **Model Evaluation**: Each model's accuracy is evaluated and compared.
- **Hyperparameter Tuning**: For MLP, hyperparameters such as hidden layers and activation functions are tuned for better performance.
- **Visualization**: Displays a bar chart comparing the accuracy of all models and outputs the results in a tabular format.

---

## Dataset

- **Source**: Kaggle (Movie Reviews (IMDB.csv)Dataset)
- **Structure**:
  - `review`: Text review of the movie.
  - `sentiment`: Sentiment label (e.g., positive, negative).
  
Make sure to place the dataset file (e.g., `IMDB.csv`) in the root directory of the project.

---
### Libraries/Tools:

Scikit-learn for traditional machine learning models.
TensorFlow / Keras / PyTorch for deep learning-based models.


## Usage

### Preprocessing

The following preprocessing steps are applied to the dataset:

1. **Noise Removal**: Special characters are removed, and text is converted to lowercase.
2. **Tokenization**: The text is split into words (tokens).
3. **Stop Words Removal**: Commonly used words (such as "the", "and", etc.) are removed.
4. **Stemming**: Words are reduced to their root form using the Porter Stemmer.

### Model Training and Evaluation

1. **Data Splitting**: The dataset is split into training (75%) and testing (25%) sets using `train_test_split`.
2. Use metrics like accuracy, precision, recall, and F1-score to evaluate the model's performance.
3. **Models Trained**: The following models are trained and evaluated:
   - Logistic Regression
   - LinearSVC
   - K-Nearest Neighbors (KNN)
   - Multi-Layer Perceptron (MLP) with tunable layers and activation functions
   - Convolutional Neural Networks (CNN)
4. **Evaluation**: The accuracy of each model is calculated and compared.

---

## Results

The models were trained and evaluated on the movie review dataset, and their accuracies are as follows:

### Model Performance

| **Model**                     | **Accuracy** |
|-------------------------------|--------------|
| Logistic Regression            | 0.9053       |
| Linear SVC                     | 0.9018       |
| K-Nearest Neighbors (KNN)      | 0.7685       |
| Multi-Layer Perceptron (MLP)   | 0.8736       |
| Convolutional Neural Network (CNN) | 0.8634       |

These results indicate that **Logistic Regression** and **Linear SVC** outperform other models in terms of accuracy.
## Conclusion
This project demonstrates the effectiveness of different machine learning models for text classification tasks and provides insights into how sentiment analysis can be applied to real-world data like movie reviews.

