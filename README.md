# AI-Based Iris Flower Classification Using KNN

## Project Overview
This project is developed as part of the DecodeLabs Artificial Intelligence Internship – Project 2: Data Classification Using AI.

The project demonstrates a basic supervised learning classification pipeline using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

## Objective
The main objective is to build a classification model that can learn patterns from the Iris dataset and classify flowers into their respective species.

## Dataset
The Iris dataset contains 150 samples, 4 features, and 3 target classes.

### Features
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Classes
- Setosa
- Versicolor
- Virginica

## Methodology
The project follows these steps:

1. Load the Iris dataset.
2. Understand the dataset and its features.
3. Split the data into 80% training and 20% testing sets.
4. Apply feature scaling using StandardScaler.
5. Test different K values for the KNN algorithm.
6. Select the best K based on the error rate.
7. Train the final KNN classification model.
8. Make predictions on the test data.
9. Evaluate the model using accuracy, confusion matrix, precision, recall, and F1-score.

## Machine Learning Algorithm
### K-Nearest Neighbors (KNN)
KNN is a supervised machine learning classification algorithm that classifies data points based on their nearest neighboring data points.

## Results
The final KNN model achieved an accuracy of **96.67%** on the test dataset.

The project also generates:
- KNN Error Rate Graph
- Confusion Matrix
- Classification Report

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Files
- `main.py` – Main machine learning program
- `requirements.txt` – Required Python libraries
- `README.md` – Project documentation
- `knn_error_rate.png` – KNN error-rate graph
- `confusion_matrix.png` – Confusion matrix

## Conclusion
This project demonstrates the fundamental supervised learning workflow for data classification. The KNN model successfully classified Iris flower species with **96.67% accuracy**, showing strong performance on the test dataset.

## Internship
**DecodeLabs Artificial Intelligence Internship – Project 2**
**Project: Data Classification Using AI**