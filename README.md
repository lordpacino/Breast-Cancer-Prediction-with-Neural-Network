# Breast Cancer Prediction with Neural Network
## 1. Summary
The aim of this project is to create a highly accurate deep learning model to predict breast cancer (whether the tumour is malignant or benign). The model is trained with data obtained from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## 2. IDE and Framework
The main IDE used for this project is Spyder and the main frameworks involved in this project are Pandas, Scikit-learn and TensorFlow Keras.

## 3. Methodology

### 3.1. Data Pipeline
The data is first loaded and preprocessed, with unwanted features are removed, and label is encoded in one-hot format. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

### 3.2. Model Pipeline
A feedforward neural network is constructed and the structure of the model is quite simple actually. Below is the figure of the model :

![Screenshot 2022-04-27 012103](https://user-images.githubusercontent.com/76200485/165357826-6c36938b-a674-4938-8f25-c7831a6b2610.jpg)

The model is trained with a batch size of 32 and for 100 epochs. Early stopping is applied and the training is stopped at epoch 24, with a training accuracy of 99% and validation accuracy of 95%. 

## 4. Results
The model accuracy is plotted in graph to better visualize the overall outcomes. The graph is shown in figure below.

![Screenshot 2022-04-27 014216](https://user-images.githubusercontent.com/76200485/165360429-57865e7b-64f9-46df-bd45-980be6fd0c98.jpg)

