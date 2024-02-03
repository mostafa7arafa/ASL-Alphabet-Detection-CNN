# ASL Alphabet Detection CNN

## 1. Introduction

American Sign Language (ASL) is a language with its own set of rules and grammar, expressed through hand movements. It serves as the primary language for individuals who are deaf or hard of hearing. Diverse sign languages exist globally, with variations such as American Sign Language (ASL) differing from British Sign Language (BSL). ASL comprises 26 letters from the English alphabet, represented uniquely through handshapes. Machine learning, particularly neural networks, plays a pivotal role in ASL alphabet detection, aiding in the prediction of letters based on visual data.

This project employs a dataset comprising images of ASL alphabet signs, organized into 29 folders representing various classes. The dataset is divided into training and testing sets, with 26 classes corresponding to the letters A-Z and 3 classes for SPACE, DELETE, and NOTHING.

## 2. Related Work

Previous research in ASL alphabet detection using AI models has shown significant advancements. Notable works include real-time ASL alphabet recognition using Convolutional Neural Networks (CNNs), transfer learning for enhanced recognition with limited data, and multimodal fusion for improved accuracy by combining visual information with facial expressions.

## 3. Proposed Model

This project adopts a handcrafted CNN model, consisting of three convolution layers, each followed by a "relu" activation function, pooling layer, and batch normalization layer. The final layer is a flatten layer, feeding the data into a fully connected neural network for classification.

## 4. Experimental Work

### 4.1 Dataset

The "ASL Alphabet" dataset from Kaggle is utilized, comprising training and testing parts with images representing hand signs for letters and additional symbols (space, nothing, and delete).

### 4.2 Evaluation Metrics

Evaluation metrics play a crucial role in assessing model performance. The Confusion Matrix, F1-score, Precision, Recall, and Accuracy are utilized to gain insights into the model's effectiveness.

### 4.3 Cross Validation

To ensure robust accuracy, a K-fold cross-validation technique with 10 folds is applied. This technique provides a comprehensive assessment of the model's generalization capabilities. The mean accuracy achieved through cross-validation is an impressive 99.99%, further validating the model's reliability.

![image](https://github.com/mostafa7arafa/ASL-Alphabet-Detection-CNN/assets/58299212/5eb1dd0e-ebd6-4d91-bb0e-3ab1899a90da)


### 4.4 Results

The experimental results demonstrate the effectiveness of the handcrafted CNN model. Testing on the dataset yielded an impressive 99.99% accuracy without overfitting. The simplicity of the dataset, with many identical images, facilitated high accuracy without compromising the model's generalization capabilities.

## 5. Conclusion

In conclusion, the project successfully leverages a handcrafted CNN model for ASL alphabet detection, showcasing robust performance validated through rigorous evaluation metrics and K-fold cross-validation. The achieved mean accuracy of 99.99% underscores the reliability of the model, indicating its potential for real-world applications in enhancing accessibility and communication for individuals using American Sign Language.
