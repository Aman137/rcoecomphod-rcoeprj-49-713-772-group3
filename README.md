# rcoeprj-49-713-772-group3

# Image Captioning using AI

> **Abstract** : AI image captioning is when a computer can describe pictures in words, using neural networks to understand the images and generate human-like captions for them.

### Project Members
1. ANSARI AMAN JAVED AHMED  [ Team Leader ] 
2. KHAN OSAMA ABDUL SALAM 
3. BHATI FAZAL AFZAL 
4. GALIYARA NOMAAN HASHIM 

### Project Guides
1. PROF. NARGIS SHAIKH (ECS)  [ Primary Guide ] 

### Deployment Steps
Please follow the below steps to run this project.
1. Dataset Preparation:
 Assemble a diverse dataset of images paired with corresponding captions.
2. Data Pre-processing:
 Standardize image sizes and tokenize captions for compatibility with model input.
3. Feature Extraction:
 Employ a pre-trained CNN to extract visual features from input images.
4. Model Design:
  Create the image captioning model, combining the CNN as an encoder and an LSTM network
  as a decoder.
  Implement an attention mechanism for improved context awareness.
5. Training:
   Split the dataset into training, validation, and test sets. o Train the model, optimizing
parameters using a chosen loss function (e.g., crossentropy).
6. Validation and Tuning:
   Evaluate model performance on the validation set using metrics (e.g., BLEU, METEOR).
   Fine-tune hyperparameters or adjust the model architecture based on validation results.
7. Testing:
   Assess the model's generalization on the test set.
8. Inference:
   Deploy the trained model for generating captions on new images.
   Input images, extract features, and generate captions iteratively.
9. Evaluation:
     Assess the quality of generated captions using metrics to ensure accuracy.
10. Refinement:
    Refine the model based on evaluation results or user feedback.
11. Deployment:
    Integrate the trained model into applications or platforms requiring automated image
    captioning.

### Subject Details
- Class : BE (AI&DS) Div A - 2023-2024
- Subject : Major Project1 (MP I (A)(R19))
- Project Type : Major Project

### Platform, Libraries and Frameworks used
1. [Pycharm](https://www.jetbrains.com/pycharm/)
2. [G-colab](https://colab.google/)
3. [TensorFlow](https://tensorflowjs.com)


### References
1. Journal:
• Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. "Show and Tell: A
Neural Image Caption Generator", No of pages - 9, 20 April,2015
• Xiangyu Zhang. Jianhua Zou, Kaiming Het . and Jian Sun. "Accelerating Very Deep
Convolutional Networks for Classification and Detection", No of pages - 14, 18
November, 2015.
• Mare Tanti, Albert Gatt, Kenneth P. Camilleri. "What is the Role of Recurrent Neural
Networks(RNNs) in an Image Caption Generator", No of pages – 10.25 August,2017
• Alex Sherstinsky . "Fundamentals of Recurrent Neural Network (RNN) and Long
Short-Term Memory (LSTM) Network, No of pages - 43, 31 May, 2020
2. https://docs.python.org/3.8/
3. https://www.jetbrains.com/pycharm/documentation/
4. https://www.tensorflow.org/api_docs/python/tf/all_symbols
5. https://keras.io/guides/
6. https://numpy.org/doc/
7. https://pillow.readthedocs.io/en/stable/
8. https://tqdm.github.io/
9. https://jupyterlab.readthedocs.io/en/stable/
