# Real-Time-Face-Expression-Recognition
build and train a convolutional neural network (CNN) in Keras from scratch to recognize facial expressions


The data consists of 48x48 pixel grayscale images of faces. The objective is to classify each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). You will use OpenCV to automatically detect faces in images and draw bounding boxes around them. Once you have trained, saved, and exported the CNN, you will directly serve the trained model predictions to a web interface and perform real-time facial expression recognition on video and image data

# Project Structure

## Task 1: Importing moduls and libraries
    Importing essential modules and helper functions from NumPy, Matplotlib, and Keras.
## Task 2: Explore the Dataset
    Displaying some images from every expression type in the Emotion FER dataset.
    Checking for class imbalance problems in the training data.
## Task 3: Generate Training and Validation Batches
    Generating batches of tensor image data with real-time data augmentation.
    Specifing paths to training and validation image directories and generates batches of augmented data.
## Task 4: Create a Convolutional Neural Network (CNN) Model
    Designing a convolutional neural network with 4 convolution layers and 2 fully connected layers to predict 7 types of facial expressions.
    Using Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.
## Task 5: Train and Evaluate Model
    Trained the CNN by invoking the model.fit() method.
    Used ModelCheckpoint() to save the weights associated with the higher validation accuracy.
    Observed live training loss and accuracy plots in Jupyter Notebook for Keras.
## Task 6: Save and Serialize Model as JSON String
    Used to_json(), which uses a JSON string, to store the model architecture.
## Task 7: Create a Flask App to Serve Predictions
    Used open-source code from "Video Streaming with Flask Example" to create a flask app to serve the model's prediction images directly to a web interface.
