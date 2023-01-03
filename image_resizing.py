import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the images by cropping them to the desired size
x_train_cropped = []
for img in x_train:
  img = Image.fromarray(img)
  img = img.crop((5, 5, 25, 25))  # Crop the image to 20x20 pixels
  x_train_cropped.append(np.array(img))
x_train = np.array(x_train_cropped)

x_test_cropped = []
for img in x_test:
  img = Image.fromarray(img)
  img = img.crop((5, 5, 25, 25))  # Crop the image to 20x20 pixels
  x_test_cropped.append(np.array(img))
x_test = np.array(x_test_cropped)

# Create a model using TensorFlow
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(20, 20)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save('my_model.h5')
