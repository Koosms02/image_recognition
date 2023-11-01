from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the data generator for training data

print("training of the model has started ------")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load and preprocess the training data
train_generator = train_datagen.flow_from_directory(
    'images',
    target_size=(224, 224),  # Input image size for the model
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)


print("---------------")
print("now building a CNN model ------")

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(703, activation='softmax'))  # Output layer with number of classes


print("compiling the model ------")
# Compile the model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)  # Adjust the number of epochs as needed


print("compiling the ------")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


print("everything works smoothly")
# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test accuracy:", test_accuracy)

print("saving the model")

import tensorflow as tf

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

