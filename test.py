# import numpy as np
# import cv2
# from PIL import Image
# from tflite_runtime.interpreter import Interpreter

# # Path to the TensorFlow Lite model file
# model_path = "model.tflite"

# # Load the TensorFlow Lite model
# interpreter = Interpreter(model_path)
# interpreter.allocate_tensors()

# # Get input and output tensors
# input_tensor_index = interpreter.get_input_details()[0]['index']
# output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

# # Load an example image for testing
# image_path = "example_image.jpg"
# image = Image.open(image_path).convert("RGB")
# image = image.resize((224, 224))  # Resize the image to match the input size of the model
# image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
# image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# # Set the input tensor data
# interpreter.set_tensor(input_tensor_index, image_array)

# # Run inference
# interpreter.invoke()

# # Get the output predictions
# predictions = output()[0]

# # Load class labels (if available)
# with open("labels.txt", "r") as f:
#     labels = f.read().splitlines()

# # Get the predicted class index and label
# predicted_class_index = np.argmax(predictions)
# predicted_label = labels[predicted_class_index]

# # Print the results
# print(f"Predicted class index: {predicted_class_index}")
# print(f"Predicted label: {predicted_label}")
# print(f"Confidence: {predictions[predicted_class_index]}")
