from PIL import Image
import os
# Specify the root directory containing class folders, e.g., 'images'
root_directory = 'c:/Users/U20531232/Desktop/food_classification ml/images'





# List to store the file paths of problematic images
problematic_images = []

# Loop through the class directories
for class_folder in os.listdir(root_directory):
    class_folder_path = os.path.join(root_directory, class_folder)
    if os.path.isdir(class_folder_path):
        # Loop through the images in the class folder
        for filename in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, filename)
            try:
                # Attempt to open the image
                with Image.open(file_path) as img:
                    # Perform any necessary processing here
                    pass
            except (OSError, Image.UnidentifiedImageError) as e:
                # Handle the specific exception (UnidentifiedImageError in this case)
                print(f"Error opening image '{file_path}' in class '{class_folder}': {e}")
                # Add the problematic image's file path to the list
                problematic_images.append(file_path)

# Print or log the list of problematic images
print("Problematic images:")
print(problematic_images)
