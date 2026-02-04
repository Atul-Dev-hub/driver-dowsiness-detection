import os
import shutil

def move_first_100_images(source_folder, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Sort the images (optional: to ensure consistent order)
    images.sort()

    # Move the first 500 images
    for i, image in enumerate(images[:100]):
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.move(source_path, destination_path)

    print(f"Successfully moved {min(100, len(images))} images to {destination_folder}")

# r"" - r: raw string to handle backslashes in Windows paths
source_folder_closed_eyes = r"F:\ABHI\2025_things\Zsem-7 Project\dataset\train\Closed_Eyes"
source_folder_open_eyes = r"F:\ABHI\2025_things\Zsem-7 Project\dataset\train\Open_Eyes"

destination_folder_closed_eyes = r"F:\ABHI\2025_things\Zsem-7 Project\dataset\test\Closed_Eyes"
destination_folder_open_eyes = r"F:\ABHI\2025_things\Zsem-7 Project\dataset\test\Open_Eyes"

move_first_100_images(source_folder_closed_eyes, destination_folder_closed_eyes)
move_first_100_images(source_folder_open_eyes, destination_folder_open_eyes)