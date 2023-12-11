import os
from PIL import Image

def resize_images(input_folder, output_folder, target_resolution):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            with Image.open(input_path) as img:
                rgb_img = img.convert('RGB')
                # Resize the image
                resized_img = rgb_img.resize(target_resolution, Image.ANTIALIAS)

                # Save the resized image
                resized_img.save(output_path)

if __name__ == "__main__":
    input_folder = "D:\\CS236_project_data_pikachu_tmp\\Pikachu"
    output_folder = "D:\\CS236_project_data_pikachu\\Pikachu"
    target_resolution = (128, 128)  # Set your target resolution

    resize_images(input_folder, output_folder, target_resolution)