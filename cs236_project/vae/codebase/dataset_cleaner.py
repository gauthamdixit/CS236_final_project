from PIL import Image
import os
import hashlib
from torchvision import datasets, transforms

def image_hash(image_path):
    # Open image and convert it to grayscale
    img = Image.open(image_path).convert("L")
    
    # Resize the image to a small size for faster hashing (e.g., 8x8 pixels)
    img = img.resize((8, 8), Image.ANTIALIAS)
    
    # Calculate the hash of the image
    pixels = list(img.getdata())
    avg_pixel = sum(pixels) / len(pixels)
    hash_str = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
    
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

def remove_duplicate_images(folder_path):
    image_hashes = set()
    
    # Iterate through all images in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            
            # Calculate the hash of the image
            hash_value = image_hash(image_path)
            
            # Check if the hash is already encountered (duplicate)
            if hash_value in image_hashes:
                print(f"Removing duplicate image: {image_path}")
                os.remove(image_path)
            else:
                # Add the hash to the set for future comparison
                image_hashes.add(hash_value)




def preprocess_and_save(dataset_path, save_path):
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomApply([transforms.Compose([transforms.GaussianBlur(kernel_size=3)])], p=0.2),
    ])

    # Create a new dataset instance without transformations
    original_dataset = datasets.ImageFolder(root=dataset_path)

    # Create a new directory to save preprocessed images
    os.makedirs(save_path, exist_ok=True)

    # Iterate through the dataset, apply transformations, and save the preprocessed images
    for idx, (image_path, label) in enumerate(original_dataset.imgs):
        image = Image.open(image_path)
        preprocessed_image = preprocess(image)

        # Save the preprocessed image to the new directory
        save_filename = os.path.join(save_path, f"preprocessed_{idx}.png")
        preprocessed_image.save(save_filename)

    print(f"Preprocessed images saved to {save_path}")

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
    final_path = "D:\CS236_project_data_pikachu\Pikachu"
    dataset_path = "D:\CS236_project_data_pikachu_tmp"
    target_resolution = (128, 128)  # Set your target resolution

    resize_images(dataset_path, dataset_path, target_resolution)
    remove_duplicate_images(dataset_path)
    preprocess_and_save(dataset_path, final_path)
    