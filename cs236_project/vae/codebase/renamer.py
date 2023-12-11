import os

folder_path = "D:\CS236_combined_characters\pikachu_410"
prefix = "pikachu_"

for filename in os.listdir(folder_path):
    new_filename = prefix + filename
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))