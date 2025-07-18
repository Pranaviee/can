import os
import shutil

root_dir = './data/train'  # Update this if your path is different

for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    images_folder = os.path.join(class_path, 'images')

    if os.path.isdir(images_folder):
        for img_file in os.listdir(images_folder):
            src = os.path.join(images_folder, img_file)
            dst = os.path.join(class_path, img_file)
            shutil.move(src, dst)

        # Remove the now-empty 'images' subfolder
        os.rmdir(images_folder)

print("✅ All images moved up to class folders.")
