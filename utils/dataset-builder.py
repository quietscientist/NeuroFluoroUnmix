import os
import numpy as np
import cv2
import scipy.io
from PIL import Image
import albumentations as A
from matplotlib import pyplot as plt
import re

########## HELPER FUNCTIONS FOR VISUALIZING IMAGES (JPG) ##########

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def data_to_jpg_herma(data, version):
    converted_imgs = []
    for i in range(data.shape[2]):
        filename1 = "data/data" + str(version) + "_blk_" + str(i) + '.jpg'
        filename2 = "data/data" + str(version) + "_blu_" + str(i) + '.jpg'
        converted_imgs.append(filename1)
        converted_imgs.append(filename2)
        pil_image1 = Image.fromarray(data[:, :, i, :3].astype(np.uint8))
        pil_image2 = Image.fromarray(data[:, :, i, 3:].astype(np.uint8))
        pil_image1.save(filename1)
        pil_image2.save(filename2)
    return converted_imgs

def data_to_jpg_YA(data, version):
    converted_imgs = []
    for i in range(data.shape[2]):
        filename1 = "data/data" + str(version) + "_lrv_" + str(i) + '.jpg'
        converted_imgs.append(filename1)
        pil_image1 = Image.fromarray(data[:, :, i, [4, 3, 0]].astype(np.uint8))
        pil_image1.save(filename1)
    return converted_imgs

def transformed_data_to_jpg(images):
    data = []
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.GaussNoise(),
        A.RGBShift(),
        A.RandomBrightnessContrast(),
    ])

    # Read an image with OpenCV and convert it to the RGB colorspace and/or save to .npz
    for filename in images:
        # Augment an image (JPG) w/ Albumentations
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)

        transformed_image = transformed["image"]
        transformed_image = Image.fromarray(transformed_image)
        transformed_image.save("transformed_" + filename)
        # visualize(transformed_image)

###################################################################

# Function to apply augmentations to the image
def apply_augmentations_herma(img):
    # Define the percentages for each augmentation
    noise_pct = 0.2
    shift_pct = 0.2
    brightness_pct = 0.2
    
    # Add noise
    if np.random.rand() < noise_pct:
        noise = np.random.randint(0, 20, img.shape)
        img = img + noise
        
    # Apply RGB shift
    if np.random.rand() < shift_pct:
        shift = np.random.randint(0, 10, size=1)
        channel = np.random.randint(0, 3, size=1)
        img[:, :, channel] = img[:, :, channel] + shift
        
    # Adjust brightness
    if np.random.rand() < brightness_pct:
        brightness_factor = np.random.uniform(0.9, 1.1)
        img = img * brightness_factor
    
    return img

def apply_augmentations_YA(img):
    # Define the percentages for each augmentation
    noise_pct = 0.2
    shift_pct = 0.2
    brightness_pct = 0.2
    
    # Add noise
    if np.random.rand() < noise_pct:
        noise = np.random.randint(0, 20, img.shape)
        img = img + noise
        
    # Apply RGB shift
    if np.random.rand() < shift_pct:
        YA_RGB_idx = [4, 3, 0]
        shift = np.random.randint(0, 10, size=1)
        channel = np.random.randint(0, 3, size=1)
        img[:, :, YA_RGB_idx[channel[0]]] = img[:, :, YA_RGB_idx[channel[0]]] + shift
        
    # Adjust brightness
    if np.random.rand() < brightness_pct:
        brightness_factor = np.random.uniform(0.9, 1.1)
        img = img * brightness_factor
    
    return img

def voxelize_images(images, voxel_size, voxel_thickness):
    # Calculate the number of voxels in each dimension
    num_voxels_x = images.shape[0] // voxel_size
    num_voxels_y = images.shape[1] // voxel_size
    num_voxels_z = images.shape[2] // voxel_thickness

    # Initialize the output voxel array
    voxels = []

    # Iterate over the voxels and fill them with the corresponding pixel values
    for i in range(num_voxels_x):
        for j in range(num_voxels_y):
            for k in range(num_voxels_z):
                voxels.append(images[i*voxel_size:(i+1)*voxel_size, j*voxel_size:(j+1)*voxel_size, k*voxel_thickness:(k+1)*voxel_thickness, :])
    
    # print(np.array(voxels).shape)         # [voxel_num, x, y, num_layer, channel]
    return np.array(voxels)

# Target directory to search
target_dir = '../'      # Make sure data directories in same folder as this file

version = 0             # For jpg file differentiation
converted_imgs = []     # For for albumentations transformations
all_data = []
# Loop through all files in the directory and its subdirectories
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith('.nd2'):
            # TODO: where are the nd2 files?
            pass
        elif file == "data.mat": 
            mat_data = scipy.io.loadmat(os.path.join(root, file))
            data = mat_data['data']
            for i in range(data.shape[2]):
                # Apply augmentations to each image layer
                data[:, :, i, :] = apply_augmentations_herma(data[:, :, i, :])
            # Voxelize image into 3D blocks of data
            data = voxelize_images(data, 64, 8)        # [num_voxels, x=64, y=64, z=8, num_channels={5, 6}]
            all_data.append({"image": data})
            # # Uncomment to generate RBG JPG files of data images
            # converted_imgs = converted_imgs + data_to_jpg_herma(data, version)
            # version += 1
        elif bool(re.search('^[0-9]+_YA...\.mat', file)):
            mat_data = scipy.io.loadmat(os.path.join(root, file))
            data = mat_data['data']
            for i in range(data.shape[2]):
                # Normalize RGB values to 0-255
                data[:, :, i, [4, 3, 0]] = data[:, :, i, [4, 3, 0]] / np.max(data[:, :, i, [4, 3, 0]])
                data[:, :, i, [4, 3, 0]] = data[:, :, i, [4, 3, 0]] * 255
                # Apply augmentations to each image layer
                data[:, :, i, :] = apply_augmentations_YA(data[:, :, i, :])
            # Voxelize image into 3D blocks of data
            data = voxelize_images(data, 64, 8)        # [num_voxels, x=64, y=64, z=8, num_channels={5, 6}]
            all_data.append({"image": data})
            # # Uncomment to generate RBG JPG files of data images
            # converted_imgs = converted_imgs + data_to_jpg_YA(data, version)
            # version += 1

# Uncomment to transform via albumentations, requires jpg format
# transformed_data_to_jpg(converted_imgs)

# # Convert the data array to numpy arrays
all_data = np.array(all_data)

# Save the data array to an .npz file
np.savez_compressed('processed_data.npz', data=all_data)