
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from confusion import confusion_encoded,Confusion
import matplotlib.pyplot as plt
import seaborn as sns

RETRAIN_MODEL = False 

DATAPATH = '..\DATASET' # main dataset containning folder 

DATASET_NAME = 'plant village dataset'

DATASET_TWO = '..\DATASET\Black Pepper Leaf Blight and Yellow Mottle Virus'

folder_list = os.listdir(os.path.join(DATAPATH,DATASET_NAME))
print(f'Number of classes available in {DATASET_NAME}: {len(folder_list)}')

class_names = []
num_of_images = []

for file in folder_list:
    class_names.append(file)
    num_images = len(os.listdir(os.path.join(DATAPATH,DATASET_NAME,file)))
    num_of_images.append(num_images)
    print(f'Class: {file}, Number of images: {num_images}')

print(f'Total available images in {DATASET_NAME}: {sum(num_of_images)}')

folder_list_two = os.listdir(DATASET_TWO)
print(f'\nNumber of classes available in DATASET_TWO: {len(folder_list_two)}')

class_names_two = []
num_of_images_two = []

for file in folder_list_two:
    class_names_two.append(file)
    num_images_two = len(os.listdir(os.path.join(DATASET_TWO, file)))
    num_of_images_two.append(num_images_two)
    print(f'Class: {file}, Number of images: {num_images_two}')

print(f'Total available images in DATASET_TWO: {sum(num_of_images_two)}')
def get_image_paths_and_labels(dataset_path):
    folder_list = os.listdir(dataset_path)
    data = []

    for folder in folder_list:
        class_path = os.path.join(dataset_path, folder)
        if os.path.isdir(class_path):  
            file_list = os.listdir(class_path)
            for file in file_list:
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path): 
                    data.append((file_path, folder))
    
    return data


data_one = get_image_paths_and_labels(os.path.join(DATAPATH, DATASET_NAME))
data_two = get_image_paths_and_labels(DATASET_TWO)

combined_data = data_one + data_two

df = pd.DataFrame(combined_data, columns=['file_path', 'class_label'])

df = df.sample(frac=1, random_state=123).reset_index(drop=True)

print(f'Total images: {len(df)}')
print(df.head())

from image_loader import load_images

images, labels = load_images(df)

print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)


BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
TARGET_SIZE = (299, 299)

def prepare_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X))

    ds = ds.map(
        lambda x, y: (tf.image.resize(x, TARGET_SIZE), y),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = prepare_dataset(X_train, y_train, training=True)
test_ds = prepare_dataset(X_test, y_test, training=False)
