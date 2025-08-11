import numpy as np
from tensorflow.keras.preprocessing import image

def load_images(df, target_size=(128, 128)):
    num_images = len(df)
    images = np.zeros((num_images, target_size[0], target_size[1], 3), dtype=np.float32)
    labels = np.zeros(num_images, dtype=np.int32)
    
    class_names = df['class_label'].unique()
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    
    for i, row in df.iterrows():
        img_path = row['file_path']
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        images[i] = img_array / 255.0  # Normalize the image
        labels[i] = class_to_index[row['class_label']]
        
        if i % 100 == 0:
            print(f'Processed {i}/{num_images} images')
    
    return images, labels
