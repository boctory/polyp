import tensorflow as tf
import numpy as np
import os
import cv2
import albumentations as A
from PIL import Image

class GIANADataLoader:
    def __init__(self, data_dir, img_size=(256, 256), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Augmentation pipeline
        self.aug_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.2),
        ])

    def _load_image(self, image_path, is_mask=False):
        # Read file contents
        img_raw = tf.io.read_file(image_path)
        
        # Decode image
        if is_mask:
            # Decode as RGB first, then take first channel
            img = tf.image.decode_bmp(img_raw, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32)
            # Take first channel and expand dims
            img = img[:, :, 0:1]
            img = tf.where(img > 0, 1.0, 0.0)  # Binarize mask
        else:
            # Decode as RGB
            img = tf.image.decode_bmp(img_raw, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
        
        return img

    def _augment(self, image, mask):
        augmented = self.aug_pipeline(image=image.numpy(), mask=mask.numpy())
        return tf.convert_to_tensor(augmented['image']), tf.convert_to_tensor(augmented['mask'])

    def create_dataset(self, image_paths, mask_paths, is_training=True):
        # Convert paths to string tensors
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.constant(image_paths, dtype=tf.string),
            tf.constant(mask_paths, dtype=tf.string)
        ))
        
        def load_data(img_path, mask_path):
            image = self._load_image(img_path, is_mask=False)
            mask = self._load_image(mask_path, is_mask=True)
            
            if is_training:
                image, mask = tf.py_function(
                    self._augment,
                    [image, mask],
                    [tf.float32, tf.float32]
                )
                image.set_shape([*self.img_size, 3])
                mask.set_shape([*self.img_size, 1])
            return image, mask

        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def get_train_test_datasets(self, train_split=0.8):
        # Implement dataset splitting logic here
        image_files = sorted([f for f in os.listdir(os.path.join(self.data_dir, 'images')) 
                            if f.endswith('.bmp')])
        mask_files = sorted([f for f in os.listdir(os.path.join(self.data_dir, 'masks')) 
                           if f.endswith('.bmp')])
        
        image_paths = [os.path.join(self.data_dir, 'images', f) for f in image_files]
        mask_paths = [os.path.join(self.data_dir, 'masks', f) for f in mask_files]
        
        # Shuffle and split
        indices = np.random.permutation(len(image_paths))
        split_idx = int(len(indices) * train_split)
        
        train_img_paths = [image_paths[i] for i in indices[:split_idx]]
        train_mask_paths = [mask_paths[i] for i in indices[:split_idx]]
        test_img_paths = [image_paths[i] for i in indices[split_idx:]]
        test_mask_paths = [mask_paths[i] for i in indices[split_idx:]]
        
        train_dataset = self.create_dataset(train_img_paths, train_mask_paths, is_training=True)
        test_dataset = self.create_dataset(test_img_paths, test_mask_paths, is_training=False)
        
        return train_dataset, test_dataset 