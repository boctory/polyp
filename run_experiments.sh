#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Run experiments for each model
echo "Running Encoder-Decoder experiment..."
python train.py --data_dir data --model_type encoder_decoder --epochs 50 --batch_size 16 --img_size 256

echo "Running U-Net experiment..."
python train.py --data_dir data --model_type unet --epochs 50 --batch_size 16 --img_size 256

echo "Running VGG16-UNet experiment..."
python train.py --data_dir data --model_type vgg16_unet --epochs 50 --batch_size 16 --img_size 256

echo "All experiments completed. Check logs directory for results." 