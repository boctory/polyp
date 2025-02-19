import tensorflow as tf
import os
import argparse
from data_loader import GIANADataLoader
from models import EncoderDecoder, UNet, VGG16UNet
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def save_prediction_samples(model, dataset, epoch, log_dir):
    # Create directory for sample images
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get a batch of images
    for images, masks in dataset.take(1):
        predictions = model(images, training=False)
        
        # Take first 4 images from the batch
        for i in range(min(4, images.shape[0])):
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(images[i])
            plt.title('Input Image')
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i, :, :, 0], cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.title(f'Prediction (Dice: {dice_coefficient(masks[i], predictions[i]):.4f})')
            plt.axis('off')
            
            plt.savefig(os.path.join(samples_dir, f'epoch_{epoch}_sample_{i}.png'))
            plt.close()

def train_model(model, train_dataset, val_dataset, epochs, model_name):
    initial_learning_rate = 1e-3
    decay_steps = 1000
    decay_rate = 0.9
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    train_loss_metric = tf.keras.metrics.Mean()
    train_dice_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    val_dice_metric = tf.keras.metrics.Mean()
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'{model_name}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create CSV file for logging metrics
    metrics_file = os.path.join(log_dir, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('epoch,train_loss,train_dice,val_loss,val_dice,learning_rate\n')
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    @tf.function
    def train_step(images, masks):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(masks, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        dice = dice_coefficient(masks, predictions)
        return loss, dice
    
    @tf.function
    def val_step(images, masks):
        predictions = model(images, training=False)
        loss = loss_fn(masks, predictions)
        dice = dice_coefficient(masks, predictions)
        return loss, dice
    
    best_val_dice = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss_metric.reset_state()
        train_dice_metric.reset_state()
        
        for images, masks in train_dataset:
            loss, dice = train_step(images, masks)
            train_loss_metric.update_state(loss)
            train_dice_metric.update_state(dice)
        
        # Validation
        val_loss_metric.reset_state()
        val_dice_metric.reset_state()
        
        for images, masks in val_dataset:
            loss, dice = val_step(images, masks)
            val_loss_metric.update_state(loss)
            val_dice_metric.update_state(dice)
        
        # Get current learning rate
        current_lr = lr_schedule(optimizer.iterations)
        
        # Log metrics
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss_metric.result(), step=epoch)
            tf.summary.scalar('train_dice', train_dice_metric.result(), step=epoch)
            tf.summary.scalar('val_loss', val_loss_metric.result(), step=epoch)
            tf.summary.scalar('val_dice', val_dice_metric.result(), step=epoch)
            tf.summary.scalar('learning_rate', current_lr, step=epoch)
        
        # Save metrics to CSV
        with open(metrics_file, 'a') as f:
            f.write(f'{epoch},{train_loss_metric.result():.4f},{train_dice_metric.result():.4f},'
                   f'{val_loss_metric.result():.4f},{val_dice_metric.result():.4f},{current_lr:.6f}\n')
        
        # Save prediction samples every 5 epochs
        if epoch % 5 == 0:
            save_prediction_samples(model, val_dataset, epoch, log_dir)
        
        # Save best model and check early stopping
        if val_dice_metric.result() > best_val_dice:
            best_val_dice = val_dice_metric.result()
            model.save_weights(os.path.join(log_dir, 'best_model.weights.h5'))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss_metric.result():.4f}, Train Dice: {train_dice_metric.result():.4f}')
        print(f'Val Loss: {val_loss_metric.result():.4f}, Val Dice: {val_dice_metric.result():.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GIANA dataset directory')
    parser.add_argument('--model_type', type=str, default='unet', choices=['encoder_decoder', 'unet', 'vgg16_unet'],
                      help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    args = parser.parse_args()
    
    # Create data loader
    data_loader = GIANADataLoader(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    train_dataset, val_dataset = data_loader.get_train_test_datasets()
    
    # Create model
    input_shape = (args.img_size, args.img_size, 3)
    if args.model_type == 'encoder_decoder':
        model = EncoderDecoder(input_shape)
    elif args.model_type == 'unet':
        model = UNet(input_shape)
    else:
        model = VGG16UNet(input_shape)
    
    # Train model
    train_model(model, train_dataset, val_dataset, args.epochs, args.model_type)

if __name__ == '__main__':
    main() 