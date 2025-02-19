import tensorflow as tf
from tensorflow.keras import layers, Model

class EncoderDecoder(Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(EncoderDecoder, self).__init__()
        
        # Encoder
        self.encoder = [
            self._conv_block(64, 3, dropout_rate=0.1),
            self._conv_block(128, 3, dropout_rate=0.1),
            self._conv_block(256, 3, dropout_rate=0.2),
            self._conv_block(512, 3, dropout_rate=0.2),
        ]
        
        # Decoder
        self.decoder = [
            self._upconv_block(512, dropout_rate=0.2),
            self._upconv_block(256, dropout_rate=0.2),
            self._upconv_block(128, dropout_rate=0.1),
            self._upconv_block(64, dropout_rate=0.1),
        ]
        
        self.final_conv = layers.Conv2D(1, 1, activation='sigmoid')
        
    def _conv_block(self, filters, kernel_size, dropout_rate=0.0):
        return tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),
            layers.MaxPooling2D()
        ])
    
    def _upconv_block(self, filters, dropout_rate=0.0):
        return tf.keras.Sequential([
            layers.Conv2DTranspose(filters, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate)
        ])
    
    def call(self, x):
        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            
        return self.final_conv(x)

class UNet(Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = [
            self._conv_block(64, 3),
            self._conv_block(128, 3),
            self._conv_block(256, 3),
            self._conv_block(512, 3),
        ]
        
        # Decoder with skip connections
        self.decoder = [
            self._upconv_block(512),
            self._upconv_block(256),
            self._upconv_block(128),
            self._upconv_block(64),
        ]
        
        self.final_conv = layers.Conv2D(1, 1, activation='sigmoid')
        
    def _conv_block(self, filters, kernel_size):
        return tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D()
        ])
    
    def _upconv_block(self, filters):
        return tf.keras.Sequential([
            layers.Conv2DTranspose(filters, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
    
    def call(self, x):
        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            if i < len(skip_connections):
                x = layers.Concatenate()([x, skip_connections[i]])
        
        return self.final_conv(x)

class VGG16UNet(Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(VGG16UNet, self).__init__()
        
        # Load pretrained VGG16 as encoder
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Extract intermediate layers for skip connections
        layer_names = [
            'block1_conv2',  # 64
            'block2_conv2',  # 128
            'block3_conv3',  # 256
            'block4_conv3',  # 512
            'block5_conv3',  # 512
        ]
        
        self.encoder_outputs = [base_model.get_layer(name).output for name in layer_names]
        self.encoder = Model(inputs=base_model.input, outputs=self.encoder_outputs)
        
        # Freeze encoder weights
        self.encoder.trainable = False
        
        # Decoder
        self.decoder = [
            self._upconv_block(512),
            self._upconv_block(256),
            self._upconv_block(128),
            self._upconv_block(64),
        ]
        
        self.final_conv = layers.Conv2D(1, 1, activation='sigmoid')
    
    def _upconv_block(self, filters):
        return tf.keras.Sequential([
            layers.Conv2DTranspose(filters, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
    
    def call(self, x):
        # Encoder
        skip_connections = self.encoder(x)
        x = skip_connections[-1]
        skip_connections = skip_connections[:-1][::-1]
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            if i < len(skip_connections):
                x = layers.Concatenate()([x, skip_connections[i]])
        
        return self.final_conv(x) 