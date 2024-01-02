# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Load Cityscapes dataset
# (Assuming you have the dataset and appropriate directories set up)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path/to/cityscapes/train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'path/to/cityscapes/val',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Build a U-Net model with a MobileNetV2 encoder
def create_unet_model(input_shape=(256, 256, 3), num_classes=20):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)
    
    # Freeze MobileNetV2 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Use skip connections for the decoder
    skip_connections = [
        base_model.get_layer('block_1_expand_relu').output,
        base_model.get_layer('block_3_expand_relu').output,
        base_model.get_layer('block_6_expand_relu').output,
        base_model.get_layer('block_13_expand_relu').output,
        base_model.get_layer('block_16_expand_relu').output
    ]
    
    # Decoder
    x = base_model.output
    for skip in reversed(skip_connections):
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# Create and compile the U-Net model
model = create_unet_model()
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks (e.g., model checkpoint)
checkpoint = ModelCheckpoint('unet_cityscapes.h5', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc * 100:.2f}%')
