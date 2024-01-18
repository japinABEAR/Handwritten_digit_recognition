# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Augment data
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)

# Build model
model = models.Sequential()
# ... (rest of the model architecture)
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # Added L2 regularization
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Define model checkpoint callback
checkpoint = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Define learning rate scheduler callback
def lr_scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (5 - epoch))

lr_scheduler_cb = callbacks.LearningRateScheduler(lr_scheduler)

model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=15,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, checkpoint, lr_scheduler_cb])

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save model
model.save('enhanced_digit_model.h5')
