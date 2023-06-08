import tensorflow as tf
import os
import sys 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

base_dir = 'dataset'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_real_money_dir = os.path.join(train_dir, 'real')
train_fake_money_dir = os.path.join(train_dir, 'fake')

test_real_money_dir = os.path.join(test_dir, 'real')
test_fake_money_dir = os.path.join(test_dir, 'fake')

BATCH_SIZE = 4
TRAIN_SIZE = 90
TEST_SIZE = 10

model = tf.keras.models.Sequential(
    [
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 

    tf.keras.layers.Dense(1, activation='sigmoid')  
]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen  = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

test_generator =  test_datagen.flow_from_directory(test_dir,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

history = model.fit(train_generator,
                    steps_per_epoch=TRAIN_SIZE / BATCH_SIZE,
                    epochs=15, 
                    validation_data=test_generator, 
                    validation_steps=TEST_SIZE / BATCH_SIZE, 
                    verbose=2)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) 


# plot accuracy with matplotlib
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Accuracy in training and validation')
plt.figure()

# plot loss with matplotlib
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Loss in training and validation')

# prediction on the uploaded image
img = image.load_img('./sample.jpg', target_size=(150, 150)) # let's use load_img to scale it 

# scaling process
x = image.img_to_array(img)
x /= 255 
x = np.expand_dims(x, axis=0)
# flatten the output
images = np.vstack([x])

# prediction!
classes = model.predict(images, batch_size=10)

print(classes[0])

if classes[0] > 0.5:
  print("real!")
else:
  print("fake!")