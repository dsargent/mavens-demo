import mnist
import numpy as np
from keras.preprocessing.image import array_to_img
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model

def view_image(img):
    img1 = np.expand_dims(img, 2)
    pil_img = array_to_img(img1)
    pil_img.show()

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) # (60000, 784)
print(test_images.shape)  # (10000, 784)

# Build keras model
model = Sequential([
    Dense(64, activation='relu', name='Input'),
    Dense(64, activation='relu', name='Hidden1'),
    Dense(10, activation='softmax', name='Output'),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
# Train the model.
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
)

# Evaluate model
evaluation = model.evaluate(
    test_images,
    to_categorical(test_labels)
)

print(evaluation)

# # Save the model weights for later
model.save_weights('mnist_model.h5')
#
# Reload model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(784,)),
#     Dense(64, activation='relu'),
#     Dense(10, activation='softmax'),
# ])

# # Load the model's saved weights.
# model.load_weights('mnist_model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]

# Plot graph of model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

