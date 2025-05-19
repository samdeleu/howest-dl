import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


# Create occlusions in the images (e.g., masking a central square)
def add_occlusion(images, occlusion_size=14):
    occluded_images = images.copy()
    h, w = images.shape[1], images.shape[2]
    start_h, start_w = (h - occlusion_size) // 2, (w - occlusion_size) // 2
    occluded_images[:, start_h:start_h + occlusion_size, start_w:start_w + occlusion_size, :] = 0
    return occluded_images

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data to (num_samples, height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)



x_train_occluded = add_occlusion(x_train)
x_test_occluded = add_occlusion(x_test)

# Build the autoencoder model
input_img = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Compile the autoencoder
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train_occluded, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test_occluded, x_test))

# Reconstruct the occluded images
reconstructed_images = autoencoder.predict(x_test_occluded)

# Plot the results
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Display occluded images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_occluded[i].reshape(28, 28), cmap='gray')
    plt.title("Occluded")
    plt.axis("off")

    # Display reconstructed images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()