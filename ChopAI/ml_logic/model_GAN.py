############################################################################
# Imports
############################################################################

# Traditional imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Neural neworks import
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.train import Checkpoint, CheckpointManager

############################################################################
# Creating GAN model
############################################################################

# Define the standalone discriminator model
def def_discriminator(in_shape=(106,106,1)): # Input is an image of shape 106 x 106 in black and white
    """
    Returns a compiled discriminator model
    """

    # Define inputs
    inputs = Input(in_shape)

    # Block 1
    convolutional_layer_1 = Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape) (inputs)
    activation_1 = LeakyReLU(alpha=0.2) (convolutional_layer_1)
    dropout_1 = Dropout(0.5) (activation_1)

    # Block 2
    convolutional_layer_2 = Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape) (dropout_1)
    activation_2 = LeakyReLU(alpha=0.2) (convolutional_layer_2)
    dropout_2 = Dropout(0.5) (activation_2)

    # Classifier
    flattened_layer = Flatten()(dropout_2)
    batch_normalization_layer = BatchNormalization()(flattened_layer)
    output_discriminator = Dense(1, activation="sigmoid")(batch_normalization_layer)

    # Defining discrimnator model
    model = Model(inputs, outputs = output_discriminator)

    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model

#################################

# Define the standalone generator model
def def_generator(in_shape = 100): # Default latent dimension of 100
    """
    Returns a generator model WITHOUT compiling it
    """
    # Defining inputs
    inputs = Input(in_shape)

    # Block 1 - foundation for 53 x 53 images
    n_nodes_1 = 128 * 53 * 53
    dense_1 = Dense(n_nodes_1, input_dim=in_shape)(inputs)
    activation_1 = LeakyReLU(alpha=0.2)(dense_1)
    reshape_layer = Reshape( (53,53,128))(activation_1)

    # Block 2
    dense_2 = Dense(1024)(reshape_layer)
    conv2d_transposed_layer_1 = Conv2DTranspose(1024,(4,4), strides=(2,2), padding="same")(dense_2)

    # Block 3
    dense_3 = Dense(1024)(conv2d_transposed_layer_1)
    activation_2 = LeakyReLU(alpha=0.2)(dense_3)
    dense_4 = Dense(1024)(activation_2)
    conv2d_transposed_layer_1 = Conv2DTranspose(1,(7,7), padding="same", activation='sigmoid')(dense_4)

    # Generate model
    model = Model(inputs, outputs=conv2d_transposed_layer_1)

    return model

#################################

# Define the combined generator and discriminator model, for updating the generator
def def_gan(generator, discriminator):
    """
    Returns a compiled GAN model
    """
    # Make weights in the discriminator not trainable - train only generator weights
    discriminator.trainable = False

    # Instantiate GAN model
    model = Sequential()

    # Add generator and discrimnator
    model.add(generator)
    model.add(discriminator)

    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt)

    return model

############################################################################
# Training GAN model
############################################################################

# Intermediary function - Select a sample of 'real' images
def generate_real_samples(dataset, n_samples):
    """
    Takes as input cleaned dataset and a number of samples to be generated
    Returns a random sample of n_samples images and their corresponding label (=1 because these are real images)
    """
    # Choose random instances (i.e., randomly select n_samples indexes from dataset)
    iX = np.random.randint(0, dataset.shape[0], n_samples)
    # Loading corresponding images
    X = dataset[iX]
    # Creating corresponding 'Real' (=1) labels
    y = np.ones((n_samples, 1))

    return X, y

#################################

# Intermediary function - Generate points in latent space as input for the generator, following Gaussian distributed variable
def generate_latent_points(latent_dim, n_samples):
    # Generate (latent dimension x n_samples) array of random values taken from x axis of Normal Distribution
    x_input = np.random.randn(latent_dim*n_samples)
    # Reshape to have num_samples entries for each one with latent_dimension values
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

#################################

# Intermediary function - Use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs
    X = generator.predict(x_input)
    # Generate corresponding 'Fake' (=0) class labels
    y = np.zeros((n_samples, 1))
    return X, y

#################################

# Custom function to display accuracy of the Discriminator in predicting correctly both real and fake music samples
def show_current_discriminator_accuracy(discriminator_model, generator_model, pixels, latent_dim = 100, num_samples_to_test = 100):

    #generate real music samples
    X_real, y_real = generate_real_samples(pixels, num_samples_to_test)

    #generate fake music samples
    X_fake, y_fake = generate_fake_samples(generator_model, latent_dim, num_samples_to_test)

    #evaluate the accuracy of the discriminator on real music samples
    _, accuracy_on_real = discriminator_model.evaluate(X_real, y_real, verbose=0)
    #evaluate the accuracy of the discriminator on fake music samples
    _, accuracy_on_fake = discriminator_model.evaluate(X_fake, y_fake, verbose=0)

    #print results
    print("   Current accuracy of the discriminator on real music samples:", round(accuracy_on_real*100,3),"%")
    print("   Current accuracy of the discriminator on fake music samples:", round(accuracy_on_fake*100,3),"% \n")

    return accuracy_on_real, accuracy_on_fake

#################################

# Train the GAN model
def train(generator_model, discriminator_model, gan_model, dataset, latent_dim = 100, n_epochs=250, n_batch=15):
    bat_per_epo = int(dataset.shape[0] / n_batch) # Number of batches per epoch
    half_batch = int(n_batch / 2) # Half the number of batches

    # Set-up checkpoints for GAN model (to save weights at each epoch)
    checkpoint_dir = 'checkpoints_GAN'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = Checkpoint(generator = generator_model, discriminator = discriminator_model, GAN = gan_model)

    ckpt_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=30)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


    # Storing results along the epochs
    # Dataframe containing loss and accuracy for each epoch
    discriminator_info_per_epoch = pd.DataFrame(columns=['loss_discriminator_on_real_music', 'loss_generator_on_fake_music', 'accuracy_on_real', 'accuracy_on_fake'])

    accuracy_on_fake = 0
    accuracy_on_real = 0

    # For each epoch
    for i in range(n_epochs): # Enumerate epochs
        print("\n\n epoch:", i)

        # For each batch
        for j in range(bat_per_epo):

            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = discriminator_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(generator_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = discriminator_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

        # save checkpoints results for the epoch
        ckpt_manager.save()

        # print the current accuracy obtained in classifing correctly both real music samples and fake music samples (the ones generated by the generator)
        accuracy_on_real, accuracy_on_fake = show_current_discriminator_accuracy(discriminator_model, generator_model, dataset, latent_dim)
        discriminator_info_per_epoch = discriminator_info_per_epoch.append({'loss_discriminator_on_real_music':  d_loss1, 'loss_generator_on_fake_music': g_loss, 'accuracy_on_real': accuracy_on_real, 'accuracy_on_fake': accuracy_on_fake}, ignore_index=True)

    return gan_model, discriminator_info_per_epoch
