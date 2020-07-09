from model.custom_layer import *
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras import initializers

def model(batch_size, k, alpha, lamda):
    
    initializer = initializers.GlorotNormal()

    ## Encoder
    inputs = Layers.Input(shape = [28, 28])
    reshape = Layers.Reshape([28, 28, 1])(inputs)
    flatten = Layers.Flatten()(inputs)
    conv_1 = Layers.Conv2D(6, kernel_size = 5, padding = 'valid',
                           activation = 'relu', kernel_initializer = initializer)(reshape)
    max_1 = Layers.MaxPool2D(pool_size = 2)(conv_1)
    conv_2 = Layers.Conv2D(16, kernel_size = 5, padding = 'valid',
                           activation = 'relu', kernel_initializer = initializer)(max_1)
    max_2 = Layers.MaxPool2D(pool_size = 2)(conv_2)
    conv_3 = Layers.Conv2D(60, kernel_size = 4, padding = 'valid',
                           activation = 'relu', kernel_initializer = initializer)(max_2)
    latent = Layers.Flatten()(conv_3)
    [non_anchor, anchor] = Discriminative(alpha = alpha, batch_size = batch_size, 
                                          k = k, name = 'discriminative')([flatten, latent])
    encoder = Model(inputs=[inputs], outputs=[non_anchor, anchor, conv_3, latent])

    ## Decoder
    decoder_inputs = Layers.Input(shape = [1, 1, 60])
    recon_1 = Layers.Conv2DTranspose(16, kernel_size = 4, strides = 1, padding="valid",
                                     activation="relu", kernel_initializer = initializer)(decoder_inputs)
    recon_2 = Layers.Conv2DTranspose(16, kernel_size = 2, strides = 2, padding="valid",
                                 activation="relu", kernel_initializer = initializer)(recon_1)
    recon_3 = Layers.Conv2DTranspose(6, kernel_size = 5, strides= 1, padding="valid",
                                 activation="relu", kernel_initializer = initializer)(recon_2)
    recon_4 = Layers.Conv2DTranspose(6, kernel_size = 2, strides= 2, padding="valid",
                                 activation="relu", kernel_initializer = initializer)(recon_3)
    recon_5 = Layers.Conv2DTranspose(1, kernel_size = 5, strides= 1, padding="valid",
                                 activation="relu", kernel_initializer = initializer)(recon_4)
    output = Layers.Reshape([28, 28])(recon_5)
    decoder = Model(inputs=[decoder_inputs], outputs=[output])

    ## Autoencoder
    _, _, codings, _ = encoder(inputs)
    reconstructions = decoder(codings)
    autoencoder = Model(inputs=[inputs], outputs=[reconstructions])
    
    ## Custom Loss
    def custom_loss(lamda):
        """" Wrapper function which calculates the loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # Latent loss
        latent_loss = tf.reduce_sum(non_anchor - anchor)
        
        def overall_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            reconstruction_loss = lamda * tf.norm(tf.subtract(y_true, y_pred))
            # Overall loss
            model_loss = reconstruction_loss + latent_loss
            return model_loss
        return overall_loss

    ## Compilation, adding loss and optimiser
    optimiser = Adam(learning_rate = 0.002)
    autoencoder.compile(loss = custom_loss(lamda = lamda), optimizer = optimiser)
    return autoencoder, encoder, decoder