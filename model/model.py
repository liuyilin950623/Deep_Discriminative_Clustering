from model.custom_layer import *
from keras.models import Model

def model(hidlayer_size, codings_size, batch_size, alpha, lamda):

    ## Encoder
    inputs = Layers.Input(shape=[28, 28])
    flatten = Layers.Flatten()(inputs)
    hid_layer1 = Layers.Dense(hidlayer_size, activation="selu")(flatten)
    hid_layer2 = Layers.Dense(codings_size, activation="selu")(hid_layer1)
    [non_anchor, anchor] = Discriminative(alpha = alpha, name = 'discriminative')([flatten, hid_layer2])
    =encoder = Model(inputs=[inputs], outputs=[non_anchor, anchor, hid_layer2])

    ## Decoder
    decoder_inputs = Layers.Input(shape = [codings_size])
    hid_layer3 = Layers.Dense(hidlayer_size, activation="selu")(decoder_inputs)
    hid_layer4 = Layers.Dense(28 * 28, activation="selu")(hid_layer3)
    outputs = Layers.Reshape([28, 28])(hid_layer4)
    decoder = Model(inputs=[decoder_inputs], outputs=[outputs])

    ## Autoencoder
    _, _, codings = encoder(inputs)
    reconstructions = decoder(codings)
    autoencoder = Model(inputs=[inputs], outputs=[reconstructions])
    
    ## Custom Loss
    def custom_loss(lamda):
        """" Wrapper function which calculates the loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # Latent loss
        latent_loss = tf.reduce_sum(tf.subtract(non_anchor, anchor))
        
        def overall_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            reconstruction_loss = lamda * tf.norm(tf.subtract(y_true, y_pred))
            # Overall loss
            model_loss = latent_loss + reconstruction_loss
            return model_loss
        return overall_loss

    ## Compilation, adding loss and optimiser
    autoencoder.compile(loss = custom_loss(lamda = lamda), optimizer = "adam")
    
    return autoencoder, encoder