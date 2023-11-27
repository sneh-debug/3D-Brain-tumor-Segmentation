#model.py
from functools import partial
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
#from keras.engine import Model
from keras.models import Model
from keras.optimizers import Adam
#from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from unet import create_convolution_block, concatenate
from metrics import weighted_dice_coefficient_loss,dice_coefficient, dice_BCE_loss
#from metrics import focal_tversky_loss
#from metrics import categorical_focal_loss

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=dice_BCE_loss, activation_name="sigmoid",metrics=dice_coefficient):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    print("n_base_filters",n_base_filters)
    print("depth",depth)
    
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        if level_number <(depth-1):
            n_level_filters = (2**level_number) * n_base_filters
            
        else:
           n_level_filters=200 #default is 200, experimented with 240 with parameters: 9.99 M
        level_filters.append(n_level_filters)
        #print("n_level_filters",n_level_filters)
        
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
            in_conv = Conv3D(n_level_filters, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(in_conv)
            in_conv = create_DepthwiseConv3D_block(in_conv, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters)
            in_conv = Conv3D(n_level_filters, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(in_conv)
            in_conv = create_DepthwiseConv3D_block(in_conv, n_level_filters,strides=(2, 2, 2))
            #in_conv = create_convolution_block(in_conv, n_level_filters, strides=(2, 2, 2))
        
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    
    for level_number in range(depth - 2, -1, -1):
        #print("level_filters",level_filters[level_number])
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    """
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    #loss='binary_crossentropy' is giving good result
    #in compile [loss_function(zero_weight=0.11, one_weight=0.89)]
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics = metrics)
    model.summary()
    plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def relu_advanced(x):
    return K.relu(x, max_value=6)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

#from DepthwiseConv3D import DepthwiseConv3D
def create_DepthwiseConv3D_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    print("n_filters",n_filters)
    layer = Conv3D(n_filters, kernel_size=(1,1,1), padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        #layer= Activation(relu_advanced)(layer)
        return LeakyReLU(alpha=0.01)(layer)
    else:
        layer= activation()(layer)
    
    #depthwise convolution
    layer = DepthwiseConv3D(n_filters, depth_multiplier=1)(layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        #layer= Activation(relu_advanced)(layer)
        return Activation('LeakyReLU(alpha=0.01)')(layer)
    else:
        layer= activation()(layer)
    
    #pointwise
    layer = Conv3D(n_filters, kernel_size=(1,1,1), padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        #layer= Activation(relu_advanced)(layer)
        return Activation('LeakyReLU(alpha=0.01)')(layer)
    else:
        layer= activation()(layer)




