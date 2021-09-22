import tensorflow as tf


TF_MODEL_NAMES = {
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',

    # ResNets
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',

    # Mobile Nets
    'mobilenet': 'MobileNet',
    'mobilenetv2': 'MobileNetV2',

    # EfficientNets
    'efficientnetb0': 'EfficientNetB0',
    'efficientnetb1': 'EfficientNetB1',
    'efficientnetb2': 'EfficientNetB2',
    'efficientnetb3': 'EfficientNetB3',
    'efficientnetb4': 'EfficientNetB4',
    'efficientnetb5': 'EfficientNetB5',
    'efficientnetb6': 'EfficientNetB6',
    'efficientnetb7': 'EfficientNetB7'
}


TF_MODEL_MODULE = {
    'vgg16': 'vgg16',
    'vgg19': 'vgg19',

    # ResNets
    'resnet50': 'resnet',
    'resnet101': 'resnet',
    'resnet152': 'resnet',

    # Mobile Nets
    'mobilenet': 'mobilenet',
    'mobilenetv2': 'mobilenet',

    # EfficientNets
    'efficientnetb0': 'efficientnet',
    'efficientnetb1': 'efficientnet',
    'efficientnetb2': 'efficientnet',
    'efficientnetb3': 'efficientnet',
    'efficientnetb4': 'efficientnet',
    'efficientnetb5': 'efficientnet',
    'efficientnetb6': 'efficientnet',
    'efficientnetb7': 'efficientnet'
}


def get_preprocess_input(name):
    if not hasattr(tf.keras.applications, TF_MODEL_MODULE[name]):
        raise NotImplemented
    module = getattr(tf.keras.applications, TF_MODEL_MODULE[name])
    return module.preprocess_input


def get_tf_model(name, target_shape):
    if not hasattr(tf.keras.applications, TF_MODEL_NAMES[name]):
        raise NotImplemented
    model_fn = getattr(tf.keras.applications, TF_MODEL_NAMES[name])
    model = model_fn(include_top=False, weights='imagenet', input_shape=target_shape + (3,))
    return model


class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, enc_1, enc_2):
        return tf.math.abs(enc_1 - enc_2)


def get_siamese_net(name, target_shape):
    backbone = get_tf_model(name, target_shape)
    flatten = tf.keras.layers.Flatten()(backbone.output)
    dense1 = tf.keras.layers.Dense(512, kernel_regularizer='l2', activation="relu")(flatten)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(256, kernel_regularizer='l2', activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    output = tf.keras.layers.Dense(256)(dense2)
    embedding = tf.keras.Model(backbone.input, output, name="siamese_network")

    input_1 = tf.keras.layers.Input(name="input_1", shape=target_shape + (3,))
    input_2 = tf.keras.layers.Input(name="input_2", shape=target_shape + (3,))
    preprocess_input = get_preprocess_input(name)
    distance = DistanceLayer()(
        embedding(preprocess_input(input_1)),
        embedding(preprocess_input(input_2))
    )
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    siamese_network = tf.keras.Model(
        inputs=[input_1, input_2], outputs=output
    )
    return siamese_network
