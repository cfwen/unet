import keras
from keras.layers import Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)

    pool5 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool5)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    conv6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    merge6 = concatenate([drop4, conv6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = concatenate([conv3, conv7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up8)
    merge8 = concatenate([conv2, conv8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up9)
    merge9 = concatenate([conv1, conv9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

    conv10 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv10)

    model = keras.Model(inputs=inputs, outputs=conv10)

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
