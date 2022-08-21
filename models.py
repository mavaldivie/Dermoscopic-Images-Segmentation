from keras import Input, Model
from keras import layers
from keras.optimizers import Adam

def create_model():
    # TODO: Use Conv2DTranspose and check results
    input_image = Input(shape=(380, 380, 3))

    down1 = layers.Conv2D(64, (3, 3), activation='relu')(input_image) # (378, 378, 64)
    down1 = layers.Conv2D(64, (3, 3), activation='relu')(down1) # (376, 376, 64)

    down2 = layers.MaxPooling2D((2, 2))(down1) # (188, 188, 128)
    down2 = layers.Conv2D(128, (3, 3), activation='relu')(down2) # (186, 186, 128)
    down2 = layers.Conv2D(128, (3, 3), activation='relu')(down2) # (184, 184, 128)

    down3 = layers.MaxPooling2D((2, 2))(down2) # (92, 92, 256)
    down3 = layers.Conv2D(256, (3, 3), activation='relu')(down3) # (90, 90, 256)
    down3 = layers.Conv2D(256, (3, 3), activation='relu')(down3) # (88, 88, 256)

    down4 = layers.MaxPooling2D((2, 2))(down3) # (44, 44, 512)
    down4 = layers.Conv2D(512, (3, 3), activation='relu')(down4) # (42, 42, 512)
    down4 = layers.Conv2D(512, (3, 3), activation='relu')(down4) # (40, 40, 512)

    drop1 = layers.Dropout(0.5)(down4) # (40, 40, 512)
    mid = layers.MaxPooling2D((2, 2))(drop1) # (20, 20, 1024)
    mid = layers.Conv2D(1024, (3, 3), activation='relu')(mid) # (18, 18, 1024)
    mid = layers.Conv2D(1024, (3, 3), activation='relu')(mid) # (16, 16, 1024)
    drop2 = layers.Dropout(0.5)(mid) # (16, 16, 512)

    crop1 = layers.Cropping2D(4)(down4) # (32, 32, 512)
    up1 = layers.UpSampling2D((2, 2))(drop2) # (32, 32, 512)
    up1 = layers.concatenate([crop1, up1]) # (32, 32, 1024)
    up1 = layers.Conv2D(512, (3, 3), activation='relu')(up1) # (30, 30, 512)
    up1 = layers.Conv2D(512, (3, 3), activation='relu')(up1) # (28, 28, 512)

    crop2 = layers.Cropping2D(16)(down3) # (56, 56, 256)
    up2 = layers.UpSampling2D((2, 2))(up1) # (56, 56, 256)
    up2 = layers.concatenate([crop2, up2]) # (56, 56, 512)
    up2 = layers.Conv2D(256, (3, 3), activation='relu')(up2) # (54, 54, 256)
    up2 = layers.Conv2D(256, (3, 3), activation='relu')(up2) # (52, 52, 256)

    crop3 = layers.Cropping2D(40)(down2) # (104, 104, 128)
    up3 = layers.UpSampling2D((2, 2))(up2) # (104, 104, 128)
    up3 = layers.concatenate([crop3, up3]) # (104, 104, 256)
    up3 = layers.Conv2D(128, (3, 3), activation='relu')(up3) # (102, 102, 128)
    up3 = layers.Conv2D(128, (3, 3), activation='relu')(up3) # (100, 100, 128)

    crop4 = layers.Cropping2D(88)(down1) # (200, 200, 64)
    up4 = layers.UpSampling2D((2, 2))(up3) # (200, 200, 64)
    up4 = layers.concatenate([crop4, up4]) # (200, 200, 128)
    up4 = layers.Conv2D(64, (3, 3), activation='relu')(up4) # (198, 198, 64)
    up4 = layers.Conv2D(64, (3, 3), activation='relu')(up4) # (196, 196, 64)

    model = layers.Conv2D(1, (1,1), activation='sigmoid')(up4)
    model = Model(inputs=input_image, outputs=model)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4), metrics=['acc'])
    return model

def padding_model():
    inputs = Input(shape=(384, 384, 3))
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)                                   # (384, 384, 64)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)                                    # (384, 384, 64)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)                                                          # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)                                   # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)                                   # (192, 192, 128)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)                                                          # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)                                   # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)                                   # (96, 96, 256)
    drop1 = layers.Dropout(0.5)(conv3)                                                                            # (96, 96, 256)

    mid = layers.MaxPooling2D(pool_size=(2, 2))(drop1)                                                            # (48, 48, 512)
    mid = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(mid)                                       # (48, 48, 512)
    mid = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(mid)                                       # (48, 48, 512)
    drop2 = layers.Dropout(0.5)(mid)                                                                              # (48, 48, 512)

    up1 = layers.UpSampling2D(size = (2,2))(drop2)                                                                # (96, 96, 256)
    up1 = layers.concatenate([conv3,up1])                                                                         # (96, 96, 512)
    up1 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(up1)                                       # (96, 96, 256)
    up1 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(up1)                                       # (96, 96, 256)

    up2 = layers.UpSampling2D(size = (2,2))(up1)                                                                  # (192, 192, 128)
    up2 = layers.concatenate([conv2,up2])                                                                         # (192, 192, 256)
    up2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(up2)                                       # (192, 192, 128)
    up2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(up2)                                       # (192, 192, 128)

    up3 = layers.UpSampling2D(size = (2,2))(up2)                                                                  # (384, 384, 64)
    up3 = layers.concatenate([conv1,up3])                                                                         # (384, 384, 128)
    up3 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(up3)                                        # (384, 384, 64)
    up3 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(up3)                                        # (384, 384, 64)

    model = layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(up3)                                       # (384, 384, 2)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)                                                    # (384, 384, 1)

    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['acc'])
    return model


def shorter_model():
    # TODO: Use Conv2DTranspose and check results
    input_image = Input(shape=(380, 380, 3))

    down1 = layers.Conv2D(64, (3, 3), activation='relu')(input_image) # (378, 378, 64)
    down1 = layers.Conv2D(64, (3, 3), activation='relu')(down1) # (376, 376, 64)

    down2 = layers.MaxPooling2D((2, 2))(down1) # (188, 188, 128)
    down2 = layers.Conv2D(128, (3, 3), activation='relu')(down2) # (186, 186, 128)
    down2 = layers.Conv2D(128, (3, 3), activation='relu')(down2) # (184, 184, 128)

    down3 = layers.MaxPooling2D((2, 2))(down2) # (92, 92, 256)
    down3 = layers.Conv2D(256, (3, 3), activation='relu')(down3) # (90, 90, 256)
    down3 = layers.Conv2D(256, (3, 3), activation='relu')(down3) # (88, 88, 256)

    drop1 = layers.Dropout(0.5)(down3) # (88, 88, 512)
    down4 = layers.MaxPooling2D((2, 2))(drop1) # (44, 44, 512)
    down4 = layers.Conv2D(512, (3, 3), activation='relu')(down4) # (42, 42, 512)
    down4 = layers.Conv2D(512, (3, 3), activation='relu')(down4) # (40, 40, 512)
    drop2 = layers.Dropout(0.5)(down4) # (40, 40, 512)

    crop2 = layers.Cropping2D(4)(down3) # (80, 80, 256)
    up2 = layers.UpSampling2D((2, 2))(drop2) # (80, 80, 256)
    up2 = layers.concatenate([crop2, up2]) # (80, 80, 512)
    up2 = layers.Conv2D(256, (3, 3), activation='relu')(up2) # (78, 78, 256)
    up2 = layers.Conv2D(256, (3, 3), activation='relu')(up2) # (76, 76, 256)

    crop3 = layers.Cropping2D(16)(down2) # (152, 152, 128)
    up3 = layers.UpSampling2D((2, 2))(up2) # (152, 152, 128)
    up3 = layers.concatenate([crop3, up3]) # (152, 152, 256)
    up3 = layers.Conv2D(128, (3, 3), activation='relu')(up3) # (150, 150, 128)
    up3 = layers.Conv2D(128, (3, 3), activation='relu')(up3) # (148, 148, 128)

    crop4 = layers.Cropping2D(40)(down1) # (296, 296, 64)
    up4 = layers.UpSampling2D((2, 2))(up3) # (296, 296, 64)
    up4 = layers.concatenate([crop4, up4]) # (296, 296, 128)
    up4 = layers.Conv2D(64, (3, 3), activation='relu')(up4) # (294, 294, 64)
    up4 = layers.Conv2D(64, (3, 3), activation='relu')(up4) # (292, 292, 64)

    model = layers.Conv2D(1, (1,1), activation='sigmoid')(up4)
    model = Model(inputs=input_image, outputs=model)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4), metrics=['acc'])
    return model