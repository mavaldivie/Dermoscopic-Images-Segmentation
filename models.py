from keras import Input, Model
from keras import layers
from keras.optimizers import Adam
from keras import regularizers

def double_convolution_block(model, channels, activation='relu', padding=None):
    if padding:
        model = layers.Conv2D(channels, (3, 3), activation = activation, padding = padding)(model) 
        model = layers.Conv2D(channels, (3, 3), activation = activation, padding = padding)(model)
    else:
        model = layers.Conv2D(channels, (3, 3), activation = activation)(model) 
        model = layers.Conv2D(channels, (3, 3), activation = activation)(model)
    return model

def contracting_block(model, channels, activation='relu', padding=None):
    model = layers.MaxPooling2D(pool_size=(2, 2))(model)
    model = double_convolution_block(model, channels, activation, padding)
    return model

def expanding_block(model, residual, channels, activation='relu', padding=None):
        model = layers.Conv2DTranspose(channels, (2,2), strides=(2,2), padding='same')(model) 
        model = layers.concatenate([model, residual])
        model = double_convolution_block(model, channels, activation, padding)
        return model

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

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  
    drop1 = layers.Dropout(0.5)(pool1)                                                         # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(drop1)                                   # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)                                   # (192, 192, 128)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)   
    drop2 = layers.Dropout(0.5)(pool2)                                                          # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(drop2)                                   # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)                                   # (96, 96, 256)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)    
    drop3 = layers.Dropout(0.5)(pool3)                                                         # (96, 96, 256)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(drop3)                                   # (96, 96, 256)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)  

    mid = layers.MaxPooling2D(pool_size=(2, 2))(conv4)                                                            # (48, 48, 512)
    drop4 = layers.Dropout(0.5)(mid) 
    mid = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(drop4)                                       # (48, 48, 512)
    mid = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(mid)                                       # (48, 48, 512)

    up1 = layers.UpSampling2D(size = (2,2))(mid)                                                                # (96, 96, 256)
    up1 = layers.concatenate([conv4,up1])                                                                         # (96, 96, 512)
    up1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(up1)                                       # (96, 96, 256)
    up1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same')(up1)                                       # (96, 96, 256)

    up2 = layers.UpSampling2D(size = (2,2))(up1)                                                                  # (192, 192, 128)
    up2 = layers.concatenate([conv3,up2])                                                                         # (192, 192, 256)
    up2 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(up2)                                       # (192, 192, 128)
    up2 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(up2)                                       # (192, 192, 128)

    up3 = layers.UpSampling2D(size = (2,2))(up2)                                                                  # (384, 384, 64)
    up3 = layers.concatenate([conv2,up3])                                                                         # (384, 384, 128)
    up3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(up3)                                        # (384, 384, 64)
    up3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(up3)                                        # (384, 384, 64)

    up4 = layers.UpSampling2D(size = (2,2))(up3)                                                                  # (384, 384, 64)
    up4 = layers.concatenate([conv1,up4])                                                                         # (384, 384, 128)
    up4 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(up4)                                        # (384, 384, 64)
    up4 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(up4) 

    model = layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(up4)                                       # (384, 384, 2)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)                                                    # (384, 384, 1)

    return Model(inputs=inputs, outputs=model)


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

def regularization_model(reg):
    inputs = Input(shape=(image_size[0], image_size[1], 3))
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(inputs)                                   # (384, 384, 64)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(conv1)                                    # (384, 384, 64)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  
    drop1 = layers.Dropout(0.5)(pool1)                                                         # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(drop1)                                   # (192, 192, 128)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(conv2)                                   # (192, 192, 128)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)   
    drop2 = layers.Dropout(0.5)(pool2)                                                          # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(drop2)                                   # (96, 96, 256)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(conv3)                                   # (96, 96, 256)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)    
    drop3 = layers.Dropout(0.5)(pool3)                                                         # (96, 96, 256)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(drop3)                                   # (96, 96, 256)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(conv4)  

    mid = layers.MaxPooling2D(pool_size=(2, 2))(conv4)                                                            # (48, 48, 512)
    drop4 = layers.Dropout(0.5)(mid) 
    mid = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(drop4)                                       # (48, 48, 512)
    mid = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(mid)                                       # (48, 48, 512)

    up1 = layers.UpSampling2D(size = (2,2))(mid)                                                                # (96, 96, 256)
    up1 = layers.concatenate([conv4,up1])                                                                         # (96, 96, 512)
    up1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up1)                                       # (96, 96, 256)
    up1 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up1)                                       # (96, 96, 256)

    up2 = layers.UpSampling2D(size = (2,2))(up1)                                                                  # (192, 192, 128)
    up2 = layers.concatenate([conv3,up2])                                                                         # (192, 192, 256)
    up2 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up2)                                       # (192, 192, 128)
    up2 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up2)                                       # (192, 192, 128)

    up3 = layers.UpSampling2D(size = (2,2))(up2)                                                                  # (384, 384, 64)
    up3 = layers.concatenate([conv2,up3])                                                                         # (384, 384, 128)
    up3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up3)                                        # (384, 384, 64)
    up3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up3)                                        # (384, 384, 64)

    up4 = layers.UpSampling2D(size = (2,2))(up3)                                                                  # (384, 384, 64)
    up4 = layers.concatenate([conv1,up4])                                                                         # (384, 384, 128)
    up4 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up4)                                        # (384, 384, 64)
    up4 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up4) 

    model = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(up4)                                       # (384, 384, 2)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)                                                    # (384, 384, 1)

    return Model(inputs=inputs, outputs=model)



def batch_normalization_model(image_size):
    def batch_convolution_block(model, channels, activation='relu', padding='same'):
        model = layers.BatchNormalization()(model)
        model = layers.Conv2D(channels, 3, activation = activation, padding = padding)(model) 
        model = layers.Conv2D(channels, 3, activation = activation, padding = padding)(model)
        return model

    inputs = Input(shape=(image_size[0], image_size[1], 3))

    conv1 = batch_convolution_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  

    conv2 = batch_convolution_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = batch_convolution_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = batch_convolution_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = batch_convolution_block(pool4, 1024) 

    up1 = layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(conv5)     
    conc1 = layers.concatenate([conv4,up1])
    conv6 = batch_convolution_block(conc1, 512) 

    up2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(conv6)     
    conc2 = layers.concatenate([conv3,up2])
    conv7 = batch_convolution_block(conc2, 256) 

    up3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(conv7)     
    conc3 = layers.concatenate([conv2,up3])
    conv8 = batch_convolution_block(conc3, 128) 

    up4 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(conv8)     
    conc4 = layers.concatenate([conv1,up4])
    conv9 = batch_convolution_block(conc4, 64) 

    model = layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)                                       # (384, 384, 2)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)                                                    # (384, 384, 1)

    return Model(inputs=inputs, outputs=model)


def unet_model(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 3))    #(300, 300)

    conv1 = double_convolution_block(inputs, 64)               #(296, 296)
    conv2 = contracting_block(conv1, 128)                      #(144, 144)
    conv3 = contracting_block(conv2, 256)                      #(68, 68)
    conv4 = contracting_block(conv3, 512)                      #(30, 30)
    conv5 = contracting_block(conv4, 1024)                     #(11, 11)

    crop1 = layers.Cropping2D(4)(conv4)                        #(22, 22)
    crop2 = layers.Cropping2D(16)(conv3)                       #(36, 36)
    crop3 = layers.Cropping2D(40)(conv2)                       #(64, 64)
    crop4 = layers.Cropping2D(88)(conv1)                       #(120, 120)

    upconv1 = expanding_block(conv5, crop1, 512)               #(18, 18)
    upconv2 = expanding_block(upconv1, crop2, 256)             #(32, 32)
    upconv3 = expanding_block(upconv2, crop3, 128)             #(60, 60) 
    upconv4 = expanding_block(upconv3, crop4, 64)              #(116, 116)

    model = layers.Conv2D(2, 1, activation = 'relu')(upconv4)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)
    return Model(inputs=inputs, outputs=model)