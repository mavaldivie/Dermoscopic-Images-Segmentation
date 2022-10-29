from keras import Input, Model
from keras import layers
from keras.optimizers import Adam
from keras import regularizers

def double_convolution_block(model, channels, activation='relu', padding=None,  kernel_regularizer=None):
    if padding:
        model = layers.Conv2D(channels, (3, 3), activation = activation, padding = padding, kernel_regularizer=kernel_regularizer)(model) 
        model = layers.Conv2D(channels, (3, 3), activation = activation, padding = padding, kernel_regularizer=kernel_regularizer)(model)
    else:
        model = layers.Conv2D(channels, (3, 3), activation = activation, kernel_regularizer=kernel_regularizer)(model) 
        model = layers.Conv2D(channels, (3, 3), activation = activation, kernel_regularizer=kernel_regularizer)(model)
    return model

def contracting_block(model, channels, activation='relu', padding=None, kernel_regularizer=None):
    model = layers.MaxPooling2D(pool_size=(2, 2))(model)
    model = double_convolution_block(model, channels, activation, padding, kernel_regularizer=kernel_regularizer)
    return model

def expanding_block(model, residual, channels, activation='relu', padding=None, kernel_regularizer=None):
    model = layers.Conv2DTranspose(channels, (2,2), strides=(2,2), padding='same', kernel_regularizer=kernel_regularizer)(model) 
    model = layers.concatenate([model, residual])
    model = double_convolution_block(model, channels, activation, padding, kernel_regularizer=kernel_regularizer)
    return model

def batch_convolution_block(model, channels, activation='relu', padding='same'):
    model = layers.BatchNormalization()(model)
    model = double_convolution_block(model, channels, activation=activation, padding=padding)
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

def short_model(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 3))    #(300, 300)

    conv1 = double_convolution_block(inputs, 64, padding = 'same')               #(296, 296)
    conv1 = layers.Dropout(0.5)(conv1)
    conv2 = contracting_block(conv1, 128, padding = 'same')                      #(144, 144)
    conv2 = layers.Dropout(0.5)(conv2)
    conv3 = contracting_block(conv2, 256, padding = 'same')                      #(68, 68)
    conv3 = layers.Dropout(0.5)(conv3)
    conv4 = contracting_block(conv3, 512, padding = 'same')                      #(30, 30)
    conv4 = layers.Dropout(0.5)(conv4)

    upconv2 = expanding_block(conv4, conv3, 256, padding = 'same')          
    upconv3 = expanding_block(upconv2, conv2, 128, padding = 'same')            
    upconv4 = expanding_block(upconv3, conv1, 64, padding = 'same')              

    model = layers.Conv2D(2, 1, activation = 'relu')(upconv4)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)
    return Model(inputs=inputs, outputs=model)


def dropout_model(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], 3))    #(300, 300)

    conv1 = double_convolution_block(inputs, 64, padding = 'same')               #(296, 296)
    conv1 = layers.Dropout(0.5)(conv1)
    conv2 = contracting_block(conv1, 128, padding = 'same')                      #(144, 144)
    conv2 = layers.Dropout(0.5)(conv2)
    conv3 = contracting_block(conv2, 256, padding = 'same')                      #(68, 68)
    conv3 = layers.Dropout(0.5)(conv3)
    conv4 = contracting_block(conv3, 512, padding = 'same')                      #(30, 30)
    conv4 = layers.Dropout(0.5)(conv4)
    conv5 = contracting_block(conv4, 1024, padding = 'same')                     #(11, 11)
    conv5 = layers.Dropout(0.5)(conv5)

    upconv1 = expanding_block(conv5, conv4, 512, padding = 'same')            
    upconv2 = expanding_block(upconv1, conv3, 256, padding = 'same')          
    upconv3 = expanding_block(upconv2, conv2, 128, padding = 'same')            
    upconv4 = expanding_block(upconv3, conv1, 64, padding = 'same')              

    model = layers.Conv2D(2, 1, activation = 'relu')(upconv4)
    model = layers.Conv2D(1, 1, activation = 'sigmoid')(model)
    return Model(inputs=inputs, outputs=model)

def regularization_model(image_size, reg):
    inputs = Input(shape=(image_size[0], image_size[1], 3))    #(300, 300)

    conv1 = double_convolution_block(inputs, 64, padding = 'same', kernel_regularizer=regularizers.L2(reg)) 
    conv2 = contracting_block(conv1, 128, padding = 'same', kernel_regularizer=regularizers.L2(reg))        
    conv3 = contracting_block(conv2, 256, padding = 'same', kernel_regularizer=regularizers.L2(reg))    
    conv4 = contracting_block(conv3, 512, padding = 'same', kernel_regularizer=regularizers.L2(reg))   
    conv5 = contracting_block(conv4, 1024, padding = 'same', kernel_regularizer=regularizers.L2(reg))   

    upconv1 = expanding_block(conv5, conv4, 512, padding = 'same', kernel_regularizer=regularizers.L2(reg))            
    upconv2 = expanding_block(upconv1, conv3, 256, padding = 'same', kernel_regularizer=regularizers.L2(reg))          
    upconv3 = expanding_block(upconv2, conv2, 128, padding = 'same', kernel_regularizer=regularizers.L2(reg))            
    upconv4 = expanding_block(upconv3, conv1, 64, padding = 'same', kernel_regularizer=regularizers.L2(reg))              

    model = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.L2(reg))(upconv4)
    model = layers.Conv2D(1, 1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(reg))(model)
    return Model(inputs=inputs, outputs=model)


def batch_normalization_model(image_size):
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

    #up1 = layers.UpSampling2D(size = (2,2), padding='same')(conv5) 
    up1 = layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(conv5)     
    conc1 = layers.concatenate([conv4,up1])
    conv6 = batch_convolution_block(conc1, 512) 

    #up2 = layers.UpSampling2D(size = (2,2), padding='same')(conv6) 
    up2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(conv6)     
    conc2 = layers.concatenate([conv3,up2])
    conv7 = batch_convolution_block(conc2, 256) 

    #up3 = layers.UpSampling2D(size = (2,2), padding='same')(conv7) 
    up3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(conv7)     
    conc3 = layers.concatenate([conv2,up3])
    conv8 = batch_convolution_block(conc3, 128) 

    #up4 = layers.UpSampling2D(size = (2,2), padding='same')(conv8) 
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