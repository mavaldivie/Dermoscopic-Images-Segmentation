from keras import backend as K
import tensorflow as tf

smooth_default = 1.

def dice_coef(y_true, y_pred, smooth = smooth_default, per_batch = True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else: 
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)
    
def jacc_coef(y_true, y_pred, smooth = smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    
def jacc_loss(y_true, y_pred):
    return 1 - jacc_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

miou = tf.metrics.MeanIoU(num_classes=2)
def mean_IoU(y_true, y_pred, threshold=0.6):
    pred = tf.cast(y_pred > threshold, tf.int32)
    return miou(y_true, pred)

acc = tf.keras.metrics.BinaryAccuracy(threshold=0.6)

def sensitivity(y_true, y_pred, threshold=0.6, smooth=smooth_default):
    true = tf.cast(y_true, tf.float32)
    pred = tf.cast(y_pred > threshold, tf.float32)

    f_true = K.batch_flatten(true)
    f_pred = K.batch_flatten(pred)
    TP = K.sum(f_true * f_pred, axis=1, keepdims=True)
    f_pred = 1.0 - f_pred
    FN = K.sum(f_true * f_pred, axis=1, keepdims=True)

    return (TP + smooth) / (FN + TP + smooth)

def specificity(y_true, y_pred, threshold=0.6, smooth=smooth_default):
    true = tf.cast(y_true, tf.float32)
    pred = tf.cast(y_pred > threshold, tf.float32)

    f_true = K.batch_flatten(true)
    f_pred = K.batch_flatten(pred)
    f_true = 1.0 - f_true

    FP = K.sum(f_true * f_pred, axis=1, keepdims=True)
    f_pred = 1.0 - f_pred
    TN = K.sum(f_true * f_pred, axis=1, keepdims=True)

    return (TN + smooth) / (TN + FP + smooth)