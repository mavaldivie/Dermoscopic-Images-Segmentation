import matplotlib.pyplot as plt
import json

def plot_graphs(history, smooth = False):
    
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
   
    acc = smooth_curve(history['binary_accuracy']) if smooth else history['binary_accuracy']
    val_acc = smooth_curve(history['val_binary_accuracy']) if smooth else history['val_binary_accuracy']
    loss = smooth_curve(history['loss']) if smooth else history['loss']
    val_loss = smooth_curve(history['val_loss']) if smooth else history['val_loss']
    dice = smooth_curve(history['dice_coef']) if smooth else history['dice_coef']
    val_dice = smooth_curve(history['val_dice_coef']) if smooth else history['val_dice_coef']
    jacc = smooth_curve(history['jacc_coef']) if smooth else history['jacc_coef']
    val_jacc = smooth_curve(history['val_jacc_coef']) if smooth else history['val_jacc_coef']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, dice, 'bo', label='Training dice')
    plt.plot(epochs, val_dice, 'b', label='Validation dice')
    plt.title('Training and validation dice coefficients')
    plt.legend()

    plt.figure()
    plt.plot(epochs, jacc, 'bo', label='Training Jaccard')
    plt.plot(epochs, val_jacc, 'b', label='Validation Jaccard')
    plt.title('Training and validation Jaccard coefficients')
    plt.legend()

    plt.show()

    print('Max Validation Acc:', max(val_acc))
    print('Min Validation Loss:', min(val_loss))
    print('Max Validation Dice:', max(val_dice))
    print('Max Validation Jaccard:', max(val_jacc))

def save_dict_as_json(data, file_name):
    with open(file_name + ".json", "w") as fp:
        json.dump(data,fp, indent = 4) 

def join_dictionaries(*args):
    history, *remaining = args
    for h in remaining:
        for key, value in h.items():
            try:
                history[key] += value
            except: 
                pass
    return history

def load_multiple(*args):
    histories = []
    for i in args:
        with open(i) as f: 
            histories.append(json.load(f))
    return join_dictionaries(*histories)