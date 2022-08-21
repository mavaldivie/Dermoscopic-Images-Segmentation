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
   
    acc = smooth_curve(history['acc']) if smooth else history['acc']
    val_acc = smooth_curve(history['val_acc']) if smooth else history['val_acc']
    loss = smooth_curve(history['loss']) if smooth else history['loss']
    val_loss = smooth_curve(history['val_loss']) if smooth else history['val_loss']
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
    plt.show()

def save_dict_as_json(data, file_name):
    with open(file_name + ".json", "w") as fp:
        json.dump(data,fp, indent = 4) 