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

    for key, value in history.items():
        if key.startswith('val'): continue

        training = smooth_curve(value) if smooth else value
        validation = smooth_curve(history[f'val_{key}']) if smooth else history[f'val_{key}'] 
        epochs = range(1, len(training) + 1)
        try:
            plt.figure()
            plt.plot(epochs, training, 'r', label=f'Training {key}')
            plt.plot(epochs, validation, 'b', label=f'Validation {key}')
            plt.title(f'Training and validation {key}')
            plt.legend()
        except: 
            pass
   
    plt.show()

    for key, value in history.items():
        if not key.startswith('val'): continue
        print(f'Min/Max Validation {key}', min(value), max(value))

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