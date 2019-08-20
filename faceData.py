import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


data = pd.read_csv("fer2013.csv")
count_training = data[(data.Usage) == "Training"].count()
training_samples = count_training[1]

for x in range(1,len(training_samples)):
    if data['Usage'][x-1] == 'Training':
        pixels = data['pixels'][x-1]
        pixels = pixels.replace(" ", ",").split(',')
        pixels = [int(i) for i in pixels]


        pixel_array = np.reshape(pixels, (-1, 48))
        pixel_array = np.true_divide(pixel_array, 256)
        fig = plt.figure()
        plt.imshow(pixel_array, cmap="gray")
        fig.savefig(str('images/' + data['Usage'][x-1] + '_Sample_' + str(x) + "_Emotion_" + str(data['emotion'][x-1])))
        plt.close()


for y in range(1, (len(data)-training_samples)):
    pixels = data['pixels'][y - 1]
    pixels = pixels.replace(" ", ",").split(',')
    pixels = [int(i) for i in pixels]

    pixel_array = np.reshape(pixels, (-1, 48))
    pixel_array = np.true_divide(pixel_array, 256)
    fig = plt.figure()
    plt.imshow(pixel_array, cmap="gray")
    fig.savefig(str('images/' + data['Usage'][y - 1] + '_Sample' + str(y) + "_Emotion" + str(data['emotion'][y - 1])))
    plt.close()