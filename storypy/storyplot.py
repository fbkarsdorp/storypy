import matplotlib.pyplot as plt
import numpy as np

def grid_plot(array, labels=[]):
    plt.imshow(1-array, cmap='gray', interpolation='nearest')
    plt.yticks(np.arange(0, len(labels)), labels)
    plt.show()


