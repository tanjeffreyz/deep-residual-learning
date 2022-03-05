"""Visualizes training and evaluation metrics."""

import os
import numpy as np
from matplotlib import pyplot as plt


def compare_20():
    plt.clf()
    info = (
        ('models/CifarResNet-20-P/03_04_2022/23_14_17',     'Plain',    'orange'),
        ('models/CifarResNet-20-R-A/03_04_2022/19_31_49',   'Option A', 'red'),
        ('models/CifarResNet-20-R-B/03_04_2022/21_02_13',   'Option B', 'blue')
    )
    for path, label, color in info:
        test_errors = np.load(os.path.join(path, 'test_errors.npy'))
        print(test_errors)
        train_errors = np.load(os.path.join(path, 'train_errors.npy'))
        plt.plot(test_errors[0], test_errors[1], label=label, color=color)
        plt.plot(train_errors[0], train_errors[1], color=color, alpha=1/3)

    plt.tight_layout
    plt.show()


if __name__ == '__main__':
    compare_20()
