import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, x: np.ndarray, y: np.ndarray, y_predict: np.ndarray, output: str) -> None:
        self.x = x
        self.y = y
        self.y_predict = y_predict
        self.output = output

    def plot(self, title: str, xlabel: str, ylabel: str):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.scatter(self.x, self.y, marker='x', c='r')
        plt.plot(self.x, self.y_predict)
        plt.savefig(self.output + 'figure-0.png')