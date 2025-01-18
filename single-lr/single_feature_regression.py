import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sf_lr_model import SF_LR_Model
from plot import Plot

def import_data() -> tuple[np.ndarray, np.ndarray]:
	file = './datasets/single_variable_dataset.csv'
	df = pd.read_csv(file, sep=',', header=1)
	training = df.to_numpy()
	x_train = np.array(training[:, 0]).astype(float)
	y_train = np.array(training[:, 1]).astype(float)

	return x_train, y_train


def main():
	x_train, y_train = import_data()

	w = 0.0
	b = 0.0
	iterations = 10000
	alpha = 0.01

	model = SF_LR_Model(w, b, alpha, iterations)
	model.compute_cost(x_train, y_train)
	model.print_status()

	model.gradient_descent(x_train, y_train)

	model.compute_cost(x_train, y_train)
	model.print_status()

	plot = Plot(x_train, y_train, model.compute_model_output(x_train), './plots/')
	plot.plot('Random values', 'x', 'y')

	model.predict(100)

if __name__ == "__main__":
	main() 