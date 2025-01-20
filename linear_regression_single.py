import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.single_feature_model import Single_Feature_Model
from plot.plot import Plot

def import_data() -> tuple[np.ndarray, np.ndarray]:
	file = './datasets/challenging_single_variable_dataset.csv'
	df = pd.read_csv(file, sep=',', header=1)
	training = df.to_numpy()
	x_train = np.array(training[:, 0]).astype(float)
	y_train = np.array(training[:, 1]).astype(float)

	return x_train, y_train


def main():
	x_train, y_train = import_data()

	iterations = 1000
	alpha = 0.001

	model = Single_Feature_Model(alpha, iterations)
	model.compute_cost(x_train, y_train)
	model.print_status()

	model.gradient_descent(x_train, y_train)

	model.compute_cost(x_train, y_train)
	model.print_status()

	plot = Plot(x_train, y_train, model.compute_model_output(x_train), model.cost_history, 'linear/single/')
	plot.plot('Random values', 'x', 'y')
	plot.plot_cost()

	# model.predict(9)

if __name__ == "__main__":
	main() 