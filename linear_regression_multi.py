import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.multi_feature_model import Multi_Feature_Model
from plot.plot import Plot

def import_data() -> tuple[np.ndarray, np.ndarray]:
	file = './datasets/multifeature_dataset.csv'
	df = pd.read_csv(file, sep=',', header=0)
	training = df.to_numpy()

	x_train = np.array(training[:,  0:-1]).astype(float)
	y_train = np.array(training[:, -1]).astype(float)

	return x_train, y_train


def main():
	x_train, y_train = import_data()

	iterations = 1000
	alpha = 0.001
	n = x_train.shape[1]

	model = Multi_Feature_Model(alpha, iterations, n)
	model.compute_cost(x_train, y_train)
	model.print_status()

	model.gradient_descent(x_train, y_train)

	model.compute_cost(x_train, y_train)
	model.print_status()

	plot = Plot(x_train, y_train, model.compute_model_output_array(x_train), model.cost_history, 'linear/multi/')
	plot.plot_cost()

	# model.predict(9)

if __name__ == "__main__":
	main() 