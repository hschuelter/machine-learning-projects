import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sf_lr_model import SF_LR_Model

# ==================
def main():
	file = './input/single_variable_dataset.csv'
	df = pd.read_csv(file, sep=',', header=1)
	training = df.to_numpy()
	x_train = np.array(training[:, 0]).astype(float)
	y_train = np.array(training[:, 1]).astype(float)

	plt.scatter(x_train, y_train, marker='x', c='r')
	plt.title("Random values")
	plt.ylabel('x')
	plt.xlabel('y')

	w = 0.0
	b = 0.0
	iterations = 10000
	alpha = 0.01

	model = SF_LR_Model(w, b, alpha, iterations)
	model.compute_cost(x_train, y_train)
	model.print_status()

	model.gradient_descent(x_train, y_train)

	print(f"==== After ====")
	model.compute_cost(x_train, y_train)
	model.print_status()

	x_predict = np.array([1.0, 1.2, 1.5, 1.7])

	f_wb = model.compute_model_output(x_predict)
	plt.plot(x_train, model.compute_model_output(x_train))
	plt.savefig('./output/figure-0.png')

	for i, obj in enumerate(f_wb):
		print(f"({i}) {x_predict[i]} is {obj:0.2f}")

if __name__ == "__main__":
	main() 