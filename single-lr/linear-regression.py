import numpy as np
from sf_lr_model import SF_LR_Model
import matplotlib.pyplot as plt

# ==================
def main():
	x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
	y_train = np.array([250, 300, 480,  430,   630, 730,])

	plt.scatter(x_train, y_train, marker='x', c='r')# Set the title
	plt.title("Housing Prices")
	# Set the y-axis label
	plt.ylabel('Price (in 1000s of dollars)')
	# Set the x-axis label
	plt.xlabel('Size (1000 sqft)')

	print(f"x_train = {x_train}")
	print(f"y_train = {y_train}")

	w = 0.0
	b = 0.0
	iterations = 10000
	alpha = 0.01

	model = SF_LR_Model(w, b, alpha, iterations)
	model.print_status()

	cost = model.compute_cost(x_train, y_train)
	print(f"w, b: {model.w, model.b}")
	print(f"cost: {cost}")

	model.gradient_descent(x_train, y_train)
	print(f"==== After ====")
	
	cost = model.compute_cost(x_train, y_train)
	print(f"w, b: {model.w, model.b}")
	print(f"cost: {model.cost}")
	
	model.print_status()

	x_predict = np.array([1.0, 1.2, 1.5, 1.7])

	f_wb = model.compute_model_output(x_predict)
	plt.plot(x_train, model.compute_model_output(x_train))
	plt.savefig('function.png')

	for i, obj in enumerate(f_wb):
		print(f"({i}) {x_predict[i] * 1000} sqft is US${obj:0.2f}")

if __name__ == "__main__":
	main() 