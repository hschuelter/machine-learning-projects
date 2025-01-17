import numpy as np
from model import Model

def compute_model_output(x, w, b):
	"""
	Computes the prediction of a linear model
	Args:
		x (ndarray (m,)): Data, m examples 
		w,b (scalar)    : model parameters  
	Returns
		f_wb (ndarray (m,)): model prediction
	"""
	m = x.shape[0]
	f_wb = np.zeros(m)
	for i in range(m):
		f_wb[i] = w * x[i] + b

	return f_wb

def compute_cost(x, y, w, b):
	n = x.shape[0]
	cost_sum = 0
	for i in range(n):
		f_wb = w * x[i] + b
		cost = (f_wb - y[i]) ** 2
		cost_sum += cost

	total_cost = (1 / (2 * n)) * cost_sum  

	return total_cost

def compute_gradient(x, y, w, b):
	n = x.shape[0]

	dw = 0.0
	db = 0.0

	for i in range(n):
		f_wb = w * x[i] + b
		dw_i = (f_wb - y[i]) * x[i]
		db_i = f_wb - y[i]

		dw += dw_i
		db += db_i

	dw /= n
	db /= n

	return dw, db

def gradient_descent(x, y, w0, b0, alpha, num_iters, gradient_function):
	w = w0
	b = b0

	for i in range(num_iters):
		dw, db = gradient_function(x, y, w, b)

		w = w - (alpha * dw)
		b = b - (alpha * db)

	return w, b

# ==================
def main():
	x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
	y_train = np.array([250, 300, 480,  430,   630, 730,])
	print(f"x_train = {x_train}")
	print(f"y_train = {y_train}")

	w = 0.0
	b = 0.0
	iterations = 10000
	alpha = 0.01

	model = Model(w, b, alpha, iterations)
	model.print_status()

	cost = model.compute_cost(x_train, y_train)
	print(f"w, b: {model.w, model.b}")
	print(f"cost: {cost}")

	model.gradient_descent(x_train, y_train)
	print(f"==== After ====")
	
	cost = model.compute_cost(x_train, y_train)
	print(f"w, b: {model.w, model.b}")
	print(f"cost: {cost}")
	
	model.print_status()

	x_predict = np.array([1.0, 1.2, 1.5, 1.7])

	for i, obj in enumerate(compute_model_output(x_predict, w, b)):
		print(f"({i}) {x_predict[i] * 1000} sqft is US${obj:0.2f}")

if __name__ == "__main__":
	main() 