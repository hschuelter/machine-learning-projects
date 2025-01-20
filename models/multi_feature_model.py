import numpy as np
import math
import matplotlib.pyplot as plt

from models.model import Model

class Multi_Feature_Model(Model):
    w = 0.0
    b = 0.0
    cost = math.inf
    cost_history = []

    def __init__(self, alpha: float, iterations: int, n: int) -> None:
        self.alpha = alpha
        self.iterations = iterations
        self.w = np.zeros(n)

    def compute_model_output(self, x: np.ndarray) -> float:
        """
        Computes the prediction
        Args:
            x (ndarray (n,)): Data, n features
        Returns
            f_wb (float): model prediction
        """
        f_wb = np.dot(self.w, x) + self.b

        return f_wb
    
    def compute_model_output_array(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the prediction of a multiple feature model
        Args:
            x (ndarray (m,n)): Data, m examples, n features
        Returns
            f_wb (ndarray (m,)): model prediction
        """
        m = x.shape[0]
        f_wb = np.zeros(m)
        for i in range(m):
            f_wb[i] = np.dot(self.w, x[i]) + self.b

        return f_wb
    
    def compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the cost of data.
        Args:
            x (ndarray (m,n)): Data, m examples, n features
            y (ndarray (m, )): Data, m examples 
        Returns:
            cost (float): 
        """
        m = x.shape[0]
        cost_sum = 0
        for i in range(m):
            f_wb = np.dot(self.w, x[i]) + self.b
            cost = (f_wb - y[i]) ** 2
            cost_sum += cost
        
        total_cost = (1 / (2 * m)) * cost_sum  
        
        self.cost = total_cost
        return total_cost
    
    def compute_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
        """
        Computes the gradient, used for gradient descent.
        Args:
            x (ndarray (m,n)): Data, m examples, n features
            y (ndarray (m, )): Data, m examples
            w (ndarray (n, )): Data, n features
            b (scalar)       : model parameters 
        Returns:
            dw (ndarray (n,): gradient for the weights
            db (float)      : gradient for the constant
        """

        m = x.shape[0]
        n = x.shape[1]

        dw = np.zeros(n)
        db = 0.0

        for i in range(m):
            f_wb = np.dot(w, x[i]) + b
            dw_i = np.dot((f_wb - y[i]), x[i])
            db_i = f_wb - y[i]

            dw = dw + dw_i
            db = db + db_i

        dw /= m
        db /= m

        return dw, db
    
    def gradient_descent(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Computes the gradient, used for gradient descent.
        Args:
            x (ndarray (m,n)): Data, m examples, n features 
            y (ndarray (m, )): Data, m examples 
        Returns:
            w,b (scalar)    : model parameters
        """
        _w = self.w
        _b = self.b

        for i in range(self.iterations):
            dw, db = self.compute_gradient(x, y, _w, _b)

            _w = _w - (self.alpha * dw)
            _b = _b - (self.alpha * db)

            self.w = _w
            self.b = _b
            self.cost_history.append(self.compute_cost(x, y))

        self.cost_history = np.array(self.cost_history)
        return _w, _b