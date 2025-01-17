import numpy as np
import math
import matplotlib.pyplot as plt

class SF_LR_Model:
    def __init__(self, w: float, b: float, alpha: float, iterations: int) -> None:
        self.w = w
        self.b = b
        self.alpha = alpha
        self.iterations = iterations
        self.cost = math.inf

    def print_status(self):
        print(f"======================")
        print(f"Alpha: {self.alpha}")
        print(f"w: {self.w:0.2f} | b: {self.b:0.2f}")
        print(f"cost: {self.cost:0.2f}")
        print(f"======================")

    def compute_model_output(self, x: np.ndarray) -> np.ndarray:
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
            f_wb[i] = self.w * x[i] + self.b

        return f_wb
    
    def compute_model_output(self, x: float) -> float:
        """
        Computes the prediction of a linear model
        Args:
            x (scalar): Data, single example 
        Returns
            f_wb (float): model prediction
        """
        f_wb = self.w * x + self.b

        return f_wb
    
    def compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        cost_sum = 0
        for i in range(n):
            f_wb = self.w * x[i] + self.b
            cost = (f_wb - y[i]) ** 2
            cost_sum += cost
        
        total_cost = (1 / (2 * n)) * cost_sum  
        
        self.cost = total_cost
        return total_cost
    
    def compute_gradient(self, x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
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
    
    def gradient_descent(self, x: np.ndarray, y: np.ndarray):
        _w = self.w
        _b = self.b

        for i in range(self.iterations):
            # dw, db = gradient_function(x, y, _w, _b)
            dw, db = self.compute_gradient(x, y, _w, _b)

            _w = _w - (self.alpha * dw)
            _b = _b - (self.alpha * db)

        self.w = _w
        self.b = _b
        return _w, _b
    
    def predict(self, n: int):
        x = np.random.rand(n)
        f_wb = self.compute_model_output(x)
        for i, obj in enumerate(f_wb):
            print(f"({i}) {x[i]:0.2f} is {obj:0.2f}")