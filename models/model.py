import numpy as np
import math
import matplotlib.pyplot as plt

class Model:
    w = 0.0
    b = 0.0
    cost = math.inf
    cost_history = []

    def __init__(self, alpha: float, iterations: int) -> None:
        self.alpha = alpha
        self.iterations = iterations


    def print_status(self):
        """
        Prints alpha and the current values of w, b and cost.
        Args:
            None
        Returns:
            None
        """
        print(f"======================")
        print(f"Alpha: {self.alpha}")
        print(f"w: {self.w}")
        print(f"b: {self.b:0.2f}")
        print(f"cost: {self.cost:0.2f}")
        print(f"======================")

    def compute_model_output(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])
    
    def compute_model_output(self, x: float) -> float:
        return 0
    
    def compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0
    
    def compute_gradient(self, x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
        return 0,0
    
    def gradient_descent(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        return 0,0
    
    def predict(self, n: int):
        """
        Uses current parameters to assert new data.
        Args:
            n (int): Number of data tried. 
        Returns:
            None
        """
        x = np.random.rand(n)
        f_wb = self.compute_model_output(x)
        for i, obj in enumerate(f_wb):
            print(f"({i}) {x[i]:0.2f} is {obj:0.2f}")