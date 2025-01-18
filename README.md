# Stanford Machine Learning Course Algorithms

This repository contains implementations of algorithms covered in the Stanford University Machine Learning course. It is designed to help reinforce concepts and provide practical coding examples.

## 📚 Overview

The primary focus of this repository is to implement foundational machine learning algorithms from scratch, with a special emphasis on linear regression. As the project evolves, more algorithms and advanced topics will be added.

### Currently Implemented:

- [x] **Single Feature Linear Regression**
  - Implements a linear regression model for datasets with a single feature.
  - Includes gradient descent for parameter optimization.

- [ ] **Multiple Feature Linear Regression**
  - Extends the single-feature model to handle multiple features.
  - Includes normalization of features and vectorized gradient descent.

## 🛠️ Technologies Used

- **Python**: Core programming language used for implementations.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting data and visualizing results.
- **Pandas**: For handling datasets.

## 📂 Repository Structure

```
├── datasets/                             # Example datasets used for testing
├── output/                               # Visualizations of results
├── models/                               # Folder for implementation of Models classes
│   ├── single_feature_model.py           # Class implementation of Single Feature Linear Regression Model 
│   └── multi_feature_model.py            # Class implementation of Multiple Feature Linear Regression Model 
├── plot/                                 # Folder for implementation of Plot classes 
│   └── plot.py                           # Class designed to handle the results plotting
├── linear_regression_single.py           # Script for running the Single Feature Linear Regression 
└── README.md                             # Project documentation
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hschuelter/stanford-ml-algorithms.git
   cd stanford-ml-algorithms
   ```

2. **Set up the environment**:
   Install the required libraries:
   ```bash
   apt-get install python3-numpy python3-matplotlib
   ```

3. **Run the code**:
   - For single feature regression:
     ```bash
     python3 linear_regression_single.py
     ```
   - For multiple feature regression:
     ```bash
     python3 linear_regression_multiple.py
     ```

## 📈 Features

- **Gradient Descent Implementation**:
  - Iterative optimization for finding the best-fit line.
  - Support for customizable learning rates and iterations.

- **Visualization**:
  - Plots the dataset and the regression line for single feature regression.
  - Learning curve visualizations (coming soon).

- **Error Metrics**:
  - Includes Mean Squared Error (MSE) for evaluating model performance.

## 🌟 Future Work

- [ ] Polynomial Regression
- [ ] Logistic Regression
- [ ] Regularization (Ridge and Lasso)
- [ ] Support Vector Machines (SVMs)
- [ ] Neural Networks (Basic Implementation)

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to help improve the repository.

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
