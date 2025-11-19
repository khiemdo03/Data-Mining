import numpy as np
import matplotlib.pyplot as plt

def check_invertibility(X):
    XtX = X.T @ X
    det = np.linalg.det(XtX)
    cond_num = np.linalg.cond(XtX)
    is_invertible = (abs(det) > 1e-10) and (cond_num < 1e15)
    return is_invertible

def LinearRegression (X, y, alpha=0, epoch=0, plot=True):
    X_with_bias = np.column_stack([np.ones(X.shape[0]),X])
    is_invertible = check_invertibility(X_with_bias)

    if is_invertible:
        print ("Use Normal Equation Solver")
        w, rss = NormalEquationSolver(X_with_bias, y)
        return w, rss, 'normal_equation', None
    else:
        print("Use Gradient Descent Solver")
        w, rss, rss_history = GradientDescentSolver(X_with_bias, y, alpha, epoch,plot)
        return w, rss, 'gradient_descent', rss_history

def NormalEquationSolver (X, y):
    XtX = X.T @ X
    Xty= X.T @ y
    w = np.linalg.inv(XtX) @ Xty
    rss = np.sum((y-(X @ w))**2)
    return w, rss

def GradientDescentSolver(X, y, alpha, epochs, plot=True):
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    rss_history = []

    for epoch in range(epochs):
        predictions = X @ w
        residuals = y - predictions
        rss = np.sum(residuals ** 2)
        rss_history.append(rss)
        gradient = (-2 / n) * (X.T @ residuals)
        w = w - alpha * gradient
        print(f"Epoch {epoch + 1}, RSS: {rss}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), rss_history)
        plt.xlabel('Iteration')
        plt.ylabel('RSS')
        plt.title('RSS vs Iteration')
        plt.grid(True)
        plt.show()
    return w, rss, rss_history

X_task1 = np.array([
    [1, 2],
    [2, 4],
    [3, 5],
    [4, 7],
    [5, 9],
    [6, 8],
    [7, 10],
    [8, 11],
    [9, 13],
    [10, 15]
])

y_task1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

X_task1_bias = np.column_stack([np.ones(X_task1.shape[0]), X_task1])

w1_normal, rss1_normal = NormalEquationSolver(X_task1_bias, y_task1)

w1_grad, rss1_grad, history1_grad = GradientDescentSolver(X_task1_bias, y_task1, alpha=0.001, epochs=2000, plot=True)

X_task2 = np.array([
    [1.0, 10.0, 1.0],
    [2.0, 20.0, 2.0],
    [3.0, 30.0, 3.0],
    [4.0, 40.0, 4.0],
    [5.0, 50.0, 5.0],
    [6.0, 60.0, 6.0],
    [7.0, 70.0, 7.0],
    [8.0, 80.0, 8.0],
    [9.0, 90.0, 9.0],
    [10.0, 100.0, 10.0]
])

y_task2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

X_task2_bias = np.column_stack([np.ones(X_task2.shape[0]), X_task2])

w2_grad, rss2_grad, history2_grad = GradientDescentSolver(X_task2_bias, y_task2, alpha=0.00001, epochs=2000, plot=True)

def load_csv(filename):
    data=[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for lines in lines [1:]:
            values = lines.strip().split(',')
            data.append([float(v) for v in values])
    return np.array(data)

def train_test_split_random(X, y, test_size=0.3):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def calculate_accuracy(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    correct = np.sum(predictions == y_true)
    return correct / len(y_true)

def evaluate_dataset(filename, dataset_name):
    data = load_csv(filename) 
    X = data[:, :-1]
    y = data[:, -1]

    print(f"Dataset size: X={X.shape}, y={y.shape}")

    train_accuracies = []
    test_accuracies = []
    train_rss_values = []
    test_rss_values = []

    for split_num in range(1, 11):
        print(f"Split {split_num}:")
        X_train, X_test, y_train, y_test = train_test_split_random(X, y, test_size=0.3)
        w, _, solver, _ = LinearRegression(X_train, y_train, alpha=0.00001, epoch=2000, plot=False)
        
        y_train_pred = X_train @ w[1:] + w[0]
        y_test_pred = X_test @ w[1:] + w[0]
        
        train_acc = calculate_accuracy(y_train, y_train_pred)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        rss_train = np.sum(train_residuals ** 2)
        rss_test = np.sum(test_residuals ** 2)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_rss_values.append(rss_train)
        test_rss_values.append(rss_test)

    return {
        'dataset': dataset_name,
        'train_acc_mean': np.mean(train_accuracies),
        'train_acc_std': np.std(train_accuracies),
        'test_acc_mean': np.mean(test_accuracies),
        'test_acc_std': np.std(test_accuracies),
        'train_rss_mean': np.mean(train_rss_values),
        'train_rss_std': np.std(train_rss_values),
        'test_rss_mean': np.mean(test_rss_values),
        'test_rss_std': np.std(test_rss_values)
    }

results_diabetes = evaluate_dataset('Pima_Indian_diabetes.csv', 'Pima Indians Diabetes Dataset')
results_blood = evaluate_dataset('transfusion.data', 'Blood Transfusion Service Center')


print("\n" + "=" * 60)
print("FINAL SUMMARY - ALL TASKS")
print("=" * 60)

print("\n--- TASK 1: Invertible Matrix (Both Solvers Work) ---")
print(f"Dataset: 10 samples, 2 features")
print(f"\nNormal Equation Results:")
print(f"  Weights: {w1_normal}")
print(f"  RSS: {rss1_normal:.4f}")
print(f"\nGradient Descent Results:")
print(f"  Weights: {w1_grad}")
print(f"  RSS: {rss1_grad:.4f}")
print(f"\nRSS Difference: {abs(rss1_normal - rss1_grad):.6f}")

print("\n--- TASK 2: Singular Matrix (Only Gradient Descent Works) ---")
print(f"Dataset: 10 samples, 3 features (with linear dependencies)")
print(f"Matrix Condition: Non-invertible (Column 2 = 10×Column 1, Column 3 = Column 1)")
print(f"\nGradient Descent Results:")
print(f"  Weights: {w2_grad}")
print(f"  RSS: {rss2_grad:.4f}")

print("\n--- TASK 3: Real-World Classification Datasets ---")
print(f"\n{results_diabetes['dataset']}:")
print(f"  Training Accuracy Mean: {results_diabetes['train_acc_mean']:.4f}")
print(f"  Training Accuracy Std: {results_diabetes['train_acc_std']:.4f}")
print(f"  Test Accuracy Mean: {results_diabetes['test_acc_mean']:.4f}")
print(f"  Test Accuracy Std: {results_diabetes['test_acc_std']:.4f}")
print(f"  Training RSS Mean: {results_diabetes['train_rss_mean']:.2f}")
print(f"  Training RSS Std: {results_diabetes['train_rss_std']:.2f}")
print(f"  Test RSS Mean: {results_diabetes['test_rss_mean']:.2f}")
print(f"  Test RSS Std: {results_diabetes['test_rss_std']:.2f}")

print(f"\n{results_blood['dataset']}:")
print(f"  Training Accuracy Mean: {results_blood['train_acc_mean']:.4f}")
print(f"  Training Accuracy Std: {results_blood['train_acc_std']:.4f}")
print(f"  Test Accuracy Mean: {results_blood['test_acc_mean']:.4f}")
print(f"  Test Accuracy Std: {results_blood['test_acc_std']:.4f}")
print(f"  Training RSS Mean: {results_blood['train_rss_mean']:.2f}")
print(f"  Training RSS Std: {results_blood['train_rss_std']:.2f}")
print(f"  Test RSS Mean: {results_blood['test_rss_mean']:.2f}")
print(f"  Test RSS Std: {results_blood['test_rss_std']:.2f}")