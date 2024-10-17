# check_dimensions

Easily check dimensions of arrays and arraylikes with a simple decorator

```python
@check_dimensions(X="n, p", y="n")
def calculate_regression_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X.T @ X) @ X.T @ y


@check_dimensions(X="3, 3")
def expects_3_3(X: np.ndarray) -> None:
    if X.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix")


# X has shape (3, 3) and y has shape (3,)
X = np.array([[1, 2, 4], [4, 5, 5], [7, 8, 10]])
y = np.array([1, 2, 4])

b = calculate_regression_coefficients(X, y)
print(b)

# Here, W has shape (4, 3) and y has shape (3,)
W = np.array([[1, 2, 4], [4, 5, 5], [7, 8, 10], [1, 2, 4]])
t = np.array([1, 2, 4])

# This raises an IncompatibleDimensionError
a = calculate_regression_coefficients(W, t)

# Can also specify axis dimensions precisely
expects_3_3(X)

# This raises an error
expects_3_3(W)
```
