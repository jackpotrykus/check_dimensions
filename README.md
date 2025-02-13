# check_shapes

Easily check shapes of arrays and arraylikes with a simple decorator

```python
import numpy as np

import check_shapes as chk


# Supports symbolic/dynamic axis dimensions
@chk.args(("n", "p"), ("n",)).returns(("p",))
def calculate_regression_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Specify the exact dimensions of inputs and outputs
@chk.args((3, 3)).returns((3, 3))
def identity_3_3(X: np.ndarray) -> np.ndarray:
    return X

# Use keyword arguments optionally/as necessary
@chk.args(array1=("n,"), array2=("n,"))
def long_function(a, b, c, d, array1, array2): ...

# Can specify just the args, or just the return
@chk.args(X=(3, 3))
def take_3_3_return_nothing(X: np.ndarray):
    return

@chk.returns((3, 3))
def take_nothing_return_3_3() -> np.ndarray:
    return np.eye(3)


# X has shape (3, 3) and y has shape (3,)
X = np.array([[1, 2, 4], [4, 5, 5], [7, 8, 10]])
y = np.array([1, 2, 4])

b = calculate_regression_coefficients(X, y)
print(b)

# Here, W has shape (4, 3)
W = np.array([[1, 2, 4], [4, 5, 5], [7, 8, 10], [1, 2, 4]])

# This raises an IncompatibleDimensionError
a = calculate_regression_coefficients(W, y)

# Can also specify axis dimensions precisely
identity_3_3(X)

# This raises an error
identity_3_3(W)
```

Why `check_shapes`? There's usually an error anyways.

* Sometimes there isn't an error.
* Catch incompatible shapes immediately, before any expensive intermediate calculation.
* Utilize defensive programming.
* Make your code self-documenting.