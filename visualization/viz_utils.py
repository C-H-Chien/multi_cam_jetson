import numpy as np

# Perform linear regression and extrapolation for each segment
def linear_regression(x, y):
    """Simple linear regression using least squares method"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept