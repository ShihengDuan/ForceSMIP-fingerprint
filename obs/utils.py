import numpy as np
from sklearn.linear_model import LinearRegression

def get_slope(x, y):
    X_with_intercept = np.empty(shape=(x.shape[0], 2))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1] = x
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_hat = model.predict(x.reshape(-1, 1))
    residuals = y - y_hat
    rss = np.sum(np.square(residuals))
    xss = np.sum(np.square(x-np.mean(x)))
    n = len(x)
    se = np.sqrt(rss/((n-2)*xss))
    slope = model.coef_[0]
    return slope, se

