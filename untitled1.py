import numpy as np

def MAPE(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def double_exponential_smoothing(x, alpha=0.3, beta=0.5, l_zero=2, b_zero=0, mape=False):
    if not (0 <= alpha <= 1):
        raise ValueError("Invalid alpha")
    if not (0 <= beta <= 1):
        raise ValueError("Invalid beta")
    
    n = len(x)
    forecasts = np.zeros(n)
    l_prev = l_zero
    b_prev = b_zero
    
    for t in range(1, n):
        l_t = alpha * x[t] + (1 - alpha) * (l_prev + b_prev)
        b_t = beta * (l_t - l_prev) + (1 - beta) * b_prev
        forecasts[t] = l_t + b_t
        l_prev = l_t
        b_prev = b_t

    forecasts[0] = np.nan

    if mape:
        mape_value = MAPE(x[1:], forecasts[1:])
        return forecasts, mape_value
    else:
        return forecasts

# Sample input observations
x = np.array([2.92, 0.84, 2.69, 2.42, 1.83, 1.22,0.10,1.32,0.56,-0.35])


forecasts = double_exponential_smoothing(x)

forecasts, mape_value = double_exponential_smoothing(x, l_zero=2 , b_zero=0 ,alpha=0.3,beta=0.5)

print("Forecasts:", forecasts)
print("MAPE:", mape_value)
