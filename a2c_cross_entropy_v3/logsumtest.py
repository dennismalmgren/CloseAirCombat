import numpy as np

def log_sum_exp(log_a, log_b):
    return np.maximum(log_a, log_b) + np.log(1 + np.exp(-np.abs(log_a - log_b)))

# Example values
a = 1.5
b = 2.0

log_a = np.log(a)
log_b = np.log(b)

# Using the log-sum-exp approximation
log_sum = log_sum_exp(log_a, log_b)

# Direct calculation for comparison
log_sum_direct = np.log(a + b)
print(log_sum)
print(log_sum_direct)
