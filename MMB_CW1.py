import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# load data
data = pd.read_csv('data.csv', header=None)

time_vals = data.iloc[:,0].values
z_vals = data.iloc[:,1].values  

def der_model(t, rA, rB):
    diff = rA - rB
    y_val = (np.exp(rA*t) - np.exp(rB*t)) / diff
    return y_val**2


# fit derived model
guess_der = [1.0,-0.5]

params_der, _ = curve_fit(der_model, time_vals, z_vals,p0=guess_der, maxfev=10000)

rA_opt , rB_opt = params_der


print("Derived Results:")
print(f"Estimated Root 1 (r1): {rA_opt:.4f}")
print(f"Estimated Root 2 (r2): {rB_opt:.4f}")
print(f"Parameter A: {-(rA_opt+rB_opt):.4f}")
print(f"Parameter B: {rA_opt*rB_opt:.4f}\n")

# exponential model
def exp_model(t,a,b,c):
    return a + b*np.exp(c*t)


start_guess = [0,0.1,1.0]

params_exp,_ = curve_fit(exp_model,time_vals,z_vals,p0=start_guess,maxfev=10000)


print("Exponential Results:")
print(f"g(t) = {params_exp[0]:.4f} + {params_exp[1]:.4f} * e^({params_exp[2]:.4f}*t)\n")



# polynomial degree
poly_deg = 3
#ploy_deg = 10
# build matrix
design_mat = np.column_stack([time_vals**i for i in range(poly_deg+1)])
z_column = z_vals.reshape(-1,1)


# solve (A^T A)^-1 A^T z
coeffs = np.linalg.inv(design_mat.T @ design_mat) @ design_mat.T @ z_column

print("Polynomial Results: ")
for i,c in enumerate(coeffs.flatten()):
    print(f"a_{i} = {c:.4f}")

# get data for models
z_der = der_model(time_vals,*params_der)
z_exp = exp_model(time_vals,*params_exp)
z_poly = (design_mat @ coeffs).flatten()


# plot results
plt.figure(figsize=(10,6))
plt.scatter(time_vals,z_vals,color='lightgrey',label='Observed Data (y^2)',s=15)
plt.plot(time_vals,z_der,color='red',label='Derived Fit')
plt.plot(time_vals,z_exp,color='blue',linestyle='--',label='Exponential Fit')
plt.plot(time_vals,z_poly,color='green',linestyle=':',label=f'Polynomial Fit (deg {poly_deg})')
plt.xlabel('Time (t) in days')
plt.ylabel('y(t)^2')
plt.title('Comparison of Epidemic Model Fits')
plt.legend()
plt.grid(True,alpha=0.3)
plt.show()