import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np

def lagrange_interpolation(x, y, x_val):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_val - x[j]) / (x[i] - x[j])
        result += term
    return result


def piecewise_linear_interpolation(x, y, x_val):
    n = len(x)
    for i in range(1, n):
        if x[i] >= x_val:
            return y[i - 1] + (y[i] - y[i - 1]) * (x_val - x[i - 1]) / (x[i] - x[i - 1])


def divided_difference(x, y):
    n = len(x)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def newton_interpolation(x, y, x_val):
    n = len(x)
    table = divided_difference(x, y)
    result = table[0][0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x[i - 1])
        result += table[0][i] * product
    return result


def romberg_integration(f, a, b, n):
    r = np.array([[0] * (n+1)] * (n+1), float)
    h = b - a
    r[0,0] = 0.5 * h * (f(a) + f(b))

    powerOf2 = 1
    for i in range(1, n + 1):
        h = 0.5 * h
        sum = 0.0
        powerOf2 = 2 * powerOf2
        for k in range(1, powerOf2, 2):
            sum = sum + f(a + k * h)

        r[i,0] = 0.5 * r[i-1,0] + sum * h

        powerOf4 = 1
        for j in range(1, i + 1):
            powerOf4 = 4 * powerOf4
            r[i,j] = r[i,j-1] + (r[i,j-1] - r[i-1,j-1]) / (powerOf4 - 1)

    return r[n][n]


data = np.genfromtxt('dataset_3.txt', delimiter=', ', skip_header=2)
items_purchased = list(data[:, 0])
time_spent = list(data[:, 1])
x = []
y = []
x_new = []
for i in data:
    if not np.isnan(i[1]):
        x.append(int(i[0]))
        y.append(int(i[1]))
    else:
        x_new.append(int(i[0]))


# Linear Interpolation:
y_new1 = []
for x_val in x_new:
    y_n = piecewise_linear_interpolation(x, y, x_val)
    y_new1.append(y_n)

print("Interpolated values by Piecewise Linear Interpolation:")
for i in range(len(x_new)):
    print(x_new[i], y_new1[i])

def f1(xi):
    return piecewise_linear_interpolation(x, y, xi)

result1 = romberg_integration(f1, x[0], x[-1], 5)
print("Romberg Integration (Piecewise Linear Interpolation):", result1)

plt.plot(x, y, marker=None, linestyle='solid', color='blue', label='Original Data')
plt.plot(x_new, y_new1, marker='*', linestyle='None', markersize=10, color='red', label='Interpolated Data')
plt.xlabel('Items Purchased')
plt.ylabel('Time Spent, minutes')
plt.legend()
plt.title('Piecewise Linear Interpolation')
plt.grid('minor')
plt.show()
print('=============================================================================')

# Cubic Spline Interpolation:
y_new2 = []
cs = CubicSpline(x, y)
for x_val in x_new:
    y_n = cs(x_val)
    y_new2.append(y_n)

print("Interpolated values by Cubic Spline Interpolation:")
for i in range(len(x_new)):
    print(x_new[i], y_new2[i])

def f2(xi):
    return cs(xi)

result2 = romberg_integration(f2, x[0], x[-1], 5)
print("Romberg Integration (Cubic Spline Interpolation):", result2)

plt.plot(x, y, marker=None, linestyle='solid', color='blue', label='Original Data')
plt.plot(x_new, y_new2, marker='*', linestyle='None', markersize=10, color='red', label='Interpolated Data')
plt.xlabel('Items Purchased')
plt.ylabel('Time Spent, minutes')
plt.legend()
plt.title('Cubic Spline Interpolation')
plt.grid('minor')
plt.show()
print('=============================================================================')

# Lagrange Interpolation:
y_new3 = []
x_alt3 = [1, 2]
y_alt3 = [10, 15]
for x_val in x_new:
    y_n = lagrange_interpolation(x, y, x_val)
    y_new3.append(y_n)

for i in range(2, len(x)):
    y_n = lagrange_interpolation(x_alt3, y_alt3, x[i])
    y_alt3.append(y_n)
    x_alt3.append(x[i])

print("Interpolated values by Lagrange Interpolation:")
for i in range(len(x_new)):
    print(x_new[i], y_new3[i])

def f3(xi):
    return lagrange_interpolation(x, y, xi)

result3 = romberg_integration(f3, x[0], x[-1], 5)
print("Romberg Integration (Lagrange Interpolation):", result3)

plt.plot(x, y, marker=None, linestyle='solid', color='blue', label='Original Data')
plt.plot(x_new, y_new3, marker='*', linestyle='None', markersize=10, color='red', label='Interpolated Data')
plt.xlabel('Items Purchased')
plt.ylabel('Time Spent, minutes')
plt.legend()
plt.title('Lagrange Interpolation')
plt.grid('minor')
plt.show()
plt.plot(x_alt3, y_alt3, marker=None, linestyle='solid', color='blue')
plt.title('Lagrange Interpolation Runge Phenomenon')
plt.grid('minor')
plt.show()
print('=============================================================================')

# Newton Interpolation:
y_new4 = []
x_alt4 = []
y_alt4 = []
for x_val in x_new:
    y_n = newton_interpolation(x, y, x_val)
    y_new4.append(y_n)

print("Interpolated values by Newton Interpolation:")
for i in range(len(x_new)):
    print(x_new[i], y_new4[i])

for x_val in x:
    y_n = newton_interpolation(x, y, x_val)
    y_alt4.append(y_n)

def f4(xi):
    return newton_interpolation(x, y, xi)

result4 = romberg_integration(f4, x[0], x[-1], 5)
print("Romberg Integration (Newton Interpolation):", result4)

plt.plot(x, y, marker=None, linestyle='solid', color='blue', label='Original Data')
plt.plot(x_new, y_new4, marker='*', linestyle='None', markersize=10, color='red', label='Interpolated Data')
plt.xlabel('Items Purchased')
plt.ylabel('Time Spent, minutes')
plt.legend()
plt.title('Newton Interpolation')
plt.grid('minor')
plt.show()
plt.plot(x, y_alt4, marker=None, linestyle='solid', color='blue')
plt.title('Newton Interpolation Runge Phenomenon')
plt.grid('minor')
plt.show()