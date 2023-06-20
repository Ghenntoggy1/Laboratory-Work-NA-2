# Ex 1. New Computer Game
import numpy as np
import sympy as sp  # for an easier handling of functions
from prettytable import PrettyTable as Pt  # for a pretty print


def Gaussian_Quadrature(tol, a, b, fun):
    GQR = []
    ERR = ['-']
    ORD = []
    err = 1
    order = 1
    while abs(err) > tol and order < 101:
        [xv, w] = np.polynomial.legendre.leggauss(order)
        GQR_val = sum(w * fun(a, b, xv))
        GQR.append(GQR_val)
        ORD.append(order)
        order += 1
        if order >= 3:
            err = GQR[order - 2] - GQR[order - 3]
            ERR.append(err)
    return GQR, ERR, ORD


function = input("Input Equation of Motion (ex. t**2+1/(t**3-t**2+3*t)): ")
lb = float(input("Input Lower Bound of Integration: "))
ub = float(input("Input Upper Bound of Integration: "))
tolerance = float(input("Input Desired Tolerance: "))
if lb != -1 or ub != 1:
    coeff = '((ub-lb)/2)*'
    function = function.replace('t', '((ub+lb+t*(lb-ub))/2)')
    func_eval = sp.lambdify([sp.symbols('lb'), sp.symbols('ub'), sp.symbols('t')], coeff + function)
else:
    func_eval = sp.lambdify([sp.symbols('lb'), sp.symbols('ub'), sp.symbols('t')], function)
GQR_List, Err_list, Ord_list = Gaussian_Quadrature(tolerance, lb, ub, func_eval)

table = Pt()
table.add_column("n", Ord_list)
table.add_column("Gaussian Quadrature", GQR_List)
table.add_column("Error", Err_list)
print(table)
