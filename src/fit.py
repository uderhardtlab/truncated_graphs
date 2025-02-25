import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def fit_log(d, C):
    """
    Fits a logarithmic function to the given data.
    
    Parameters:
    d (array-like): Distance values.
    C (array-like): Corresponding centrality values.
    
    Returns:
    tuple: Coefficients (a, b) and fitted values.
    """
    d_nonzero = d[d > 0]
    C_nonzero = C[d > 0]  
    d_transformed = np.log(d_nonzero).reshape(-1, 1)  
    model = LinearRegression()
    model.fit(d_transformed, C_nonzero)
    a = model.coef_[0]
    b = model.intercept_
    return a, b, a * np.log(d_nonzero) + b

    
def fit_piece_wise_linear(d, C, M=1000):
    """
    Fits a piecewise linear function to the given data using optimization.
    
    Parameters:
    d (array-like): Distance values.
    C (array-like): Corresponding centrality values.
    M (int, optional): Large constant for big-M constraints. Default is 1000.
    
    Returns:
    tuple: Optimized slope (m), intercept (c0), and breakpoint (b).
    """
    n = len(d) 
    
    model = gp.Model()
    
    m = model.addVar(vtype=GRB.CONTINUOUS, name="m") # Slope before breakpoint
    c0 = model.addVar(vtype=GRB.CONTINUOUS, name="c0") # y-intercept
    b = model.addVar(vtype=GRB.CONTINUOUS, lb=min(d), ub=max(d), name="b") # Breakpoint
    
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    epsilon = model.addVars(n, vtype=GRB.CONTINUOUS, name="epsilon")
    
    model.setObjective(gp.quicksum(epsilon[i] * epsilon[i] for i in range(n)), GRB.MINIMIZE)
    
    # Setting solver parameters for precision
    model.setParam('OptimalityTol', 1e-4) 
    model.setParam('MIPGap', 0.01)  
    
    for i in range(n):
        # Constraints enforcing piecewise linear fit
        model.addConstr(epsilon[i] >= (m * d[i] + c0 - C[i]) - (1 - z[i]) * M)
        model.addConstr(epsilon[i] >= -(m * d[i] + c0 - C[i]) - (1 - z[i]) * M)
    
        model.addConstr(epsilon[i] >= (m * b + c0 - C[i]) - z[i] * M)
        model.addConstr(epsilon[i] >= -(m * b + c0 - C[i]) - z[i] * M)
    
        # Binary switch for piecewise behavior
        model.addConstr(z[i] * M >= b - d[i])
        model.addConstr((1 - z[i]) * M >= d[i] - b)
    
    model.optimize()
    return m.X, c0.X, b.X


def plot_piece_wise_linear(d, C, m_opt, c0_opt, b_opt, measure, n, t, path):
    """
    Plots the original data and optimized piecewise linear fit.
    """
    d_curve = np.linspace(min(d), max(d), 500)
    C_curve = np.piecewise(
        d_curve,
        [d_curve <= b_opt, d_curve > b_opt],
        [lambda x: m_opt * x + c0_opt, lambda x: m_opt * b_opt + c0_opt]
    )
    
    plt.scatter(d, C, color="blue", label="Original", alpha=0.5)
    plt.plot(d_curve, C_curve, color="red", label="Optimized", linewidth=2)
    plt.xlabel("Distance to border")
    plt.ylabel(f"{measure.capitalize()} centrality")
    plt.title(f"Optimized piece-wise linear fit for {t} graph and {measure} centrality")
    plt.legend()
    plt.savefig(path)


def plot_log(d, C, a, b, f, measure, n, t, path=None):
    """
    Plots the original data and the logarithmic fit.
    """
    d_nonzero = d[d > 0]
    C_nonzero = C[d > 0]  
    plt.scatter(d_nonzero, C_nonzero, label="Original")
    plt.plot(d_nonzero, f, color="red", label=f"Fitted curve: {a:.2f} * log(x) + {b:.2f}")
    
    plt.ylabel(f"{measure.capitalize()} centrality")
    plt.xlabel("Distance to border")
    plt.legend()
    if path:
        plt.savefig(path)
