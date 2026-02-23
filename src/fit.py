#import gurobipy as gp
#from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def fit_constant(C):
    c = np.mean(C)
    C_model = np.full_like(C, c, dtype=float)
    return c, C_model

    
def fit_exponential_saturation(d, C):
    """
    Fits an exponential saturation model: C(d) = a * (1 - exp(-b * d)) + c
    
    Parameters:
    d (array-like): Distance values.
    C (array-like): Corresponding centrality values.
    
    Returns:
    tuple: Optimized parameters (a, b, c) and fitted values
    """
    d = np.array(d, dtype=float)
    C = np.array(C, dtype=float)

    # Define the model function
    def exp_sat(d, a, b, c):
        return a * (1 - np.exp(-b * d)) + c

    # Initial guess: a = range of C, b = 1 / mean(d), c = min(C)
    p0 = [max(C) - min(C), 1 / (np.mean(d) + 1e-6), min(C)]

    # Fit the model using curve_fit
    popt, _ = curve_fit(exp_sat, d, C, p0=p0, maxfev=5000)

    a, b, c = popt
    C_model = exp_sat(d, a, b, c)
    return a, b, c, C_model

    
def fit_log(d, C):
    """
    Fits a logarithmic function to the given data.
    
    Parameters:
    d (array-like): Distance values.
    C (array-like): Corresponding centrality values.
    
    Returns:
    tuple: Coefficients (a, b) and fitted values.
    """
    d = np.array(d, dtype=float)
    C = np.array(C, dtype=float)
    d_transformed = np.log(d + 1).reshape(-1, 1) 
    
    model = LinearRegression()
    model.fit(d_transformed, C)
    a = model.coef_[0]
    b = model.intercept_
    C_model = a * np.log(d + 1) + b
    return a, b, C_model


def fit_piece_wise_linear(d, C):
    """
    Fits a piecewise linear function with a plateau to the data.
    
    Parameters:
        d (array-like): x-values (e.g., distances)
        C (array-like): y-values (e.g., centrality)
    
    Returns:
        tuple: Optimized parameters (b, m, c0) and fitted values
    """
    d = np.array(d, dtype=float)
    C = np.array(C, dtype=float)

    # Define piecewise linear function: slope until b, then flat
    def piecewise_plateau(x, b, m, c0):
        return np.where(
            x <= b,
            m * x + c0,
            m * b + c0  # fixed from your previous typo
        )
    
    # Initial guess: median for breakpoint, slope=1, intercept=mean
    p0 = [np.median(d), 1.0, np.mean(C)]
    
    # Fit curve
    p_opt, _ = curve_fit(piecewise_plateau, d, C, p0=p0)
    b_opt, m_opt, c0_opt = p_opt
    
    # Compute fitted values
    C_fit = piecewise_plateau(d, b_opt, m_opt, c0_opt)
    
    return m_opt, c0_opt, b_opt, C_fit




def plot_piece_wise_linear(d, C, m_opt, c0_opt, b_opt, measure, graph_type, path=None):
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
    plt.title(f"Optimized piece-wise linear fit for {graph_type} graph and {measure} centrality")
    plt.legend()
    if path:
        plt.savefig(path)


def plot_log(d, C, a, b, f, measure, path=None):
    """
    Plots the original data and the logarithmic fit.
    """   
    sorted_pairs = sorted(zip(d, f))  # [(1, 8), (2, 7), (3, 9)]
    xs_sorted, ys_sorted = zip(*sorted_pairs)

    plt.scatter(d, C, label="Original")
    plt.plot(xs_sorted, ys_sorted, color="red", label=f"Fitted curve: {a:.2f} * log(x) + {b:.2f}")
    
    plt.ylabel(f"{measure.capitalize()} centrality")
    plt.xlabel("Distance to border")
    plt.legend()
    if path:
        plt.savefig(path)


"""
def fit_piece_wise_linear_old(d, C, M=1000):
    Fits a piecewise linear function to the given data using optimization.
    
    Parameters:
    d (array-like): Distance values.
    C (array-like): Corresponding centrality values.
    M (int, optional): Large constant for big-M constraints. Default is 1000.
    
    Returns:
    tuple: Optimized slope (m), intercept (c0), and breakpoint (b).
    n = len(d) 
    
    model = gp.Model()
    model.Params.LogToConsole = 0

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
    
    C_model = np.piecewise(
        d,
        [d <= b.X, d > b.X],
        [lambda x: m.X * x + c0.X, lambda x: m.X * b.X + c0.X]
    )
    
    b_opt = b.X
    c0_opt = c0.X
    m_opt = m.X
    
    C_model = np.where(
            d <= b_opt,
            m_opt * d + c0_opt,
            m_opt * b_opt + c0_opt  # fixed from your previous typo
    )
    
    return m.X, c0.X, b.X, C_model
"""