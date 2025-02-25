import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def fit_log(d, C):
    d_nonzero = d[d > 0]
    C_nonzero = C[d > 0]  
    d_transformed = np.log(d_nonzero).reshape(-1, 1)  
    model = LinearRegression()
    model.fit(d_transformed, C_nonzero)
    a = model.coef_[0]
    b = model.intercept_
    return a, b, a * np.log(d_nonzero) + b

    
def fit_piece_wise_linear(d, C, M=1000):
    n = len(d) 
    
    model = gp.Model()
    
    m = model.addVar(vtype=GRB.CONTINUOUS, name="m") # Steigung vor Ellebogen
    c0 = model.addVar(vtype=GRB.CONTINUOUS, name="c0") # y-Achsenabschnitt C(0)
    b = model.addVar(vtype=GRB.CONTINUOUS, lb=min(d), ub=max(d), name="b") # Ellebogen
    
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    epsilon = model.addVars(n, vtype=GRB.CONTINUOUS, name="epsilon")
    
    model.setObjective(gp.quicksum(epsilon[i] * epsilon[i] for i in range(n)), GRB.MINIMIZE)

    model.setParam('OptimalityTol', 1e-4) 
    model.setParam('MIPGap', 0.01)  
    
    for i in range(n):
        # 1. wenn z[i] = 1 (d_i <= b, links von Ellebogen), dann epsilon_i >= m * d_i + c0 - C_i
        model.addConstr(epsilon[i] >= (m * d[i] + c0 - C[i]) - (1 - z[i]) * M)
        model.addConstr(epsilon[i] >= -(m * d[i] + c0 - C[i]) - (1 - z[i]) * M)
    
        # 2. wenn z[i] = 0 (d_i > b, rechts von Ellebogen), dann epsilon_i >= m * b + c0 - C_i
        model.addConstr(epsilon[i] >= (m * b + c0 - C[i]) - z[i] * M)
        model.addConstr(epsilon[i] >= -(m * b + c0 - C[i]) - z[i] * M)
    
        # binärer "Switch"
        model.addConstr(z[i] * M >= b - d[i])
        model.addConstr((1 - z[i]) * M >= d[i] - b)
    
    model.optimize()
    return m.X, c0.X, b.X


def plot_piece_wise_linear(d, C, m_opt, c0_opt, b_opt, measure, n, t, path):
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
    d_nonzero = d[d > 0]
    C_nonzero = C[d > 0]  
    plt.scatter(d_nonzero, C_nonzero, label="Original")
    plt.plot(d_nonzero, f, color="red", label=f"Fitted curve: {a:.2f} * log(x) + {b:.2f}")
    
    plt.ylabel(f"{measure.capitalize()} centrality")
    plt.xlabel("Distance to border")
    plt.legend()
    if path:
        plt.savefig(path)