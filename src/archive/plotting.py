import anndata as ad
from itertools import combinations
import networkx as nx
from matplotlib import animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import NearestNeighbors
from matplotlib import rc
rc('animation', html='jshtml')


def add_sig_bar(ax, x1, x2, y, text, line_height=0.02):
    """
    Draw a significance bar with asterisks between x1 and x2.
    
    Parameters
    ----------
    ax : matplotlib axis
    x1, x2 : positions of the categories (0, 1, 2, ...)
    y : height of the bar
    text : significance label ("*", "**", "***", "ns")
    line_height : vertical padding
    """
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1.2, c='black')
    ax.text((x1 + x2) * 0.5, y + line_height, text, ha='center', va='bottom',  color='black')
 

# in notebook: ani = make_ani(nodes, edges) anis