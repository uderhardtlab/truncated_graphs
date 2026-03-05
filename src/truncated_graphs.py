import numpy as np
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error
import pandas as pd
from scipy import stats
from scipy.spatial import distance


def distance_to_border(coords):
    if coords.shape[1] != 2:
        raise ValueError("Spatial coordinates must be Nx2.")
    x = coords[:, 0]
    y = coords[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # distances to each of the four borders
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # distance to the rectangle boundary = smallest distance to any border
    d_border = np.vstack([d_left, d_right, d_bottom, d_top]).min(axis=0)

    return pd.Series(d_border, name="distance_to_border")
    
    
def distance_to_mask(coords, mask):
    #TODO
    pass

