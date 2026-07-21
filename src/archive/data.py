import squidpy as sq
import numpy as np

def get_mibitof_with_spatial(crop):
    adata = sq.datasets.mibitof()
    if crop:
        adata = adata[(adata.obsm["spatial"][:, 0] > 300) 
                    & (adata.obsm["spatial"][:, 1] > 300) 
                    & (adata.obsm["spatial"][:, 0] < 600) 
                    & (adata.obsm["spatial"][:, 1] < 600)].copy()
        
    del adata.obsp["connectivities"]
    del adata.var
    del adata.obsm["X_scanorama"]
    del adata.obsm["X_umap"]
    del adata.obs
    del adata.uns
    
    sq.gr.spatial_neighbors(adata, n_neighs=2, coord_type="generic", radius=20)
    return adata


