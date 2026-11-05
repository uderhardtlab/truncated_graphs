# physically_realized_graphs — measured networks truncated by the imaging volume

Networks whose edges are measured (synapses, shared cell walls), not constructed from coordinates by a rule. The imaging volume truncates them: nodes and edges outside it exist but are not observed.

- **MICrONS connectome** (analysis-buffer panel):
    - `micron_comparison.py` loads the MICrONS mm³ connectome (`data/reimann_data/`), fits degree vs. distance to the alpha shape and compares the BOSPERRUS buffer with the authors' heuristic.
    - Notebook: `notebooks/physically_realized_graphs/MICRONS_analysis_buffer.ipynb`; `notebooks/overview/boundary_effects.ipynb` uses it too.
    - Further MICrONS code and data: `/home/woody/iwbn/iwbn007h/MICrONS_data/`.
- **Arabidopsis shoot apical meristem** (Jackson et al. 2019, Cell Systems; cell-contact networks, OSF https://osf.io/mxgrh/):
    - `jackson_centroids.py` reads the MorphoGraphX `.mgxs` stacks from the OSF zips in `data/jackson2019/` and writes per-cell centroids, volume, area, anisotropy and layer to `results/physically_realized_graphs/jackson2019/centroids.csv`.
    - Notebook: `notebooks/physically_realized_graphs/jackson2019_meristem.ipynb`. SLURM job for both: `jobs/physically_realized_graphs_jackson2019.sbatch`.

Results, decisions and to-dos: `../../../notes/physically_realized_graphs.md`.
