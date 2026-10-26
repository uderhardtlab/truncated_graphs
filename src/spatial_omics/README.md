# spatial_omics — paper sections 2 (analysis buffer) and 3 (RNA diffusion QC)

Visium HD and Stereo-seq (STOmics) at 8 × 8 µm. One shared pipeline for both sections:

- `st_precompute.py` — per sample: tissue footprint from the filtered bins (`bosperrus.load_filtered`),
  piecewise-linear elbow per connected component (`identify_analysis_buffer_from_filtered`), BLADE
  (`blade.py`) on raw and corrected counts, and diffusion outside the image mask
  (`quantify_diffusion_from_raw`). Writes `results/spatial_omics/st_reroll/` (gitignored).
  Run per sample via `jobs/spatial_omics_st_precompute.sbatch <sample>`.
- `st_samples.csv` — sample list (name, technology, path, mask settings).
- Figures: `notebooks/spatial_omics/ST_demo.ipynb` (section 2, demo data) and `ST_paired.ipynb` (section 3, paired Visium HD vs. Stereo-seq); checks: `border_sanity_checks_demo.ipynb`, `border_sanity_checks_paired.ipynb`.
- Data: Visium HD demo set (11 "Protocol 2.0" datasets with H&E) in `/home/woody/iwbn/iwbn007h/visium_hd_demo_data/`.

Results, decisions and to-dos: `../../../notes/spatial_omics.md`.
