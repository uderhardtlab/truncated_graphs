# spatial_omics — paper sections 2 (analysis buffer) and 3 (RNA diffusion QC)

Visium HD and Stereo-seq (STOmics) at 8 × 8 µm. One shared pipeline for both sections:

- `st_precompute.py` — per sample: diffusion outside the image mask (`quantify_diffusion_from_raw`), and the
  border analysis: rising-only piecewise-linear elbow per connected component
  (`identify_analysis_buffer_from_filtered(rising_only=True)`) vs. distance to the tissue edge of the image
  mask, fitted within 250 µm (`--image-mask`; Space Ranger `in_tissue` bins for the supplement: `--in-tissue`;
  elbow only at another bin size for the 8 vs. 2 µm check: `--tissue-edge --resolution R`), plus BLADE
  (`blade.py`) on raw and corrected counts; spots discarded by BLADE (`in_tissue`) vs. bosperrus (image border)
  (`--blade-image-border`). Every elbow is tested against interior windows of its own component
  (`bosperrus.interior_window_test`: 500 µm windows, steepness statistic, non-significant buffers dropped,
  components < ~2 mm untested). Writes `results/spatial_omics/st_reroll/` (gitignored).
  Run per sample via `jobs/spatial_omics_st_precompute.sbatch <sample> [mode]`, or all border modes of one sample
  via `jobs/spatial_omics_window_test.sbatch <sample> demo|paired`.
- `st_samples.csv` — sample list (name, technology, path, mask settings).
- Figures: `figure_plotting/spatial_omics/ST_demo.ipynb` (section 2, demo data) and `ST_paired.ipynb` (section 3, paired Visium HD vs. Stereo-seq); checks: `border_sanity_checks_demo.ipynb`, `border_sanity_checks_paired.ipynb`.
- Data: Visium HD demo set (11 "Protocol 2.0" datasets with H&E, 8 included) in `/home/woody/iwbn/iwbn007h/visium_hd_demo_data/`.

Results, decisions and to-dos: `../../../notes/spatial_omics.md`.
