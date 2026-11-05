"""
Cell centroids and volumes for the Arabidopsis shoot apical meristem data of
Jackson et al. 2019 (Cell Systems 8:53, "Global topological order emerges
through local mechanical control of cell divisions in the Arabidopsis shoot
apical meristem"; OSF https://osf.io/mxgrh/).

The published CSVs give the cellular connectivity network (`*_network.csv`)
and, for the central cells only, betweenness, division etc.
(`*_topology.csv`), but no positions. Positions come from the MorphoGraphX
label stacks (`.mgxs`), read here directly from the downloaded zips.

.mgxs 1.3 layout as used here: b"MGXS 1.3 ", a 1-byte label flag, 3 float32
(origin), 3 uint32 (size x, y, z), 3 float32 (voxel size in µm), then one or
more zlib streams holding the uint16 label volume (z, y, x).

Per cell it also computes the geometry the paper reports for the central
cells, so that all cells can be analysed (checked against their
`*_topology.csv` on WT Rep 4 T1, Spearman): volume = voxel count x voxel
volume (0.994), area = marching-cubes surface (0.991), anisotropy = largest
eigenvalue / sum of eigenvalues of the voxel coordinate covariance (0.995).
Layer: along every (x, y) column, cells are numbered in the order they are
met; each cell gets its most frequent number, counted from the top and from
the bottom of the stack (which end is the meristem surface is checked
against the published layers in the notebook).

Writes results/physically_realized_graphs/jackson2019/centroids.csv: genotype,
rep, time, label, x, y, z (µm, relative to the stack corner), volume (µm^3),
area (µm^2), anisotropy, layer_from_start, layer_from_end.

Run:
    cd truncated_graphs/
    pixi run python src/physically_realized_graphs/jackson_centroids.py
"""
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "jackson2019"  # WT_Arabidopsis.zip, ktn1_Arabidopsis.zip from OSF
OUT_CSV = ROOT / "results" / "physically_realized_graphs" / "jackson2019" / "centroids.csv"


def read_mgxs(raw):
    """Return the label volume (z, y, x) and the voxel size (x, y, z) in µm."""
    assert raw[:9] == b"MGXS 1.3 ", raw[:9]
    size = struct.unpack("<3I", raw[22:34])
    voxel = np.array(struct.unpack("<3f", raw[34:46]))
    data, pos = bytearray(), raw.find(b"\x78\x5e", 46)
    while pos >= 0:
        stream = zlib.decompressobj()
        data += stream.decompress(raw[pos:])
        rest = stream.unused_data
        nxt = rest.find(b"\x78\x5e") if rest else -1
        pos = len(raw) - len(rest) + nxt if nxt >= 0 else -1
    n = size[0] * size[1] * size[2]
    return np.frombuffer(bytes(data[: 2 * n]), dtype="<u2").reshape(size[2], size[1], size[0]), voxel


def column_layers(vol):
    """Per voxel, the order in which its cell is met along the z column (1 = first)."""
    nz = vol > 0
    z = np.arange(vol.shape[0])[:, None, None]
    last = np.maximum.accumulate(np.where(nz, z, 0), axis=0)  # forward-fill the last labelled voxel
    filled = np.take_along_axis(vol, last, axis=0)
    previous = np.concatenate([np.zeros_like(filled[:1]), filled[:-1]])
    change = nz & (vol != previous)
    return np.where(nz, np.cumsum(change, axis=0), 0)


def per_label_mode(vol, values, labels):
    pairs = vol.astype(np.int64) * 16 + np.minimum(values, 15)
    counts = np.bincount(pairs[vol > 0].ravel(), minlength=(int(vol.max()) + 1) * 16).reshape(-1, 16)
    return counts[labels].argmax(axis=1)


def cell_geometry(vol, voxel, labels):
    rows, objects = [], ndimage.find_objects(vol)
    for lab in labels:
        mask = np.pad(vol[objects[lab - 1]] == lab, 1)
        verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), 0.5, spacing=tuple(voxel[::-1]))
        zz, yy, xx = np.nonzero(mask)
        ev = np.linalg.eigvalsh(np.cov(np.column_stack([xx, yy, zz]) * voxel, rowvar=False)) if len(zz) > 3 else np.ones(3)
        rows.append((mesh_surface_area(verts, faces), ev.max() / ev.sum()))
    return np.array(rows)


def stack_centroids(vol, voxel):
    labels = np.unique(vol)
    labels = labels[labels > 0]
    com = np.array(ndimage.center_of_mass(np.ones_like(vol, dtype=np.uint8), vol, labels))
    counts = ndimage.sum_labels(np.ones_like(vol, dtype=np.uint32), vol, labels)
    xyz = com[:, ::-1] * voxel
    geometry = cell_geometry(vol, voxel, labels)
    return pd.DataFrame({"label": labels, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                         "volume": counts * voxel.prod(), "area": geometry[:, 0], "anisotropy": geometry[:, 1],
                         "layer_from_start": per_label_mode(vol, column_layers(vol), labels),
                         "layer_from_end": per_label_mode(vol, column_layers(vol[::-1])[::-1], labels)})


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for zpath in sorted(DATA_DIR.glob("*_Arabidopsis.zip")):
        with zipfile.ZipFile(zpath) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".mgxs")):
                genotype, rep, time = name.split("/")[:3]
                df = stack_centroids(*read_mgxs(z.read(name)))
                df.insert(0, "time", time)
                df.insert(0, "rep", rep)
                df.insert(0, "genotype", genotype.split()[0])
                frames.append(df)
                print(name, len(df), "labels", flush=True)
    pd.concat(frames).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
