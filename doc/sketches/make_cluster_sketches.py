"""
Write the clean sources of the clustering sketches (src/clustering.svg, src/groups.svg). The terrain is three mountain
catchments around Piz Kesch (Switzerland), prepared by prepare_kesch.py in data/. The cells are clustered as TopoPyScale
does: features standardised, multiplied by their weights, k-means; with groups, one k-means per group with
ceil(N x area share) clusters. Then run roughen.js to get the hand-drawn figures.

    python make_cluster_sketches.py && node roughen.js

The maps of clusters are embedded as PNG images, one block of pixels per DEM cell: hundreds of small clusters would
make heavy SVG files once hand-drawn. Needs numpy, scipy, xarray, netCDF4, pillow, scikit-learn, scikit-image, rasterio
and shapely.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from rasterio import features as rio_features
from scipy import ndimage
from shapely.geometry import LineString, shape
from shapely.ops import linemerge, unary_union
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SRC = HERE / 'src'
N_DEM = 300                                       # n_clusters over the whole DEM (figure clustering)
N_MASK = 150                                      # n_clusters inside the mask (figure groups)
SEED = 2                                          # random_seed of config.yml
WEIGHTS = {'x': 1, 'y': 1, 'elevation': 1, 'slope': 1, 'aspect_cos': 1, 'aspect_sin': 1, 'svf': 1}   # defaults
UPSCALE = 6                                       # image pixels per cell
HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {h}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="13">\n<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>\n'
        '<rect width="900" height="{h}" fill="#fff"/>\n')
PASTEL = ['#f4c7a1', '#bcd7ee', '#c9e3b8', '#f1d38b', '#d9c3e6', '#f2b9b9', '#b8e0dc', '#e3d2b3']
FAMILY = {1: ['#2f6fa3', '#5b93c4', '#8db6dc', '#bcd7ee', '#dceaf6', '#4a7fae'],
          2: ['#3f7f3a', '#6aa35c', '#97c487', '#c3e0b5', '#e1f0d8', '#55904c'],
          3: ['#b8641e', '#d98b45', '#eab07a', '#f4cfa8', '#fae6d2', '#c97834']}
BASIN_TINT = {1: '#dceaf6', 2: '#e1f0d8', 3: '#fae6d2'}
BEIGE = '#f6f1ea'
MASKED = '#e4e4e4'
EDGE = '#8c8c8c'
RED = '#c0392b'
REDS = ['#c0392b', '#e8846f', '#9e2b20', '#f2a593']
LIGHT = ['#f7e3d2', '#dde9f4', '#e2efd9', '#f6ead0', '#ebe1f1', '#d9eeeb']
SUB_G = '<tspan baseline-shift="sub" font-size="10">g</tspan>'


def load():
    ds = xr.open_dataset(HERE / 'data' / 'kesch.nc')
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    feat = {'x': X, 'y': Y, 'elevation': ds.elevation.values, 'slope': ds.slope.values,
            'aspect_cos': np.cos(ds.aspect.values), 'aspect_sin': np.sin(ds.aspect.values), 'svf': ds.svf.values}
    streams = [np.asarray(LineString(line).simplify(0.5).coords)
               for line in json.loads((HERE / 'data' / 'kesch_streams.json').read_text())]
    return feat, ds.watershed.values.astype(int), streams


def kmeans(feat, cells, n):
    cols = list(WEIGHTS)
    data = np.column_stack([feat[c][cells] for c in cols])
    scaled = StandardScaler().fit_transform(data) * np.array([WEIGHTS[c] for c in cols])
    return KMeans(n_clusters=n, n_init=4, random_state=SEED).fit(scaled).labels_


def cell_polygons(labels, label):
    """Shapely polygons of the cells of a label, along the cell edges, in cell units (x = column, y = row)."""
    mask = labels == label
    return [shape(g) for g, _ in rio_features.shapes(mask.astype(np.uint8), mask=mask)]


def rings(polys, tol=0.4):
    out = []
    for poly in polys:
        poly = poly.simplify(tol, preserve_topology=True)
        for p in getattr(poly, 'geoms', [poly]):
            xy = np.asarray(p.exterior.coords)[:-1]
            if len(xy) >= 3:
                out.append(xy)
    return out


def text(x, y, s, **kw):
    attrs = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
    return f'<text x="{x}" y="{y}" {attrs}>{s}</text>'


def neighbours(labels):
    pairs = set()
    for a, b in [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]:
        diff = (a != b) & (a >= 0) & (b >= 0)
        pairs |= set(zip(a[diff].tolist(), b[diff].tolist()))
    return pairs


def colouring(labels, palette, order):
    """Greedy colouring: neighbouring clusters get different colours."""
    adj = {}
    for a, b in neighbours(labels):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    colour = {}
    for lab in order:
        used = {colour.get(b) for b in adj.get(lab, ())}
        free = [c for c in palette if c not in used]
        colour[lab] = free[0] if free else palette[lab % len(palette)]
    return colour


def hex_rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


class Map:
    """Geometry of a map panel: origin (ox, oy) and scale s [px per cell]."""

    def __init__(self, t, ox, oy, s):
        self.t, self.ox, self.oy, self.s = t, ox, oy, s
        self.w, self.h = t.nx * s, t.ny * s

    def pt(self, px, py):
        return f'{self.ox + self.s * px:.1f},{self.oy + self.s * py:.1f}'

    def d(self, ring, close=True):
        return 'M' + ' L'.join(self.pt(px, py) for px, py in ring) + (' Z' if close else '')

    def rect(self, fill):
        return f'<rect x="{self.ox}" y="{self.oy}" width="{self.w:.1f}" height="{self.h:.1f}" fill="{fill}"/>'

    def frame(self):
        return (f'<rect x="{self.ox}" y="{self.oy}" width="{self.w:.1f}" height="{self.h:.1f}" fill="none" '
                f'stroke="#333" stroke-width="1.2"/>')

    def fill(self, polys, colour):
        return [f'<path d="{self.d(r)}" fill="{colour}" stroke="none"/>' for r in rings(polys)]

    def outline(self, width=1.8):
        return [f'<path d="{self.d(r)}" fill="none" stroke="#5a3a22" stroke-width="{width}"/>'
                for r in rings([self.t.union])]

    def divides(self):
        return [f'<path d="{self.d(np.asarray(g.coords), close=False)}" fill="none" stroke="#5a3a22" '
                f'stroke-width="2.2" stroke-dasharray="6 4"/>' for g in self.t.divides]

    def streams(self, width=1.3):
        return [f'<path d="{self.d(line, close=False)}" fill="none" stroke="#2f6fa3" stroke-width="{width}"/>'
                for line in self.t.streams]

    def contours(self, levels, colour='#a08870', width=0.8):
        out = []
        z = ndimage.gaussian_filter(self.t.feat['elevation'], 1.0)
        for level in levels:
            for c in measure.find_contours(z, level):
                c = measure.approximate_polygon(c, tolerance=0.5)
                if len(c) > 3:
                    out.append(f'<path d="{self.d(c[:, ::-1] + 0.5, close=False)}" fill="none" stroke="{colour}" '
                               f'stroke-width="{width}"/>')
        return out

    def highlight(self, labels, lab):
        return [f'<path d="{self.d(r)}" fill="none" stroke="#7b1d14" stroke-width="1.6"/>'
                for r in rings(cell_polygons(labels, lab), 0.2)]

    def mosaic(self, labels, colours):
        """Clusters as an image: one block of pixels per cell, thin grey lines between clusters."""
        k = UPSCALE
        rgb = np.full(labels.shape + (3,), hex_rgb(MASKED), np.uint8)
        for lab, c in colours.items():
            rgb[labels == lab] = hex_rgb(c)
        img = np.repeat(np.repeat(rgb, k, 0), k, 1)
        big = np.repeat(np.repeat(labels, k, 0), k, 1)
        edge = np.zeros(big.shape, bool)
        edge[:, 1:] |= (big[:, 1:] != big[:, :-1])
        edge[1:, :] |= (big[1:, :] != big[:-1, :])
        edge &= big >= 0
        img[edge] = hex_rgb(EDGE)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG', optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return (f'<image x="{self.ox}" y="{self.oy}" width="{self.w:.1f}" height="{self.h:.1f}" '
                f'preserveAspectRatio="none" href="data:image/png;base64,{data}"/>')


class Terrain:
    def __init__(self):
        self.feat, self.basin, self.streams = load()
        self.ny, self.nx = self.basin.shape
        self.mask = self.basin > 0
        self.basin_polys = {b: unary_union(cell_polygons(self.basin, b)) for b in (1, 2, 3)}
        self.union = unary_union(list(self.basin_polys.values()))
        self.divides = []
        for a, b in [(1, 2), (1, 3), (2, 3)]:
            shared = self.basin_polys[a].boundary.intersection(self.basin_polys[b].boundary)
            if not shared.is_empty:
                merged = linemerge(shared) if shared.geom_type != 'LineString' else shared
                self.divides += [g.simplify(0.4) for g in getattr(merged, 'geoms', [merged]) if g.length > 1]
        # distance of each cell to the nearest divide [cells]
        b = self.basin
        edge = np.zeros(b.shape, bool)
        d = (b[:, :-1] != b[:, 1:]) & (b[:, :-1] > 0) & (b[:, 1:] > 0)
        edge[:, :-1] |= d
        edge[:, 1:] |= d
        d = (b[:-1] != b[1:]) & (b[:-1] > 0) & (b[1:] > 0)
        edge[:-1] |= d
        edge[1:] |= d
        self.dist_divide = ndimage.distance_transform_edt(~edge)

    def shares(self, labels, lab):
        counts = np.bincount(self.basin[labels == lab], minlength=4)[1:]
        return counts / counts.sum()

    def label_point(self, b):
        p = self.basin_polys[b].representative_point()
        return p.x, p.y


def scattered_cluster(t, labels):
    """Among the clusters larger than average, the one whose cells lie farthest apart on the map."""
    labs = sorted(set(np.unique(labels)) - {-1})
    big = [lab for lab in labs if (labels == lab).sum() >= (labels >= 0).sum() / len(labs)]

    def spread(lab):
        cells = labels == lab
        return np.hypot(t.feat['x'][cells].std(), t.feat['y'][cells].std())
    return max(big, key=spread)


def clustering_svg(t):
    labels = kmeans(t.feat, np.ones(t.basin.shape, bool), N_DEM).reshape(t.basin.shape)
    s, oy = 2.4, 50
    h = oy + t.ny * s
    out = [HEAD.format(h=int(h + 70))]
    # (a) DEM and features
    m = Map(t, 20, oy, s)
    out.append(text(m.ox, 26, '(a) DEM: features of each cell', font_weight='bold'))
    out.append(m.rect(BEIGE))
    out += m.contours(range(1600, 3400, 100))
    out += m.streams()
    out.append(m.frame())
    out.append(text(m.ox, h + 22, 'x, y, z, slope S, cos A, sin A, V<tspan baseline-shift="sub" font-size="10">d'
                    '</tspan>'))
    out.append(text(m.ox, h + 40, 'standardised, then weighted (all 1 here)', fill='#555'))
    out.append(text(m.ox, h + 58, 'SRTM, 100 m cells, around Piz Kesch', fill='#555'))
    # (b) feature space, conceptual
    ox2 = 330
    out.append(text(ox2, 26, '(b) k-means in feature space', font_weight='bold'))
    base = h - 10
    out.append(f'<line x1="{ox2 + 20}" y1="{base}" x2="{ox2 + 250}" y2="{base}" stroke="#333" stroke-width="1.2" '
               f'marker-end="url(#a)"/>')
    out.append(f'<line x1="{ox2 + 20}" y1="{base}" x2="{ox2 + 20}" y2="{oy}" stroke="#333" stroke-width="1.2" '
               f'marker-end="url(#a)"/>')
    out.append(text(ox2 + 160, base + 20, 'elevation', fill='#555'))
    out.append(text(ox2 + 26, oy + 12, 'cos A (+1: south-facing)', fill='#555'))
    rng = np.random.default_rng(4)
    centres = [(80, 205), (150, 125), (205, 190), (120, 163)]
    cols = ['#d98b45', '#5b93c4', '#6aa35c', '#9b6fb8']
    for (cx, cy), col in zip(centres, cols):
        for dx, dy in rng.normal(0, 15, (9, 2)):
            out.append(f'<circle cx="{ox2 + cx + dx:.1f}" cy="{cy + dy:.1f}" r="3" fill="{col}" stroke="{col}"/>')
        out.append(f'<line x1="{ox2 + cx - 7}" y1="{cy - 7}" x2="{ox2 + cx + 7}" y2="{cy + 7}" stroke="#222" '
                   f'stroke-width="2"/>')
        out.append(f'<line x1="{ox2 + cx - 7}" y1="{cy + 7}" x2="{ox2 + cx + 7}" y2="{cy - 7}" stroke="#222" '
                   f'stroke-width="2"/>')
    out.append(text(ox2 + 10, h + 40, '× centroid: each cell joins the nearest one,'))
    out.append(text(ox2 + 10, h + 58, "then each centroid moves to its cells' mean", fill='#555'))
    # (c) clusters on the map
    m = Map(t, 620, oy, s)
    out.append(text(m.ox, 26, f'(c) {N_DEM} clusters on the map', font_weight='bold'))
    lab = scattered_cluster(t, labels)
    colours = colouring(labels, PASTEL, sorted(np.unique(labels)))
    colours[lab] = RED
    out.append(m.mosaic(labels, colours))
    out += m.highlight(labels, lab)
    out.append(m.frame())
    out.append(text(m.ox, h + 22, 'one point per cluster (its centroid)'))
    out.append(text(m.ox, h + 40, 'red: the cells of one cluster,', fill='#555'))
    out.append(text(m.ox, h + 58, 'scattered, share one time series', fill='#555'))
    out.append('</svg>\n')
    (SRC / 'clustering.svg').write_text('\n'.join(out))
    return {'red patches': ndimage.label(labels == lab)[1], 'red cells': int((labels == lab).sum())}


def groups_svg(t):
    area = {b: (t.basin == b).sum() / t.mask.sum() for b in (1, 2, 3)}
    n = {b: int(np.ceil(N_MASK * area[b])) for b in (1, 2, 3)}
    no_groups = np.full(t.basin.shape, -1)
    no_groups[t.mask] = kmeans(t.feat, t.mask, N_MASK)
    groups = np.full(t.basin.shape, -1)
    offset = 0
    for b in (1, 2, 3):
        cells = t.basin == b
        groups[cells] = kmeans(t.feat, cells, n[b]) + offset
        offset += n[b]
    s = 2.6
    col = [20, 470]
    row = [40, 40 + t.ny * s + 115]
    out = [HEAD.format(h=int(row[1] + t.ny * s + 75))]

    def under(m, i, line, **kw):
        return text(m.ox, f'{m.oy + m.h + 22 + 18 * i:.0f}', line, **kw)

    # (a) mask
    m = Map(t, col[0], row[0], s)
    out.append(text(m.ox, m.oy - 14, '(a) mask: catchments of interest', font_weight='bold'))
    out.append(m.rect(MASKED))
    out += m.fill([t.union], BEIGE)
    out += m.streams()
    out += m.outline(2.2)
    out.append(m.frame())
    p = t.union.representative_point()
    px, py = p.x, p.y
    out.append(f'<circle cx="{m.ox + s * px:.1f}" cy="{m.oy + s * py:.1f}" r="13" fill="white" stroke="#5a3a22" '
               f'stroke-width="1.5"/>')
    out.append(text(f'{m.ox + s * px:.1f}', f'{m.oy + s * py + 6:.1f}', '1', font_size='17', font_weight='bold',
                    text_anchor='middle', fill='#5a3a22'))
    out.append(text(f'{m.ox + 14:.1f}', f'{m.oy + m.h - 12:.1f}', '0', font_size='17', font_weight='bold',
                    fill='#888'))
    out.append(under(m, 0, '1: clustered, 0 (grey): left out'))
    out.append(under(m, 1, '(clustering_mask: mask.tif)', fill='#555'))
    # (b) groups
    m = Map(t, col[1], row[0], s)
    out.append(text(m.ox, m.oy - 14, '(b) groups: one integer per catchment', font_weight='bold'))
    out.append(m.rect(MASKED))
    for b in (1, 2, 3):
        out += m.fill([t.basin_polys[b]], BASIN_TINT[b])
    out += m.streams()
    out += m.divides()
    out += m.outline()
    out.append(m.frame())
    for b in (1, 2, 3):
        px, py = t.label_point(b)
        cx, cy = m.ox + s * px, m.oy + s * py
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="white" stroke="{FAMILY[b][0]}" '
                   f'stroke-width="1.5"/>')
        out.append(text(f'{cx:.1f}', f'{cy + 6:.1f}', str(b), font_size='17', font_weight='bold',
                        text_anchor='middle', fill=FAMILY[b][0]))
    out.append(under(m, 0, 'each group will hold many clusters'))
    out.append(under(m, 1, '(clustering_groups: groups.tif)', fill='#555'))
    # (c) mask, no groups
    m = Map(t, col[0], row[1], s)
    out.append(text(m.ox, m.oy - 14, f'(c) mask, no groups: {N_MASK} clusters', font_weight='bold'))
    order = sorted(set(np.unique(no_groups)) - {-1})
    crossing = [k for k in order if (t.shares(no_groups, k) > 0).sum() > 1]
    others = [k for k in order if k not in crossing]
    colours = colouring(np.where(np.isin(no_groups, crossing), no_groups, -1), REDS, crossing)
    colours.update(colouring(np.where(np.isin(no_groups, others), no_groups, -1), LIGHT, others))
    out.append(m.mosaic(no_groups, colours))
    out += m.divides()
    out += m.outline()
    out.append(m.frame())
    cross = len(crossing)
    out.append(under(m, 0, f'red: {cross} of {N_MASK} clusters cross a divide'))
    out.append(under(m, 1, 'each has one series for cells of 2 or 3 catchments', fill='#555'))
    # (d) mask and groups
    m = Map(t, col[1], row[1], s)
    out.append(text(m.ox, m.oy - 14, '(d) mask and groups: k-means per catchment', font_weight='bold'))
    colours = {}
    for b in (1, 2, 3):
        sub = np.where(t.basin == b, groups, -1)
        colours.update(colouring(sub, FAMILY[b], sorted(set(np.unique(sub)) - {-1})))
    out.append(m.mosaic(groups, colours))
    out += m.divides()
    out += m.outline()
    out.append(m.frame())
    sizes = ' + '.join(str(n[b]) for b in (1, 2, 3))
    out.append(under(m, 0, f'n{SUB_G} = ⌈N w{SUB_G}⌉: {sizes} = {sum(n.values())} clusters'))
    out.append(under(m, 1, f'w{SUB_G}: area share (default) or clustering_group_weights;', fill='#555'))
    out.append(under(m, 2, 'no cluster crosses a divide', fill='#555'))
    out.append('</svg>\n')
    (SRC / 'groups.svg').write_text('\n'.join(out))
    crossing_groups = sum((t.shares(groups, k) > 0).sum() > 1 for k in sorted(set(np.unique(groups)) - {-1}))
    return {'n': n, 'area': {b: round(float(a), 3) for b, a in area.items()}, 'cross': cross,
            'cross with groups': crossing_groups}


if __name__ == '__main__':
    t = Terrain()
    print('clustering:', clustering_svg(t))
    print('groups:', groups_svg(t))
