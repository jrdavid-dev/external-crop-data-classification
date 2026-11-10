"""Display-only crop previews; keep raster grids, scores and bundle identities unchanged."""
import numpy as np

from crop_pipeline.paoay import crop_baseline, maps, review

COLORS = ['#ed9137', '#4b9a65', '#8b70b1', '#e8c873', '#d9d9d9']


def _draw(axis, bundle, mask, title):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    values = bundle['data'][0].copy()
    values[bundle['data'][1] == 2] = 4
    values[bundle['data'][1] == 3] = 5
    axis.set_axis_off()
    axis.set_title(title, fontsize=11)
    if not mask.any():
        axis.text(.5, .5, 'No mapped pixel centres', ha='center', transform=axis.transAxes)
        return
    rows, columns = np.where(mask)
    # Tight limits with a small margin; no raster interpolation or resampling.
    padding = max(1, .04 * max(np.ptp(rows) + 1, np.ptp(columns) + 1))
    axis.imshow(np.ma.array(values, mask=~mask), cmap=ListedColormap(COLORS).with_extremes(bad='white'),
                norm=BoundaryNorm(np.arange(.5, 6), 5), interpolation='nearest')
    axis.set_xlim(columns.min() - .5 - padding, columns.max() + .5 + padding)
    axis.set_ylim(rows.max() + .5 + padding, rows.min() - .5 - padding)


def _legend(figure):
    from matplotlib.patches import Patch
    labels = [*crop_baseline.CROPS.values(), crop_baseline.STATUSES[2], crop_baseline.STATUSES[3]]
    figure.legend(handles=[Patch(color=color, label=label) for color, label in zip(COLORS, labels)],
                  loc='outside lower center', ncol=3, fontsize=10)


def plot_overview(bundle):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 8), layout='constrained')
    _draw(axis, bundle, bundle['roi'], 'Three-crop predictions + Unknown: reference boundaries\nZoomed display; original raster resolution unchanged')
    _legend(figure)
    return figure


def plot_boundaries(bundle):
    import matplotlib.pyplot as plt
    inputs = maps.context(bundle['manifest']['prediction_handoff'])
    reference = inputs['reference']
    areas = [r for r in reference['records'] if r['reference_id'] != reference['manifest']['roi_id']]
    columns = min(3, max(1, len(areas)))
    rows = max(1, (len(areas) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4),
                                layout='constrained', squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, area in zip(axes.flat, areas):
        mask, status = review.area_mask(area, bundle['manifest']['grid'], bundle['roi'])
        _draw(axis, bundle, mask, area['name'] if status == 'included' else area['name'] + '\n' + status.replace('_', ' '))
    figure.suptitle('Crop predictions by boundary — each panel zoomed independently\nConditional model predictions; not verified crop occupancy', fontsize=12)
    _legend(figure)
    return figure
