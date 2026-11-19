"""Presentation-only charts from saved pipeline tables; never select or fit models."""
import math

CROP_COLORS = {"carrot": "#D55E00", "cabbage": "#009E73", "potato": "#0072B2"}


def plot_usable_samples(records):
    """Separate cohort/partition panels: counts must never be added across cohorts."""
    import matplotlib.pyplot as plt
    rows = list(records)
    cohorts = list(dict.fromkeys(r['cohort'] for r in rows))
    if not cohorts:
        raise ValueError('No usable-sample records to plot')
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(11, 3 * len(cohorts)), squeeze=False,
                             layout='constrained')
    for axes_row, cohort in zip(axes, cohorts):
        subset = [r for r in rows if r['cohort'] == cohort]
        labels = [f"{r['crop'].title()} · {r['partition']}" for r in subset]
        for ax, metric, title in zip(axes_row, ('usable_parcel_groups', 'usable_pixels'), ('Parcels', 'Pixels')):
            values = [int(r[metric]) for r in subset]
            bars = ax.barh(labels, values, color=[CROP_COLORS.get(r['crop'], '#777777') for r in subset])
            ax.bar_label(bars, padding=3, fontsize=8)
            ax.set(xlabel=title, title=f"{cohort.replace('_', ' ').title()} · {title}")
            ax.set_xlim(0, max(values + [1]) * 1.22)
            ax.invert_yaxis()
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Usable samples after quality checks and parcel cap · cohorts overlap')
    return fig


def plot_validation(results):
    """Show every saved leaf-size candidate on a common 0–1 score scale."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    labels, scores, colors = [], [], []
    for cohort, result in results.items():
        for candidate in result['candidates']:
            score = candidate['metrics'].get('macro_f1')
            if score is None:
                continue
            leaf = candidate['min_samples_leaf']
            labels.append(f"{cohort.replace('_', ' ')} · leaf {leaf}")
            scores.append(score)
            colors.append('#0072B2' if leaf == result['selected_min_samples_leaf'] else '#999999')
    bars = ax.barh(labels, scores, color=colors)
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set(xlim=(0, 1.12), xlabel='Validation parcel macro-F1',
           title='Validation candidates · blue marks the saved selection')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    if not labels:
        ax.text(.5, .5, 'No available validation scores', ha='center', transform=ax.transAxes)
    return fig


def plot_confusion(records, population='jecam'):
    """Count matrices, common crop order, explicit population and sample sizes."""
    import matplotlib.pyplot as plt
    rows = [r for r in records if r['test_population'] == population]
    cohorts = list(dict.fromkeys(r['cohort'] for r in rows))
    if not cohorts:
        raise ValueError(f'No confusion matrices for {population}')
    crops = ('carrot', 'cabbage', 'potato')
    fig, axes = plt.subplots(math.ceil(len(cohorts) / 2), 2,
                             figsize=(10, 4 * math.ceil(len(cohorts) / 2)), squeeze=False,
                             layout='constrained')
    vmax = max([int(r['parcels']) for r in rows] + [1])
    for ax, cohort in zip(axes.flat, cohorts):
        counts = {(r['actual'], r['predicted']): int(r['parcels']) for r in rows if r['cohort'] == cohort}
        matrix = [[counts.get((a, p), 0) for p in crops] for a in crops]
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=vmax)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha='center', va='center', color='white' if value > vmax / 2 else 'black')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=[c.title() for c in crops],
               yticklabels=[c.title() for c in crops], xlabel='Predicted crop', ylabel='Reference crop',
               title=f"{cohort.replace('_', ' ').title()} · n={sum(map(sum, matrix))} parcels")
    for ax in list(axes.flat)[len(cohorts):]:
        ax.set_visible(False)
    fig.suptitle(f'{population} · parcel confusion counts (zero can mean no reference support)')
    return fig
