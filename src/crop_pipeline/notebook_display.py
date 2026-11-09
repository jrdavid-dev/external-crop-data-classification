"""Readable notebook tables; presentation never modifies processing inputs."""

from collections.abc import Mapping
from numbers import Integral, Real

_LABELS = {
    "crop": "Crop", "partition": "Partition", "pixels": "Pixels",
    "parcel_groups": "Parcels", "parcel_group": "Parcel", "parcel_id": "Parcel ID",
    "parcels": "Parcels", "target_met": "Target met", "pixel_deficit": "Pixels still needed",
    "parcel_deficit": "Parcels still needed", "source_crs": "Source CRS",
    "savi_soil_adjustment": "SAVI soil adjustment", "peak_interval": "Peak interval",
}
_VALUES = {"carrot": "Carrot", "cabbage": "Cabbage", "potato": "Potato",
           "train": "Training", "validation": "Validation", "test": "Reserved test"}


def label(value):
    text = str(value)
    return _LABELS.get(text, text.replace("_", " ").capitalize())


def cell(value):
    """Format scalar values without changing their stored values."""
    import pandas as pd

    if value is None or pd.isna(value):
        return "Not available"
    if pd.api.types.is_bool(value):
        return "Yes" if value else "No"
    if isinstance(value, Integral):
        return f"{value:,}"
    if isinstance(value, Real):
        return format(value, ",g")
    return _VALUES.get(str(value), str(value))


def table_frame(records):
    """Copy a record table and label its columns, retaining row order and values."""
    import pandas as pd

    frame = records.copy(deep=True) if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if any(name is not None for name in frame.index.names):
        frame = frame.reset_index()
    return frame.rename(columns=label)


def show_table(records, title=None):
    """Show HTML in a notebook and a readable plain-text table in a terminal."""
    from IPython import get_ipython
    from IPython.display import HTML, display
    from html import escape

    frame = table_frame(records)
    if frame.empty:
        print(f"{title}: no records." if title else "No records.")
        return
    if get_ipython() is None:
        if title:
            print(title)
        print(frame.to_string(index=False, formatters={column: cell for column in frame.columns}))
    else:
        html = frame.to_html(index=False, border=0,
                             formatters={column: cell for column in frame.columns}, escape=True)
        if title:
            opening, separator, body = html.partition(">")
            html = opening + separator + "<caption>" + escape(str(title)) + "</caption>" + body
        display(HTML(html))


def show_mapping(values, title=None, key_label="Item", value_label="Value", humanize_values=False):
    """Display count maps, nested audits or settings without Python/JSON notation."""
    if not values:
        show_table([], title=title)
        return
    if set(values) <= {"carrot", "cabbage", "potato"}:
        key_label = "Crop"
    if all(isinstance(value, Mapping) and all(not isinstance(item, (Mapping, list, tuple))
           for item in value.values()) for value in values.values()):
        show_table([{key_label: key, **dict(value)} for key, value in values.items()], title=title)
        return
    rows = []

    def flatten(prefix, value):
        if isinstance(value, Mapping):
            if not value:
                rows.append({key_label: prefix, value_label: "None"})
            for key, item in value.items():
                flatten(f"{prefix} / {label(key)}" if prefix else label(key), item)
        elif isinstance(value, (list, tuple)):
            if any(isinstance(item, (Mapping, list, tuple)) for item in value):
                for number, item in enumerate(value, 1):
                    flatten(f"{prefix} / {number}", item)
            else:
                rows.append({key_label: prefix, value_label: ", ".join(cell(item) for item in value) or "None"})
        else:
            if humanize_values and isinstance(value, str):
                value = value.replace("_", " ")
            rows.append({key_label: prefix, value_label: value})

    for key, value in values.items():
        flatten(_VALUES.get(str(key), label(key)), value)
    show_table(rows, title=title)


def show_items(values, title):
    print(f"{title}: " + (", ".join(str(value) for value in values) or "None"))
