"""Read KMZ reference notes as evidence, never as verified crop labels.

This module only reads local inputs and publishes a separate, verified reference
bundle. No imagery requests, model fitting or automatic ground-truth assignment.
"""

import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform
from shapely.validation import explain_validity

FORMAT_VERSION = 1
FILES = {
    "roi.geojson", "reference_areas.geojson", "notes.json", "source_inputs.json",
    "area_audit.csv", "overlaps.csv", "review_template.csv",
}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class _NoteText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"br", "p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def note_text(description):
    parser = _NoteText()
    parser.feed(description)
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _child(element, name):
    return next((child for child in element if _tag(child) == name), None)


def _text(element, name):
    child = _child(element, name)
    return "".join(child.itertext()).strip() if child is not None else ""


def _ring(boundary):
    coordinates = next((node for node in boundary.iter() if _tag(node) == "coordinates"), None)
    if coordinates is None or not coordinates.text:
        raise ValueError("Polygon ring has no coordinates")
    points = []
    for token in coordinates.text.split():
        parts = token.split(",")
        if len(parts) < 2:
            raise ValueError("Polygon coordinates need longitude and latitude")
        longitude, latitude = map(float, parts[:2])
        if not (math.isfinite(longitude) and math.isfinite(latitude)
                and -180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError("Coordinates are outside finite WGS84 bounds")
        points.append((longitude, latitude))
    if len(set(points)) < 3:
        raise ValueError("Polygon ring needs at least three distinct positions")
    return points


def _geometry(placemark):
    polygons = []
    for element in placemark.iter():
        if _tag(element) != "Polygon":
            continue
        outer = _child(element, "outerBoundaryIs")
        if outer is None:
            raise ValueError("Polygon has no outer boundary")
        holes = [_ring(child) for child in element if _tag(child) == "innerBoundaryIs"]
        polygons.append(Polygon(_ring(outer), holes))
    if not polygons:
        raise ValueError("Placemark has no polygon geometry")
    geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    if geometry.is_empty:
        raise ValueError("Empty polygon")
    return geometry


def read_inputs(paths):
    """Read KML members in memory; preserve HTML notes, metadata and invalid-area flags."""
    paths = [Path(path).expanduser().resolve() for path in paths]
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ValueError("Supply reference files with distinct filenames")
    records, sources = [], []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reference input missing: {path}. Set paoay_kmz_root or copy the KMZ files to data/paoay.")
        source = {"name": path.name, "path": str(path), "sha256": checksum(path), "bytes": path.stat().st_size}
        sources.append(source)
        if path.suffix.lower() == ".kmz":
            with zipfile.ZipFile(path) as archive:
                members = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".kml")]
                if not members or sum(entry.file_size for entry in members) > 64 * 1024**2:
                    raise ValueError("KMZ must contain KML with a combined uncompressed size under 64 MiB")
                documents = [(entry.filename, archive.read(entry)) for entry in members]
        elif path.suffix.lower() == ".kml":
            if path.stat().st_size > 64 * 1024**2:
                raise ValueError("KML is larger than 64 MiB")
            documents = [(path.name, path.read_bytes())]
        else:
            raise ValueError("Reference inputs must be KML or KMZ files")
        for member, data in documents:
            root = ET.fromstring(data)
            for number, placemark in enumerate(node for node in root.iter() if _tag(node) == "Placemark"):
                name = _text(placemark, "name") or f"Unnamed area {number + 1}"
                description = _text(placemark, "description")
                try:
                    geometry = _geometry(placemark)
                    geometry_json = mapping(geometry)
                    geometry_status = "valid" if geometry.is_valid else explain_validity(geometry)
                except (ValueError, TypeError) as error:
                    geometry_json, geometry_status = None, str(error)
                metadata = {}
                for element in placemark.iter():
                    if _tag(element) == "Data":
                        metadata[element.get("name", "unnamed")] = _text(element, "value")
                    elif _tag(element) == "SimpleData":
                        metadata[element.get("name", "unnamed")] = element.text or ""
                times = {tag: ET.tostring(element, encoding="unicode") for tag in ("TimeStamp", "TimeSpan")
                         if (element := _child(placemark, tag)) is not None}
                identity = {"source": path.name, "member": member, "index": number,
                            "name": name, "geometry": geometry_json}
                reference_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
                records.append({"reference_id": reference_id, "name": name, "source_file": path.name,
                    "kml_member": member, "description_html": description, "notes": note_text(description),
                    "extended_data": metadata, "time_metadata": times,
                    "geometry": geometry_json, "geometry_status": geometry_status})
        if checksum(path) != source["sha256"]:
            raise ValueError("Reference file changed while it was being read; retry with a stable copy")
    if not records:
        raise ValueError("No placemarks found in reference inputs")
    return records, sources


def inspect_records(records, roi_name="Region of Interest"):
    """Inspect geometry without repairing it or inferring crop/year from notes."""
    boundaries = [record for record in records if record["name"].strip().casefold() == roi_name.strip().casefold()]
    if len(boundaries) != 1 or boundaries[0]["geometry_status"] != "valid":
        raise ValueError("Choose exactly one valid study-boundary polygon using roi_name")
    roi = boundaries[0]
    roi_shape = shape(roi["geometry"])
    centre = roi_shape.representative_point()
    zone = min(60, max(1, int((centre.x + 180) / 6) + 1))
    metric_crs = f"EPSG:{(32600 if centre.y >= 0 else 32700) + zone}"
    project = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    roi_projected = transform(project, roi_shape)
    audit, projected = [], {}
    for record in records:
        boundary = record["reference_id"] == roi["reference_id"]
        area, location = None, "unavailable"
        if record["geometry_status"] == "valid":
            geometry = transform(project, shape(record["geometry"]))
            area = geometry.area / 10000
            if not boundary:
                projected[record["reference_id"]] = geometry
            location = "inside" if roi_projected.covers(geometry) else "partly outside" if roi_projected.intersects(geometry) else "outside"
        audit.append({"reference_id": record["reference_id"], "name": record["name"],
            "source_file": record["source_file"], "role": "Study boundary" if boundary else "Reference area",
            "area_ha": area, "geometry_status": record["geometry_status"], "location": location,
            "notes_present": bool(record["notes"]), "evaluation_status": "Not applicable" if boundary else "Unverified"})
    overlaps = []
    areas = [record for record in records if record["reference_id"] in projected]
    for index, first in enumerate(areas):
        for second in areas[index + 1:]:
            intersection = projected[first["reference_id"]].intersection(projected[second["reference_id"]])
            if intersection.area > 0.01:
                overlaps.append({"first_id": first["reference_id"], "first_area": first["name"],
                    "second_id": second["reference_id"], "second_area": second["name"],
                    "overlap_ha": intersection.area / 10000})
    return {"roi": roi, "metric_crs": metric_crs, "audit": audit, "overlaps": overlaps}


def _review_file(folder, manifest):
    return folder.parent.parent / "reviews" / f"{manifest['bundle_id']}.csv"


def export_reference_bundle(records, sources, output_root, roi_name="Region of Interest"):
    """Publish an immutable snapshot; keep editable annotations separately and never overwrite them."""
    inspection = inspect_records(records, roi_name)
    for source in sources:
        if checksum(source["path"]) != source["sha256"]:
            raise ValueError("Reference input changed after review; read it again before saving")
    identity = {"version": FORMAT_VERSION, "sources": [{"name": source["name"], "sha256": source["sha256"]} for source in sources],
                "roi_name": roi_name, "implementation_sha256": checksum(__file__)}
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output_root = Path(output_root)
    folder = output_root / "bundles" / bundle_id
    manifest_path = folder / "reference_handoff.json"
    if folder.exists():
        bundle = load_reference_bundle(manifest_path)
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".references-", dir=folder.parent) as temporary:
            staging = Path(temporary) / bundle_id
            staging.mkdir()
            roi_id = inspection["roi"]["reference_id"]
            features = [{"type": "Feature", "geometry": record["geometry"], "properties": {
                "reference_id": record["reference_id"], "name": record["name"], "source_file": record["source_file"],
                "notes": record["notes"], "geometry_status": record["geometry_status"],
                "evidence_status": "Unverified", "role": "Study boundary" if record["reference_id"] == roi_id else "Reference area",
            }} for record in records]
            _json(staging / "roi.geojson", {"type": "FeatureCollection", "features": [feature for feature in features if feature["properties"]["reference_id"] == roi_id]})
            _json(staging / "reference_areas.geojson", {"type": "FeatureCollection", "features": [feature for feature in features if feature["properties"]["reference_id"] != roi_id]})
            _json(staging / "notes.json", records)
            _json(staging / "source_inputs.json", sources)
            _csv(staging / "area_audit.csv", inspection["audit"], list(inspection["audit"][0]))
            _csv(staging / "overlaps.csv", inspection["overlaps"], ["first_id", "first_area", "second_id", "second_area", "overlap_ha"])
            review_rows = [{"reference_id": record["reference_id"], "name": record["name"], "crop": "",
                "observation_year": "", "observation_date": "", "planting_date": "", "harvest_date": "",
                "evidence_source": "", "verification_status": "unreviewed", "review_notes": ""}
                for record in records if record["reference_id"] != roi_id]
            _csv(staging / "review_template.csv", review_rows, list(review_rows[0]) if review_rows else ["reference_id", "name", "crop", "observation_year", "observation_date", "planting_date", "harvest_date", "evidence_source", "verification_status", "review_notes"])
            manifest = {"format_version": FORMAT_VERSION, "kind": "paoay_reference_review", "bundle_id": bundle_id,
                "identity": identity, "status": "prepared", "roi_id": roi_id, "metric_crs": inspection["metric_crs"],
                "area_count": len(records) - 1, "invalid_areas": sum(row["geometry_status"] != "valid" for row in inspection["audit"]),
                "overlap_pairs": len(inspection["overlaps"]), "evaluation_ready": False,
                "files": {name: {"sha256": checksum(staging / name)} for name in sorted(FILES)}}
            _json(staging / "reference_handoff.json", manifest)
            staging.rename(folder)
        bundle = load_reference_bundle(manifest_path)
    review_file = _review_file(folder, bundle["manifest"])
    review_file.parent.mkdir(parents=True, exist_ok=True)
    if not review_file.exists():
        with review_file.open("x", encoding="utf-8", newline="") as stream:
            stream.write((folder / "review_template.csv").read_text(encoding="utf-8"))
    return manifest_path


def load_reference_bundle(manifest_path):
    """Load only the verified snapshot; original KMZs and mutable review edits are not required."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") != "paoay_reference_review"
            or manifest.get("status") != "prepared" or set(manifest.get("files", {})) != FILES):
        raise ValueError("Unsupported Paoay reference handoff or incomplete inventory")
    folder = manifest_path.parent
    for name, record in manifest["files"].items():
        if not (folder / name).is_file() or checksum(folder / name) != record["sha256"]:
            raise ValueError(f"Saved reference file changed: {name}; previous outputs were preserved")
    values = {name: json.loads((folder / name).read_text(encoding="utf-8"))
              for name in ("roi.geojson", "reference_areas.geojson", "notes.json", "source_inputs.json")}
    with (folder / "area_audit.csv").open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    with (folder / "overlaps.csv").open(encoding="utf-8", newline="") as stream:
        overlaps = list(csv.DictReader(stream))
    return {"manifest": manifest, "folder": folder, "roi": values["roi.geojson"],
            "references": values["reference_areas.geojson"], "records": values["notes.json"],
            "sources": values["source_inputs.json"], "audit": audit, "overlaps": overlaps,
            "review_file": _review_file(folder, manifest)}


def plot_reference_areas(records, roi_id, metric_crs):
    """Preview local outlines only; do not request map tiles or infer field labels."""
    import matplotlib.pyplot as plt

    project = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    figure, axis = plt.subplots(figsize=(9, 7))
    for record in records:
        if record["geometry_status"] != "valid":
            continue
        geometry = transform(project, shape(record["geometry"]))
        boundary = record["reference_id"] == roi_id
        for polygon in [geometry] if geometry.geom_type == "Polygon" else geometry.geoms:
            x, y = polygon.exterior.xy
            axis.plot(x, y, color="firebrick" if boundary else "steelblue", linewidth=2 if boundary else 1)
            for hole in polygon.interiors:
                x, y = hole.xy
                axis.plot(x, y, color="steelblue", linewidth=1)
        if not boundary:
            position = geometry.representative_point()
            axis.annotate(record["name"], (position.x, position.y), fontsize=8)
    axis.set(aspect="equal", xlabel=f"Easting (m), {metric_crs}", ylabel="Northing (m)",
             title="Paoay input polygons — notes are unverified references")
    axis.ticklabel_format(style="plain", useOffset=False)
    figure.tight_layout()
    return figure
