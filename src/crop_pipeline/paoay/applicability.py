"""Area of applicability of the fixed crop model: a diagnostic flag, never a filter.

Follows Meyer & Pebesma (2021), "Predicting into unknown space? Estimating the area of
applicability of spatial prediction models" (Methods in Ecology and Evolution 12, 1620-1633):

1. Standardize each predictor with the training mean and standard deviation, then weight it
   by the fitted model's feature importance.
2. Dissimilarity index (DI) of a pixel = distance to its nearest training pixel, divided by
   the mean distance between training pixels.
3. Threshold = the outlier-removed maximum DI of the training pixels, where each training
   pixel may only use neighbours from a different fold.

This model was selected on one train/validation split, without cross-validation folds, so the
folds are the training parcel groups: a training pixel is compared with other parcels only.
Predictions are not changed. Inside the area of applicability means "resembles the training
pixels in the weighted predictor space"; it does not mean the prediction is correct.
"""

import csv
import json
from pathlib import Path
import tempfile

from affine import Affine
import numpy as np
import rasterio

from crop_pipeline.paoay import features, imagery, predictions, references

VERSION = 1
KIND = "paoay_model_applicability"
RULES = {"scaling": "training_mean_sd_then_importance_weight", "distance": "euclidean",
         "folds": "training_partition_group", "threshold": "outlier_removed_maximum_training_di",
         "outlier_iqr_multiplier": 1.5}
FILES = {"applicability.tif", "reference_di.npz", "summary.csv"}
STATUS = {0: "Outside ROI or no model features", 1: "Inside area of applicability", 2: "Outside area of applicability"}


def weighted(matrix, mean, scale, weights):
    return (np.asarray(matrix, dtype="float64") - mean) / scale * weights


def training_reference(train, groups, weights, iqr_multiplier=1.5):
    """Scaling, mean training distance, fold-wise training DI and the applicability threshold."""
    from scipy.spatial.distance import cdist

    train = np.asarray(train, dtype="float64")
    groups = np.asarray(groups)
    weights = np.asarray(weights, dtype="float64")
    if train.ndim != 2 or len(train) != len(groups) or train.shape[1] != len(weights):
        raise ValueError("Training matrix, groups and weights disagree")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not weights.sum() > 0:
        raise ValueError("Training features and importance weights must be finite, with a positive total weight")
    if len(set(groups.tolist())) < 2:
        raise ValueError("Applicability needs training pixels from at least two parcel groups")
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)  # A constant predictor cannot separate pixels.
    space = weighted(train, mean, scale, weights)
    distances = cdist(space, space)
    mean_distance = float(distances[np.triu_indices(len(space), 1)].mean())
    if not mean_distance > 0:
        raise ValueError("Training pixels are identical in the weighted predictor space")
    distances[groups[:, None] == groups[None, :]] = np.inf
    training_di = distances.min(axis=1) / mean_distance
    lower, upper = np.percentile(training_di, [25, 75])
    threshold = float(training_di[training_di <= upper + iqr_multiplier * (upper - lower)].max())
    return {"mean": mean, "scale": scale, "weights": weights, "space": space,
            "mean_distance": mean_distance, "training_di": training_di, "threshold": threshold}


def dissimilarity(reference, matrix, chunk_size=4096):
    """DI of new pixels: nearest training pixel distance over the mean training distance."""
    from sklearn.neighbors import NearestNeighbors

    matrix = np.asarray(matrix, dtype="float64").reshape(-1, len(reference["weights"]))
    if not len(matrix):
        return np.empty(0)
    search = NearestNeighbors(n_neighbors=1).fit(reference["space"])
    parts = []
    for start in range(0, len(matrix), chunk_size):
        block = weighted(matrix[start:start + chunk_size], reference["mean"], reference["scale"], reference["weights"])
        parts.append(search.kneighbors(block)[0][:, 0])
    return np.concatenate(parts) / reference["mean_distance"]


def locate_training_rows(model_manifest):
    """Find the exact development rows behind a frozen model: (path, cohort key, reason if unavailable)."""
    model_manifest = Path(model_manifest)
    manifest = json.loads(model_manifest.read_text())
    recipe = manifest.get("identity", {}).get("recipe", {})
    cohort, window = manifest.get("experiment_cohort"), recipe.get("window_recipe", {}).get("type")
    if cohort not in {"full", "matched"} or window not in {"fixed", "aligned"}:
        return None, None, "this model package does not record a training cohort"
    snapshot = Path(manifest["feature_handoff"])
    if not snapshot.is_file():
        return None, None, "the model's development record is missing"
    wanted = json.loads(snapshot.read_text()).get("feature_sha256")
    candidates = []
    for folder in (model_manifest.parent.parent.parent, snapshot.parent.parent):
        candidates += [folder / "features_development.json", *sorted(folder.glob("*.json"))]
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if references.checksum(path) == wanted:
            return path, f"{window}_{cohort}", None
    return None, None, "the development feature rows that trained this model were not found beside it"


def _training_arrays(path, cohort, model_manifest, columns):
    manifest = json.loads(Path(model_manifest).read_text())
    rows = json.loads(Path(path).read_text())["rows"][cohort]
    split = manifest["development_split"]
    if any(split.get(row["partition_group"]) != row["partition"] for row in rows):
        raise ValueError("Development rows differ from the model's recorded training/validation split")
    def arrays(partition):
        chosen = [row for row in rows if row["partition"] == partition]
        return (np.asarray([[float(row[name]) for name in columns] for row in chosen], dtype="float64").reshape(-1, len(columns)),
                np.asarray([row["partition_group"] for row in chosen]))
    return arrays("train"), arrays("validation")


def prepare_request(feature_handoff, output_root, training_rows=None):
    feature = features.load_feature_bundle(feature_handoff)
    model_manifest = Path(feature["manifest"]["model_manifest"])
    if training_rows is None:
        path, cohort, reason = locate_training_rows(model_manifest)
        if path is None:
            raise FileNotFoundError("Applicability unavailable: " + reason)
    else:
        path, (_, cohort, reason) = Path(training_rows), locate_training_rows(model_manifest)
        snapshot = json.loads(Path(json.loads(model_manifest.read_text())["feature_handoff"]).read_text())
        if cohort is None or references.checksum(path) != snapshot.get("feature_sha256"):
            raise ValueError("The selected training rows are not the ones recorded by this model")
    identity = {"version": VERSION, "feature_sha256": references.checksum(feature_handoff),
                "model_manifest_sha256": references.checksum(model_manifest),
                "training_rows_sha256": references.checksum(path), "cohort": cohort, "rules": RULES,
                "implementation_sha256": references.checksum(__file__)}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "feature_handoff": str(Path(feature_handoff).resolve()),
            "model_manifest": str(model_manifest.resolve()), "training_rows": str(Path(path).resolve()),
            "folder": Path(output_root) / "bundles" / key}


def build(request):
    """Save the DI raster, the reference DI distributions and a summary; reuse a verified bundle."""
    folder, manifest_path = request["folder"], request["folder"] / "applicability_handoff.json"
    if folder.exists():
        load_bundle(manifest_path)
        return manifest_path
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Applicability request changed after planning")
    feature = features.load_feature_bundle(request["feature_handoff"])
    classifier, recipe = predictions.load_classifier(request["model_manifest"])
    columns = recipe["feature_columns"]
    (train, groups), (validation, _) = _training_arrays(
        request["training_rows"], request["identity"]["cohort"], request["model_manifest"], columns)
    reference = training_reference(train, groups, classifier.feature_importances_, RULES["outlier_iqr_multiplier"])
    pixel_di = dissimilarity(reference, feature["matrix"])
    validation_di = dissimilarity(reference, validation)
    threshold = reference["threshold"]
    grid = feature["manifest"]["grid"]
    index = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    status = np.zeros(index.shape, dtype="float32")
    index[feature["rows"], feature["columns"]] = pixel_di
    status[feature["rows"], feature["columns"]] = np.where(pixel_di <= threshold, 1, 2)
    inside = pixel_di <= threshold
    summary = [
        {"pixels": "Training pixels (other parcels only)", "count": len(train),
         "median_di": float(np.median(reference["training_di"])), "inside_fraction": float((reference["training_di"] <= threshold).mean())},
        {"pixels": "Validation pixels (held-out parcels)", "count": len(validation),
         "median_di": float(np.median(validation_di)) if len(validation) else "",
         "inside_fraction": float((validation_di <= threshold).mean()) if len(validation) else ""},
        {"pixels": "Study-area pixels", "count": len(pixel_di),
         "median_di": float(np.median(pixel_di)), "inside_fraction": float(inside.mean())},
    ]
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".applicability-", dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        staging.mkdir()
        with rasterio.open(staging / "applicability.tif", "w", driver="GTiff", width=grid["width"], height=grid["height"],
                           count=2, dtype="float32", crs=grid["crs"], transform=Affine(*grid["transform"]),
                           nodata=float("nan"), compress="deflate") as raster:
            raster.write(index, 1); raster.set_band_description(1, "dissimilarity_index")
            raster.write(status, 2); raster.set_band_description(2, "applicability_status")
        np.savez_compressed(staging / "reference_di.npz", training=reference["training_di"], validation=validation_di)
        with (staging / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        manifest = {"format_version": VERSION, "kind": KIND, "status": "prepared", "bundle_id": request["bundle_id"],
            "identity": request["identity"], "feature_handoff": request["feature_handoff"],
            "model_manifest": request["model_manifest"], "training_rows": request["training_rows"],
            "grid": grid, "period": feature["manifest"]["period"], "threshold": threshold,
            "mean_training_distance": reference["mean_distance"],
            "training_pixels": len(train), "training_parcel_groups": len(set(groups.tolist())),
            "validation_pixels": len(validation), "study_pixels": len(pixel_di),
            "study_pixels_inside": int(inside.sum()), "study_inside_fraction": float(inside.mean()),
            "status_codes": STATUS, "predictions_changed": False,
            "method": "Meyer & Pebesma 2021, doi:10.1111/2041-210X.13650; folds are training parcel groups",
            "limitations": ["Similarity to training pixels is not evidence that a prediction is correct",
                            "The reference error is the European validation score, not a local accuracy",
                            "No pixel is masked, relabelled or removed"],
            "files": {name: {"sha256": references.checksum(staging / name)} for name in sorted(FILES)}}
        (staging / "applicability_handoff.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(folder)
    load_bundle(manifest_path)
    return manifest_path


def load_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != KIND or manifest.get("status") != "prepared"
            or manifest.get("predictions_changed") is not False or set(manifest.get("files", {})) != FILES
            or imagery._digest(manifest.get("identity", {}))[:20] != manifest.get("bundle_id")):
        raise ValueError("Unsupported or incomplete applicability handoff")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved applicability output changed: {name}; files preserved")
    if references.checksum(manifest["feature_handoff"]) != manifest["identity"]["feature_sha256"]:
        raise ValueError("Applicability belongs to different model features")
    with rasterio.open(path.parent / "applicability.tif") as raster:
        index, status = raster.read(1), raster.read(2)
    with np.load(path.parent / "reference_di.npz", allow_pickle=False) as arrays:
        reference = {name: arrays[name] for name in arrays.files}
    with (path.parent / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    assessed = status > 0
    if (int(assessed.sum()) != manifest["study_pixels"] or int((status == 1).sum()) != manifest["study_pixels_inside"]
            or not np.all(np.isfinite(index[assessed])) or np.any((index[assessed] <= manifest["threshold"]) != (status[assessed] == 1))):
        raise ValueError("Applicability raster differs from its saved summary")
    return {"manifest": manifest, "folder": path.parent, "index": index, "status": status,
            "training_di": reference["training"], "validation_di": reference["validation"], "summary": summary}


def by_prediction(bundle, prediction_bundle):
    """Cross-tabulate raw crop predictions by applicability; nothing is filtered."""
    if Path(prediction_bundle["manifest"]["feature_handoff"]).resolve() != Path(bundle["manifest"]["feature_handoff"]).resolve():
        raise ValueError("Predictions and applicability use different model features")
    feature = features.load_feature_bundle(bundle["manifest"]["feature_handoff"])
    inside = bundle["status"][feature["rows"], feature["columns"]] == 1
    codes, top = prediction_bundle["class_codes"], prediction_bundle["top_probability"]
    rows = []
    for name, code in prediction_bundle["manifest"]["class_codes"].items():
        chosen = codes == code
        rows.append({"raw_prediction": name.title(), "pixels": int(chosen.sum()),
                     "inside_applicability": int((chosen & inside).sum()),
                     "inside_fraction": float(inside[chosen].mean()) if chosen.any() else "",
                     "median_top_probability_inside": float(np.median(top[chosen & inside])) if (chosen & inside).any() else "",
                     "median_top_probability_outside": float(np.median(top[chosen & ~inside])) if (chosen & ~inside).any() else ""})
    return rows


def plot(bundle):
    """Left: DI map with pixels outside the area of applicability outlined in red. Right: DI distributions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    manifest = bundle["manifest"]
    threshold = manifest["threshold"]
    index = np.ma.masked_invalid(bundle["index"])
    limit = max(threshold * 2, float(np.percentile(index.compressed(), 99)))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    image = axes[0].imshow(index, vmin=0, vmax=limit, cmap="viridis", interpolation="nearest")
    outside = np.ma.masked_where(bundle["status"] != 2, np.ones_like(bundle["status"]))
    axes[0].imshow(outside, cmap=ListedColormap(["#e02020"]), alpha=.7, interpolation="nearest")
    axes[0].set_title(f"Dissimilarity to training pixels (red: outside, {1 - manifest['study_inside_fraction']:.0%})")
    axes[0].set_axis_off()
    figure.colorbar(image, ax=axes[0], shrink=.8, label="Dissimilarity index")
    bins = np.linspace(0, limit, 60)
    study = index.compressed()
    for values, label, color in ((bundle["training_di"], "Training pixels (other parcels only)", "#4b77a9"),
                                 (bundle["validation_di"], "Validation pixels (held-out parcels)", "#e0a030"),
                                 (study, "Study-area pixels", "#3a8a5c")):
        if len(values):
            axes[1].hist(np.clip(values, 0, limit), bins=bins, density=True, histtype="step", linewidth=1.8, label=label, color=color)
    axes[1].axvline(threshold, color="firebrick", linestyle="--", label=f"Applicability threshold ({threshold:.2f})")
    axes[1].set(xlabel="Dissimilarity index (values above the axis limit are stacked at the right edge)", ylabel="Density",
                title="How far pixels sit from the training data")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Model applicability, {manifest['period']['start']} to {manifest['period']['end']} (end exclusive): "
                    "a flag, not an accuracy estimate")
    return figure
