"""Apply the verified fixed crop model; never refit, filter by cover or infer accuracy."""

import csv
import json
from pathlib import Path
import tempfile

from affine import Affine
import joblib
import numpy as np
import rasterio
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from crop_pipeline.paoay import features, imagery, references

VERSION = 1
NODATA = -9999


def _class_schema(recipe):
    classes, codes = recipe.get("study_classes", []), recipe.get("class_codes", {})
    if (len(classes) != 3 or set(classes) != {"carrot", "cabbage", "potato"} or set(codes) != set(classes)
            or any(type(code) is not int for code in codes.values()) or set(codes.values()) != {1, 2, 3}
            or recipe.get("nodata_code") != 0):
        raise ValueError("Unsupported three-crop class schema; zero must remain nodata")
    return list(classes), codes


def load_classifier(model_manifest):
    """Deserialize only the local notebook 06 artifact after its hashes are verified."""
    verified = features.load_model_recipe(model_manifest)
    recipe = verified["recipe"]
    classes, _ = _class_schema(recipe)
    artifact = joblib.load(Path(model_manifest).parent / "model.joblib")
    if (not isinstance(artifact, dict) or set(artifact) != {"classifier", "feature_columns", "configuration"}
            or artifact["feature_columns"] != recipe["feature_columns"]):
        raise ValueError("Fitted artifact feature order differs from its verified recipe")
    classifier = artifact["classifier"]
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Expected notebook 06's fitted RandomForestClassifier")
    check_is_fitted(classifier)
    if (classifier.n_features_in_ != len(recipe["feature_columns"]) or classifier.n_outputs_ != 1
            or len(classifier.classes_) != 3 or set(classifier.classes_.tolist()) != set(classes)):
        raise ValueError("Fitted classifier dimensions/classes differ from the recipe")
    if hasattr(classifier, "feature_names_in_") and classifier.feature_names_in_.tolist() != recipe["feature_columns"]:
        raise ValueError("Fitted classifier feature names differ from the recipe")
    configuration = artifact["configuration"]
    if (configuration.get("feature_policy") != recipe["feature_policy"]
            or configuration.get("study_classes") != classes):
        raise ValueError("Fitted model configuration differs from the recipe")
    return classifier, recipe


def prepare_request(feature_handoff, output_root):
    bundle = features.load_feature_bundle(feature_handoff)  # No model loading for pending/empty inputs.
    model_manifest = bundle["manifest"]["model_manifest"]
    recipe = features.load_model_recipe(model_manifest)["recipe"]
    _class_schema(recipe)
    identity = {"version": VERSION, "feature_sha256": references.checksum(feature_handoff),
                "model_manifest_sha256": references.checksum(model_manifest),
                "implementation_sha256": references.checksum(__file__),
                "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    key = imagery._digest(identity)[:20]
    return {"bundle_id": key, "identity": identity, "feature_handoff": str(Path(feature_handoff).resolve()),
            "model_manifest": str(Path(model_manifest).resolve()), "folder": Path(output_root) / "bundles" / key}


def _verify(request):
    if imagery._digest(request["identity"])[:20] != request["bundle_id"]:
        raise ValueError("Prediction request changed after planning")
    for key, field in (("feature_handoff", "feature_sha256"), ("model_manifest", "model_manifest_sha256")):
        if references.checksum(request[key]) != request["identity"][field]:
            raise ValueError(f"Prediction dependency changed: {key}")
    bundle = features.load_feature_bundle(request["feature_handoff"])
    if Path(bundle["manifest"]["model_manifest"]).resolve() != Path(request["model_manifest"]).resolve():
        raise ValueError("Prediction model differs from the feature-bound model")
    return bundle


def predict_matrix(classifier, matrix, recipe, chunk_size=4096, iterator=None):
    """Map probabilities by classifier.classes_, preserving sklearn's first-class tie rule."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    classes, codes = _class_schema(recipe)
    learned = classifier.classes_.tolist()
    if len(learned) != 3 or set(learned) != set(classes):
        raise ValueError("Classifier probability labels differ from the recipe")
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != len(recipe["feature_columns"]) or not np.all(np.isfinite(matrix)):
        raise ValueError("Predict only finite features in the exact saved schema")
    probabilities = np.empty((len(matrix), 3), dtype="float64")
    labels = np.empty(len(matrix), dtype="uint8")
    starts = range(0, len(matrix), chunk_size)
    for start in (iterator(starts) if iterator else starts):
        end = min(start + chunk_size, len(matrix))
        raw = np.asarray(classifier.predict_proba(matrix[start:end]), dtype="float64")
        if (raw.shape != (end-start, 3) or not np.all(np.isfinite(raw) & (raw >= 0) & (raw <= 1))
                or not np.allclose(raw.sum(axis=1), 1, atol=1e-8, rtol=0)):
            raise ValueError("Classifier returned invalid three-crop probabilities")
        winners = np.argmax(raw, axis=1)
        labels[start:end] = [codes[learned[index]] for index in winners]
        probabilities[start:end] = raw[:, [learned.index(name) for name in classes]]
    ordered = np.sort(probabilities, axis=1)
    return {"class_codes": labels, "probabilities": probabilities, "top_probability": ordered[:, -1],
            "probability_margin": ordered[:, -1]-ordered[:, -2], "tied": ordered[:, -1] == ordered[:, -2]}


def build(request, chunk_size=4096, iterator=None):
    source = _verify(request)
    folder = request["folder"]
    handoff = folder / "prediction_handoff.json"
    if folder.exists():
        load_prediction_bundle(handoff)
        return handoff
    classifier, recipe = load_classifier(request["model_manifest"])
    result = predict_matrix(classifier, source["matrix"], recipe, chunk_size, iterator)
    grid, period = source["manifest"]["grid"], source["manifest"]["period"]
    classes, codes = _class_schema(recipe)
    probability_bands = ["probability_" + name for name in classes] + ["top_probability", "probability_margin"]
    rows, columns = source["rows"], source["columns"]
    label_raster = np.zeros((grid["height"], grid["width"]), dtype="uint8")
    label_raster[rows, columns] = result["class_codes"]
    probability_raster = np.full((5, grid["height"], grid["width"]), NODATA, dtype="float32")
    probability_raster[:3, rows, columns] = result["probabilities"].T
    probability_raster[3, rows, columns] = result["top_probability"]
    probability_raster[4, rows, columns] = result["probability_margin"]
    transform = Affine(*grid["transform"])
    crs = rasterio.crs.CRS.from_user_input(grid["crs"])
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Prediction area summaries require projected metre coordinates")
    hectare = abs(transform.a*transform.e-transform.b*transform.d)/10000
    summary = [{"crop": name, "code": codes[name], "pixels": int((result["class_codes"] == codes[name]).sum()),
                "area_ha": int((result["class_codes"] == codes[name]).sum())*hectare,
                "percent_of_predicted_pixels": float((result["class_codes"] == codes[name]).mean()*100)} for name in classes]
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predictions-", dir=folder.parent) as temporary:
        staging = Path(temporary) / request["bundle_id"]
        staging.mkdir()
        np.savez_compressed(staging / "predictions.npz", **result, sample_ids=source["sample_ids"])
        profile = dict(driver="GTiff", width=grid["width"], height=grid["height"], crs=grid["crs"], transform=transform, compress="deflate")
        with rasterio.open(staging / "crop_class.tif", "w", **profile, count=1, dtype="uint8", nodata=0) as raster:
            raster.write(label_raster, 1); raster.set_band_description(1, "crop_class")
        with rasterio.open(staging / "crop_probabilities.tif", "w", **profile, count=5, dtype="float32", nodata=NODATA) as raster:
            raster.write(probability_raster); raster.descriptions = tuple(probability_bands)
        with (staging / "crop_areas.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        _verify(request)  # Detect any dependency changes before publication; never save the classifier.
        manifest = {"format_version": VERSION, "kind": "paoay_crop_predictions", "status": "prepared",
            "bundle_id": request["bundle_id"], "identity": request["identity"],
            "feature_handoff": request["feature_handoff"], "model_manifest": request["model_manifest"],
            "period": period, "grid": grid, "row_count": len(source["matrix"]), "roi_pixels": source["manifest"]["roi_pixels"],
            "study_classes": classes, "class_codes": codes, "classifier_class_order": classifier.classes_.tolist(),
            "probability_bands": probability_bands, "nodata_code": 0, "probability_nodata": NODATA,
            "tied_pixels": int(result["tied"].sum()), "tie_policy": "first_class_in_classifier_classes_matches_sklearn_predict",
            "land_cover_filter_applied": False, "confidence_threshold_applied": False,
            "model_refitted": False, "pixel_area_ha": hectare,
            "limitations": ["Crop-only forced choice among the three trained classes; no other-crop detection",
                "Probabilities and margins are model scores, not calibrated local accuracy",
                "Predictions are retained even where broad-cover guidance is other cover or unresolved",
                "Fixed calendar demonstration; local growth-stage alignment remains unverified",
                "No independent Paoay accuracy evaluation performed"],
            "files": {name: {"sha256": references.checksum(staging / name)} for name in
                      ("predictions.npz", "crop_class.tif", "crop_probabilities.tif", "crop_areas.csv")}}
        (staging / "prediction_handoff.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        staging.rename(folder)
    load_prediction_bundle(handoff)
    return handoff


def load_prediction_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("format_version") != VERSION or manifest.get("kind") != "paoay_crop_predictions"
            or manifest.get("status") != "prepared" or manifest.get("model_refitted") is not False
            or manifest.get("land_cover_filter_applied") is not False or manifest.get("confidence_threshold_applied") is not False
            or set(manifest.get("files", {})) != {"predictions.npz", "crop_class.tif", "crop_probabilities.tif", "crop_areas.csv"}):
        raise ValueError("Unsupported or incomplete crop prediction handoff")
    source = _verify({key: manifest[key] for key in ("bundle_id", "identity", "feature_handoff", "model_manifest")})
    recipe = features.load_model_recipe(manifest["model_manifest"])["recipe"]
    classes, codes = _class_schema(recipe)
    if (manifest.get("grid") != source["manifest"]["grid"] or manifest.get("period") != source["manifest"]["period"]
            or manifest.get("study_classes") != classes or manifest.get("class_codes") != codes
            or manifest.get("row_count") != source["manifest"]["row_count"]):
        raise ValueError("Prediction schema/dates/grid differ from verified inputs")
    for name, record in manifest["files"].items():
        if references.checksum(path.parent / name) != record["sha256"]:
            raise ValueError(f"Saved prediction changed: {name}; existing files preserved")
    with np.load(path.parent / "predictions.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    n = manifest["row_count"]
    if (set(values) != {"class_codes", "probabilities", "top_probability", "probability_margin", "tied", "sample_ids"}
            or values["probabilities"].shape != (n, 3)
            or any(values[name].shape != (n,) for name in values if name != "probabilities")
            or not np.array_equal(values["sample_ids"], source["sample_ids"])
            or not np.isin(values["class_codes"], list(codes.values())).all()
            or not np.all(np.isfinite(values["probabilities"]) & (values["probabilities"] >= 0) & (values["probabilities"] <= 1))
            or not np.allclose(values["probabilities"].sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Prediction values or row identities changed")
    learned = manifest.get("classifier_class_order", [])
    if len(learned) != 3 or set(learned) != set(classes):
        raise ValueError("Saved probability class order changed")
    learned_probabilities = values["probabilities"][:, [classes.index(name) for name in learned]]
    expected_labels = np.asarray([codes[learned[index]] for index in np.argmax(learned_probabilities, axis=1)])
    sorted_probabilities = np.sort(values["probabilities"], axis=1)
    top, margin = sorted_probabilities[:, -1], sorted_probabilities[:, -1]-sorted_probabilities[:, -2]
    tied = margin == 0
    if (not np.array_equal(expected_labels, values["class_codes"])
            or not np.array_equal(top, values["top_probability"])
            or not np.array_equal(margin, values["probability_margin"])
            or not np.array_equal(tied, values["tied"]) or manifest.get("tied_pixels") != int(tied.sum())):
        raise ValueError("Saved classes or uncertainty scores disagree with the probabilities")
    grid = manifest["grid"]
    rows, columns = source["rows"], source["columns"]
    expected_bands = ["probability_"+name for name in classes]+["top_probability", "probability_margin"]
    if manifest.get("probability_bands") != expected_bands:
        raise ValueError("Saved probability band order changed")
    ready = np.zeros((grid["height"], grid["width"]), dtype=bool)
    ready[rows, columns] = True
    for name, count, dtype, nodata, bands in (("crop_class.tif", 1, "uint8", 0, ["crop_class"]),
            ("crop_probabilities.tif", 5, "float32", NODATA, expected_bands)):
        with rasterio.open(path.parent / name) as raster:
            if ((raster.height, raster.width) != ready.shape or raster.count != count
                    or raster.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                    or not raster.transform.almost_equals(Affine(*grid["transform"]))
                    or list(raster.descriptions) != bands or raster.nodata != nodata
                    or any(value != dtype for value in raster.dtypes)):
                raise ValueError("Saved prediction raster grid/schema changed")
            data = raster.read()
        expected = values["class_codes"][None, :] if count == 1 else np.vstack([values["probabilities"].T, top, margin]).astype("float32")
        if not np.array_equal(data[:, rows, columns], expected) or not np.all(data[:, ~ready] == nodata):
            raise ValueError("Saved raster predictions do not match the feature pixel locations")
    with (path.parent / "crop_areas.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != 3 or {row["crop"] for row in summary} != set(classes):
        raise ValueError("Saved class-area table changed")
    for row in summary:
        expected_count = int((values["class_codes"] == codes[row["crop"]]).sum())
        if (int(row["code"]) != codes[row["crop"]] or int(row["pixels"]) != expected_count
                or not np.isclose(float(row["area_ha"]), expected_count*manifest["pixel_area_ha"])
                or not np.isclose(float(row["percent_of_predicted_pixels"]), expected_count/n*100)):
            raise ValueError("Saved class areas differ from the predictions")
    return {"manifest": manifest, "folder": path.parent, **values, "summary": summary}


def plot_predictions(bundle):
    """Crop-only preview; final map assembly and land-cover guidance remain separate."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    manifest = bundle["manifest"]
    colors = {"carrot": "#ed9137", "cabbage": "#4b9a65", "potato": "#8b70b1"}
    palette = [colors[next(name for name, code in manifest["class_codes"].items() if code == value)] for value in (1, 2, 3)]
    with rasterio.open(bundle["folder"] / "crop_class.tif") as raster:
        labels = raster.read(1, masked=True)
    with rasterio.open(bundle["folder"] / "crop_probabilities.tif") as raster:
        scores = raster.read(4, masked=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].imshow(labels, cmap=ListedColormap(palette).with_extremes(bad="white"),
                   norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], 3), interpolation="nearest")
    axes[0].set_title("Crop-only model prediction")
    axes[0].legend(handles=[Patch(color=colors[name], label=name.title()) for name in manifest["study_classes"]], loc="lower left")
    image = axes[1].imshow(scores, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Top model probability (not local accuracy)")
    figure.colorbar(image, ax=axes[1], shrink=.8)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(f"{manifest['period']['start']} to {manifest['period']['end']} (end exclusive) | fixed-calendar prototype")
    return figure
