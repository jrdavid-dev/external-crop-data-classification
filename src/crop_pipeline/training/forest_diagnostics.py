"""Inspect frozen forests without fitting or changing their predictions."""

from collections import defaultdict

import numpy as np

from crop_pipeline.training import exploratory
from crop_pipeline.training.model_comparisons import WEIGHTS


def parcel_index(rows):
    ids, index = np.unique([row["partition_group"] for row in rows], return_inverse=True)
    return ids.tolist(), index, np.bincount(index)


def tree_probabilities(tree, values, n_classes):
    # The forest fits its trees with encoded class indices.
    probabilities = tree.predict_proba(values, check_input=False)
    if not np.array_equal(tree.classes_, np.arange(n_classes)):
        raise ValueError("Tree class order differs from the forest")
    return probabilities


def parcel_probabilities(probabilities, index, counts):
    return np.column_stack([
        np.bincount(index, weights=column, minlength=len(counts)) / counts
        for column in probabilities.T
    ])


def oob_statistics(forest, train, columns):
    values = np.ascontiguousarray(exploratory.feature_matrix(train, columns), dtype=np.float32)
    ids, index, counts = parcel_index(train)
    labels = np.array([row["crop"] for row in train])
    parcel_classes = {}
    for row in train:
        if parcel_classes.setdefault(row["partition_group"], row["crop"]) != row["crop"]:
            raise ValueError("Conflicting training parcel labels")
    reference = np.array([parcel_classes[group] for group in ids])
    weights = exploratory.normalized_training_weights(train, WEIGHTS)
    if (not forest.bootstrap or forest._n_samples != len(train)
            or not np.array_equal(forest._sample_weight, weights)):
        raise ValueError("Saved weighted bootstrap differs from the ordered development snapshot")

    n_trees = len(forest.estimators_)
    classes = forest.classes_
    pixel_sums = np.zeros((len(train), len(classes)))
    pixel_trees = np.zeros(len(train), dtype=int)
    omitted_parcel_sums = np.zeros((len(ids), len(classes)))
    omitted_parcel_trees = np.zeros(len(ids), dtype=int)
    tree_rows = []
    for number, (tree, in_bag) in enumerate(zip(forest.estimators_, forest.estimators_samples_, strict=True), 1):
        sampled = np.bincount(in_bag, minlength=len(train)) > 0
        if len(sampled) != len(train) or int(sampled.sum()) != tree.tree_.n_node_samples[0]:
            raise ValueError("Reconstructed distinct training rows differ from the fitted tree root")
        probabilities = tree_probabilities(tree, values, len(classes))
        omitted = ~sampled
        pixel_sums[omitted] += probabilities[omitted]
        pixel_trees[omitted] += 1
        present = np.bincount(index[sampled], minlength=len(ids)) > 0
        parcel_means = parcel_probabilities(probabilities, index, counts)
        omitted_parcel_sums[~present] += parcel_means[~present]
        omitted_parcel_trees[~present] += 1
        tree_rows.append({
            "tree": number, "drawn_rows_with_replacement": len(in_bag),
            "distinct_training_rows": int(sampled.sum()), "oob_training_rows": int(omitted.sum()),
            "oob_whole_parcels": int((~present).sum()),
        })

    covered = pixel_trees > 0
    pixel_means = np.divide(pixel_sums, pixel_trees[:, None], out=np.zeros_like(pixel_sums),
                            where=covered[:, None])
    # The pixel-OOB parcel summary weights each covered pixel equally.
    covered_count = np.bincount(index[covered], minlength=len(ids))
    pixel_parcel_sum = np.column_stack([
        np.bincount(index, weights=column, minlength=len(ids)) for column in pixel_means.T
    ])
    views = (
        ("pixel_oob", labels, pixel_sums, pixel_trees, len(train)),
        ("parcel_from_pixel_oob", reference, pixel_parcel_sum, covered_count, len(ids)),
        ("whole_parcel_oob", reference, omitted_parcel_sums, omitted_parcel_trees, len(ids)),
    )
    summaries = []
    for method, actual, votes, support, size in views:
        eligible = support > 0
        n = int(eligible.sum())
        correct = int((classes[votes[eligible].argmax(axis=1)] == actual[eligible]).sum())
        summaries.append({
            "method": method, "eligible_units": size, "scored_units": n, "correct": correct,
            "error_rate": (n - correct) / n if n else None, "trees": n_trees,
            "minimum_contributors": int(support[eligible].min()) if n else None,
            "mean_contributors": float(support[eligible].mean()) if n else None,
            "contributor_unit": "covered_pixels" if method == "parcel_from_pixel_oob" else "trees",
        })
    parcel_rows = []
    for j, group in enumerate(ids):
        parcel_rows.append({
            "partition_group": group, "crop": reference[j], "pixels": int(counts[j]),
            "pixels_with_oob_votes": int(covered_count[j]), "whole_parcel_oob_trees": int(omitted_parcel_trees[j]),
            "pixel_oob_parcel_prediction": classes[pixel_parcel_sum[j].argmax()] if covered_count[j] else None,
            "whole_parcel_oob_prediction": classes[omitted_parcel_sums[j].argmax()]
            if omitted_parcel_trees[j] else None,
        })
    return summaries, tree_rows, parcel_rows


def hard_vote_statistics(forest, rows, columns):
    """Count one classification vote per tree for each pixel and each parcel."""
    values = np.ascontiguousarray(exploratory.feature_matrix(rows, columns), dtype=np.float32)
    ids, index, counts = parcel_index(rows)
    classes = forest.classes_
    pixel_votes = np.zeros((len(rows), len(classes)), dtype=int)
    parcel_votes = np.zeros((len(ids), len(classes)), dtype=int)
    parcel_sums = np.zeros_like(parcel_votes, dtype=float)
    for tree in forest.estimators_:
        probabilities = tree_probabilities(tree, values, len(classes))
        pixel_votes[np.arange(len(rows)), probabilities.argmax(axis=1)] += 1
        parcel_means = parcel_probabilities(probabilities, index, counts)
        parcel_votes[np.arange(len(ids)), parcel_means.argmax(axis=1)] += 1
        parcel_sums += parcel_means
    n_trees = len(forest.estimators_)

    def vote_row(votes):
        fractions = votes / n_trees
        return {"hard_vote_winner": classes[fractions.argmax()],
                "max_hard_vote_fraction": float(fractions.max()),
                **{f"{crop}_hard_vote_fraction": float(value) for crop, value in zip(classes, fractions)}}

    parcels = {}
    for i, group in enumerate(ids):
        probability_winner = parcel_sums[i].argmax()
        parcels[group] = {
            **vote_row(parcel_votes[i]), "probability_prediction": classes[probability_winner],
            "mean_probability_for_prediction": float(parcel_sums[i, probability_winner] / n_trees),
            "hard_vote_for_probability_prediction": float(parcel_votes[i, probability_winner] / n_trees),
        }
    pixel_rows = [{**{key: row[key] for key in ("partition_group", "region_id", "crop", "x", "y")},
                   **vote_row(votes)} for row, votes in zip(rows, pixel_votes, strict=True)]
    return parcels, pixel_rows


def impurity_importance(forest, columns):
    details, groups = [], defaultdict(float)
    for column, value in zip(columns, forest.feature_importances_, strict=True):
        prefix, bin_number, variable = column.split("_")
        if prefix != "bin" or not 1 <= int(bin_number) <= 8:
            raise ValueError(f"Unexpected interval predictor {column}")
        details.append({"predictor": column, "bin": int(bin_number), "variable": variable,
                        "importance": float(value)})
        groups["band_or_index", variable] += float(value)
        groups["half_month_bin", bin_number] += float(value)
    return details, [{"grouping": grouping, "group": group, "importance": value}
                     for (grouping, group), value in groups.items()]
