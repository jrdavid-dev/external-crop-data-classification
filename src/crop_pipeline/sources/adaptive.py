"""Collect small parcel batches and top up from measured, quality-passing windows."""
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from shapely.geometry import shape

from crop_pipeline.common import save_json
from crop_pipeline.sources import annual, sampling
from crop_pipeline.windows.seasons import file_sha256

COLLECTION_NAME = "adaptive_training"
DEFAULT_POLICY = {
    "version": 1, "seed": 42, "candidate_parcels_per_crop_per_source": 25,
    "pixels_per_parcel": 20,
    "targets": {"train": {"pixels": 1200, "parcels": 60},
                "validation": {"pixels": 400, "parcels": 20},
                "test": {"pixels": 400, "parcels": 20}},
}


def policy_key(policy):
    from crop_pipeline.imagery.batch_export import content_key
    if set(policy["targets"]) != {"train", "validation", "test"}:
        raise ValueError("Provide train, validation and test targets")
    if any(type(v) is not int or v <= 0 for v in [policy["pixels_per_parcel"],
            policy["candidate_parcels_per_crop_per_source"],
            *[v for target in policy["targets"].values() for v in target.values()]]):
        raise ValueError("Sampling targets and batch sizes must be positive integers")
    return content_key(policy)[:12]


def root(report_folder, policy=None):
    policy = policy or DEFAULT_POLICY
    return Path(report_folder).parent / COLLECTION_NAME / policy_key(policy)


def read_json(path, default):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def partition_for(key, seed=42):
    bucket = int(sampling.stable_rank(key, seed)[:12], 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def physical_key(source, properties):
    return "|".join((source, str(properties.get("country", "")),
                     str(properties.get("parcel_id") or properties["partition_group"])))


def prepare_batch(source, features, archive_root, report_folder, provenance,
                  inward_buffer_metres=10, policy=None, batch_label=None, round_parcels_per_crop=None, candidate_crops=None):
    """Select references before rasterization. Re-running a round selects the same parcels."""
    policy = policy or DEFAULT_POLICY
    candidate_crops = set(annual.TARGET_CROPS if candidate_crops is None else candidate_crops)
    if not candidate_crops or candidate_crops - set(annual.TARGET_CROPS):
        raise ValueError("Provide known candidate crops")
    campaign = root(report_folder, policy)
    campaign.mkdir(parents=True, exist_ok=True)
    save_json(campaign / "policy.json", policy)
    feedback = read_json(campaign / "feedback.json", {})
    round_number = feedback.get("next_round", 1)
    if round_parcels_per_crop is not None and (type(round_parcels_per_crop) is not int or round_parcels_per_crop <= 0):
        raise ValueError("round_parcels_per_crop must be a positive integer")
    if batch_label is not None and (not batch_label or not all(c.isalnum() or c in "_-" for c in batch_label)):
        raise ValueError("Use letters, digits, underscores or hyphens in a batch label")
    folder = campaign / "rounds" / f"{round_number:04d}" / (f"{source}_{batch_label}" if batch_label else source)
    selection_path = folder / "selection.json"
    selected = read_json(selection_path, None)
    groups = defaultdict(list)
    for feature in features:
        p = feature["properties"]
        if p.get("crop") in candidate_crops and p.get("source_split", "train") == "train":
            groups[p["partition_group"]].append(feature)
    if selected is None:
        previous = [read_json(path, {}) for path in campaign.glob("rounds/*/*/selection.json")
                    if read_json(path, {}).get("source") == source]
        seen = {key for selection in previous for key in selection.get("physical_keys", [])}
        round_counts = Counter(groups[group][0]["properties"]["crop"]
            for selection in previous if selection.get("round") == round_number
            for group in selection.get("groups", []) if group in groups)
        candidates = defaultdict(lambda: defaultdict(list))
        unique_physical = set()
        for group in sorted(groups, key=lambda key: sampling.stable_rank(key, policy["seed"])):
            p = groups[group][0]["properties"]
            if len({f["properties"]["crop"] for f in groups[group]}) != 1:
                continue
            key = physical_key(source, p)
            if key in seen or key in unique_physical:
                continue
            unique_physical.add(key)
            partition = partition_for(key, policy["seed"])
            deficit = feedback.get("counts", {}).get(p["crop"], {}).get(partition)
            if feedback.get("targets_met") or (deficit and deficit["target_met"]):
                continue
            centre = shape(groups[group][0]["geometry"]).representative_point()
            stratum = (str(p.get("country", "")), str(p.get("year", "")),
                       math.floor(centre.x * 2), math.floor(centre.y * 2))
            candidates[p["crop"]][stratum].append((group, key, partition))
        chosen = []
        for crop in annual.TARGET_CROPS:
            limit = (max(0, round_parcels_per_crop - round_counts[crop]) if round_parcels_per_crop is not None
                     else policy["candidate_parcels_per_crop_per_source"])
            queues = [deque(candidates[crop][key]) for key in sorted(candidates[crop],
                       key=lambda key: sampling.stable_rank(str(key), policy["seed"]))]
            while queues and sum(item[3] == crop for item in chosen) < limit:
                queue = queues.pop(0)
                group, key, partition = queue.popleft()
                chosen.append((group, key, partition, crop))
                if queue:
                    queues.append(queue)
        selected = {"round": round_number, "source": source, "policy": policy,
                    "groups": [item[0] for item in chosen],
                    "physical_keys": [item[1] for item in chosen],
                    "assignments": {item[0]: item[2] for item in chosen},
                    "status": "selected" if chosen else "targets_met" if feedback.get("targets_met") else "no_candidates_for_deficits"}
        save_json(selection_path, selected)
    if set(selected["groups"]) - groups.keys():
        raise ValueError("Selected reference parcels changed; use a new campaign")
    selected_features = [f for group in selected["groups"] for f in groups[group]]
    context = {"campaign": str(campaign), "folder": str(folder), "round": round_number,
               "source": source, "policy": policy,
               "report_folder": str(folder / "inventory"),
               "annual_tif_root": str(Path(archive_root).parent / "annual_tifs"),
               "candidate_parcels": len(selected["groups"]), "selection_status": selected["status"]}
    save_json(folder / "context.json", context)
    if not selected_features:
        print(f"{source}: no new candidates needed/available; run notebook 04 for the current deficits.")
        return None, context
    archive_batch_root = Path(archive_root) / COLLECTION_NAME / campaign.name / f"round_{round_number:04d}"
    if batch_label:
        archive_batch_root /= batch_label
    plan = annual.prepare_source(source, features, archive_batch_root, provenance,
                                 inward_buffer_metres=inward_buffer_metres,
                                 candidate_groups=selected["groups"])
    context["plan_path"] = str(Path(plan["run_folder"]) / "collection_plan.json")
    save_json(folder / "context.json", context)
    print(f"Adaptive round {round_number}: {len(selected['groups'])} {source} candidate parcels; "
          "annual histories retained, usable targets checked after four-month selection.")
    return plan, context


def collect_batch(plan, context, project, enabled=False, authenticate=False, **kwargs):
    if plan is None:
        result = {"status": "skipped", "reason": context["selection_status"]}
        save_json(Path(context["folder"]) / "state.json", result)
        return result
    result = annual.collect_source(plan, Path(context["report_folder"]), project,
        enabled=enabled, authenticate=authenticate, archive_format="pixel_tables",
        collection_backend="batch", **kwargs)
    folder = Path(context["folder"])
    save_json(folder / "state.json", result)
    if result["status"] == "completed":
        bundle = read_json(result["registry"], {})
        from crop_pipeline.imagery import annual_tif
        for handoff in bundle["handoffs"]:
            manifest_path = Path(handoff["path"])
            manifest = read_json(manifest_path, {})
            output_root = Path(context["annual_tif_root"]) / context["source"] / str(manifest["year"]) / Path(context["campaign"]).name / f"round_{context['round']:04d}"
            index = annual_tif.export_source(manifest, output_root)
            manifest.update(adaptive_collection=COLLECTION_NAME, adaptive_round=context["round"],
                            annual_tif_index=index, annual_tif_root=str(output_root.resolve()))
            save_json(manifest_path, manifest)
            handoff["sha256"] = file_sha256(manifest_path)
        save_json(Path(result["registry"]), bundle)
        save_json(folder / "completed.json", {"source": context["source"], "round": context["round"],
                                               "handoffs": bundle["handoffs"]})
    return result


def resolve_collection(report_folder, policy=None):
    campaign = root(report_folder, policy)
    pending = [path for path in campaign.glob("rounds/*/*/state.json")
               if read_json(path, {}).get("status") not in {"completed", "disabled", "skipped"}]
    if pending:
        raise RuntimeError("Finish the current candidate batches before window assessment: " + ", ".join(map(str, pending)))
    records = [record for path in sorted(campaign.glob("rounds/*/*/completed.json"))
               for record in read_json(path, {})["handoffs"]]
    if not records:
        raise FileNotFoundError("Run source notebooks to complete a small adaptive candidate batch first")
    for record in records:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Adaptive handoff changed after registration")
    from crop_pipeline.imagery.batch_export import content_key
    return [Path(record["path"]) for record in records], f"{COLLECTION_NAME}_{policy_key(policy or DEFAULT_POLICY)}_{content_key(records)[:12]}"


def assignments(report_folder, policy=None):
    result = {}
    for path in root(report_folder, policy).glob("rounds/*/*/selection.json"):
        for group, partition in read_json(path, {})["assignments"].items():
            if group in result and result[group] != partition:
                raise ValueError("A reserved parcel partition changed")
            result[group] = partition
    return result


def select_usable(reference_rows, estimated_rows, report_folder, policy=None):
    """Count complete windows only, cap pixels per parcel, preserve locked evaluation groups."""
    policy = policy or DEFAULT_POLICY
    campaign = root(report_folder, policy)
    locked = assignments(report_folder, policy)
    from crop_pipeline.windows import seasons
    by_group = defaultdict(list)
    seen = set()
    # Reference-supported observations take precedence at an identical location.
    for origin, rows in [("reference", reference_rows), ("estimated", estimated_rows)]:
        for row in sorted(rows, key=lambda row: sampling.stable_rank(str(sampling.pixel_location(row)), policy["seed"])):
            if origin == "estimated" and not seasons.timing_is_qualified(row):
                continue
            group = row["partition_group"]
            if group not in locked:
                raise ValueError("Usable parcel is absent from the locked candidate assignments")
            location = sampling.pixel_location(row)
            if location in seen:
                continue
            seen.add(location)
            by_group[group].append((origin, row))
    counts = {crop: {partition: {"pixels": 0, "parcels": 0, "target_met": False}
                    for partition in policy["targets"]} for crop in annual.TARGET_CROPS}
    selected = {"reference": [], "estimated": []}
    for group in sorted(by_group, key=lambda group: sampling.stable_rank(group, policy["seed"])):
        partition = locked[group]
        rows = by_group[group]
        crop = rows[0][1]["crop"]
        if len({row["crop"] for _, row in rows}) != 1:
            raise ValueError("A parcel has conflicting crop labels")
        status, target = counts[crop][partition], policy["targets"][partition]
        remaining = target["pixels"] - status["pixels"]
        # Keep at least one pixel from additional groups if parcel diversity is still deficient.
        take = min(policy["pixels_per_parcel"], len(rows), max(remaining, 1 if status["parcels"] < target["parcels"] else 0))
        if take:
            for origin, row in rows[:take]:
                selected[origin].append(row)
            status["pixels"] += take
            status["parcels"] += 1
    for crop, partitions in counts.items():
        for partition, status in partitions.items():
            target = policy["targets"][partition]
            status.update(pixel_deficit=max(0, target["pixels"] - status["pixels"]),
                          parcel_deficit=max(0, target["parcels"] - status["parcels"]))
            status["target_met"] = not status["pixel_deficit"] and not status["parcel_deficit"]
    completed = [read_json(path, {})["round"] for path in campaign.glob("rounds/*/*/completed.json")]
    feedback = {"policy": policy, "counts": counts,
                "targets_met": all(status["target_met"] for partitions in counts.values() for status in partitions.values()),
                "next_round": max(completed, default=0) + 1,
                "rule": "Complete quality-passing four-month pixels only; no model/test scores used for top-ups"}
    save_json(campaign / "feedback.json", feedback)
    save_json(campaign / "locked_split.json", {"kind": "exploratory_existing_source_parcel_evaluation", "group_assignments": locked})
    print("Usable sample counts after four-month quality checks:")
    for crop, partitions in counts.items():
        for partition, status in partitions.items():
            target = policy["targets"][partition]
            print(f"  {crop} {partition}: {status['pixels']}/{target['pixels']} pixels, "
                  f"{status['parcels']}/{target['parcels']} parcels")
    print("Targets met:", feedback["targets_met"], "Next candidate round:", feedback["next_round"])
    return selected["reference"], selected["estimated"], [
        {"crop": crop, "partition": partition, **status}
        for crop, partitions in counts.items() for partition, status in partitions.items()]


def require_targets(report_folder, policy=None):
    feedback = read_json(root(report_folder, policy) / "feedback.json", {})
    if not feedback.get("targets_met"):
        raise RuntimeError("Usable training/validation/test targets are not met. Run source notebooks for deficit crops, then rerun notebook 04.")
    return root(report_folder, policy) / "locked_split.json"


def prepare_round_batches(source, features, archive_root, report_folder, provenance,
                          parcels_per_crop=100, inward_buffer_metres=10, candidate_crops=None):
    """Add a separate extension while preserving existing jobs and the campaign split."""
    campaign = root(report_folder)
    round_number = read_json(campaign / "feedback.json", {}).get("next_round", 1)
    label = f"overnight_{parcels_per_crop}"
    extension_folder = campaign / "rounds" / f"{round_number:04d}" / f"{source}_{label}"
    batches = []
    for path in sorted((campaign / "rounds" / f"{round_number:04d}").glob("*/context.json")):
        context = read_json(path, {})
        if context.get("source") != source or path.parent == extension_folder:
            continue
        state = read_json(path.parent / "state.json", {})
        if state.get("status") in {"completed", "skipped"}:
            continue
        plan_path = context.get("plan_path")
        if not plan_path and state.get("retry_queue"):
            plan_path = str(Path(state["retry_queue"]).parents[2] / "collection_plan.json")
        if not plan_path:
            # A prepared batch may have been interrupted before its first collection pass.
            configuration = read_json(path.parent / "selection.json", {})
            wanted = set(configuration.get("groups", []))
            for candidate in (Path(archive_root) / COLLECTION_NAME / campaign.name / f"round_{round_number:04d}").glob(
                    f"**/{source}/runs/{annual.COLLECTION_NAME}/*/collection_plan.json"):
                saved = read_json(candidate.parent / "run_configuration.json", {})
                if set(saved.get("candidate_groups", [])) == wanted:
                    plan_path = str(candidate)
                    break
        if not plan_path or not Path(plan_path).is_file():
            raise ValueError(f"Cannot resume the existing batch in {path.parent}; its saved plan is missing")
        batches.append((read_json(plan_path, {}), context))
    plan, context = prepare_batch(source, features, archive_root, report_folder, provenance,
        inward_buffer_metres=inward_buffer_metres, batch_label=label,
        round_parcels_per_crop=parcels_per_crop, candidate_crops=candidate_crops)
    batches.append((plan, context))
    print(f"Round {round_number}: target up to {parcels_per_crop} candidates per available crop. "
          f"Resume/submit {len(batches)} batches; previous jobs and locked partitions are preserved.")
    return batches


def collect_batches(batches, project, enabled=False, authenticate=False, **kwargs):
    """Collect/submit every active batch once; never wait overnight inside this call."""
    results = []
    for index, (plan, context) in enumerate(batches):
        result = collect_batch(plan, context, project, enabled=enabled,
                               authenticate=authenticate and index == 0, **kwargs)
        results.append({"batch": Path(context["folder"]).name, "result": result})
    states = [entry["result"]["status"] for entry in results]
    status = ("disabled" if not enabled else "completed"
              if all(state in {"completed", "skipped"} for state in states) else "pending_retry")
    return {"status": status, "batches": results,
            "progress_by_year": {f"{entry['batch']}/{year}": progress
                for entry in results for year, progress in entry["result"].get("progress_by_year", {}).items()}}
