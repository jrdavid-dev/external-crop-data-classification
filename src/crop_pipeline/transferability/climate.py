"""Temperature profiles of the source regions and climate distances to the Atok target."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ATOK_ELEVATION_M, LAPSE_C_PER_M

TMAX = [f"tmax_{m:02d}" for m in range(1, 13)]
TMIN = [f"tmin_{m:02d}" for m in range(1, 13)]
TERRACLIMATE_SCALE_M = 4638
EARTH_RADIUS_KM = 6371


def sample_terraclimate(paths, data, atok_lon, atok_lat):
    """Monthly TerraClimate TMAX/TMIN at every parcel centroid and at Atok (2019-2024), cached as CSV."""
    climate_path = paths.output / "climate_points.csv"
    if climate_path.is_file():
        return pd.read_csv(climate_path), f"Loaded cached TerraClimate samples: {climate_path.name}"
    import ee
    ee.Initialize(project=paths.earth_engine_project)
    terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("2019-01-01", "2025-01-01").select(["tmmx", "tmmn"])
    monthly = ee.Image.cat([terraclimate.filter(ee.Filter.calendarRange(m, m, "month")).mean().multiply(0.1)
                            .rename([f"tmax_{m:02d}", f"tmin_{m:02d}"]) for m in range(1, 13)])
    parcel_points = data.groupby(["source_region", "partition_group"])[["lon", "lat"]].mean().reset_index()
    features = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"source_region": r.source_region, "group": r.partition_group})
                for r in parcel_points.itertuples()]
    features.append(ee.Feature(ee.Geometry.Point([atok_lon, atok_lat]), {"source_region": "Atok", "group": "atok"}))
    sampled = monthly.reduceRegions(ee.FeatureCollection(features), ee.Reducer.first(), scale=TERRACLIMATE_SCALE_M).getInfo()
    points = pd.DataFrame([f["properties"] for f in sampled["features"]])
    cell = ee.Geometry.Point([atok_lon, atok_lat]).buffer(TERRACLIMATE_SCALE_M / 2).bounds()  # one TerraClimate cell
    points["cell_elevation_m"] = np.where(
        points["source_region"] == "Atok",
        ee.Image("USGS/SRTMGL1_003").reduceRegion(ee.Reducer.mean(), cell, 90).get("elevation").getInfo(), np.nan)
    points.to_csv(climate_path, index=False)
    return points, "Sampled TerraClimate from Earth Engine"


@dataclass
class ClimateProfiles:
    profiles: pd.DataFrame  # source region x (TMAX + TMIN)
    target: pd.Series  # Atok target profile
    atok_raw: pd.Series  # TerraClimate at Atok, unadjusted
    note: str
    is_fallback: bool
    valid_points: int
    all_points: int


def atok_target(points, paths):
    """Source-region profiles and the Atok target: PAGASA Baguio lapse-adjusted to Atok, else TerraClimate."""
    valid = points.dropna(subset=TMAX + TMIN)
    profiles = valid[valid["source_region"] != "Atok"].groupby("source_region")[TMAX + TMIN].mean()
    atok_raw = valid.loc[valid["source_region"] == "Atok", TMAX + TMIN].iloc[0]
    cell_elevation = float(points.loc[points["source_region"] == "Atok", "cell_elevation_m"].iloc[0])
    if paths.pagasa is not None and paths.pagasa.is_file():
        raw = pd.read_csv(paths.pagasa).rename(columns=str.lower)  # PAGASA export: YEAR, MONTH, RAINFALL, TMAX, TMIN
        pagasa = raw.groupby("month")[["tmax", "tmin"]].mean().reindex(range(1, 13))
        if pagasa.isna().any().any():
            raise ValueError(f"{paths.pagasa.name} lacks TMAX/TMIN for some months")
        years = f"{raw['year'].min()}-{raw['year'].max()}" if "year" in raw else "climatology"
        shift = -LAPSE_C_PER_M * (ATOK_ELEVATION_M - paths.pagasa_elevation_m)
        target = pd.Series(np.r_[pagasa["tmax"].to_numpy() + shift, pagasa["tmin"].to_numpy() + shift], index=TMAX + TMIN)
        note = f"PAGASA Baguio {years} ({paths.pagasa_elevation_m:.0f} m), lapse-adjusted {shift:+.1f} °C to {ATOK_ELEVATION_M} m"
        fallback = False
    else:
        shift = -LAPSE_C_PER_M * (ATOK_ELEVATION_M - cell_elevation)
        target = atok_raw + shift
        note = (f"FALLBACK (no pagasa_baguio_path): TerraClimate at the Atok ROI, lapse-adjusted {shift:+.1f} °C "
                f"from the cell's mean elevation ({cell_elevation:.0f} m) to {ATOK_ELEVATION_M} m")
        fallback = True
    return ClimateProfiles(profiles, target, atok_raw, note, fallback, len(valid), len(points))


def season_aligned(profile, latitude):
    """Shift a southern-hemisphere profile by six months so seasons line up with Atok's."""
    if latitude >= 0:
        return profile
    return pd.Series(np.r_[np.roll(profile[TMAX].to_numpy(), 6), np.roll(profile[TMIN].to_numpy(), 6)], index=TMAX + TMIN)


def distances(climate):
    """Euclidean distance between each region's 24-value profile and the Atok target, ranked."""
    profiles, target = climate.profiles, climate.target
    table = pd.DataFrame({
        "climate_distance": np.sqrt(((profiles - target) ** 2).sum(axis=1)),
        "mean_tmax_diff": profiles[TMAX].mean(axis=1) - target[TMAX].mean(),
        "mean_tmin_diff": profiles[TMIN].mean(axis=1) - target[TMIN].mean(),
    }).sort_values("climate_distance")
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def add_aligned_distances(table, climate, region_lat):
    table = table.copy()
    table["aligned_distance"] = [float(np.sqrt(((season_aligned(climate.profiles.loc[r], region_lat[r]) - climate.target) ** 2).sum()))
                                 for r in table.index]
    table["aligned_rank"] = table["aligned_distance"].rank().astype(int)
    table["hemisphere"] = np.where(region_lat.reindex(table.index) < 0, "south", "north")
    return table


class RegionSimilarity:
    """Season-aligned climate distance and great-circle distance between any two regions, Atok included."""

    def __init__(self, climate, data, atok_lon, atok_lat, atok_name="Atok (provisional)"):
        region_lat = data.groupby("source_region")["lat"].mean()
        self.profiles = {r: season_aligned(climate.profiles.loc[r], region_lat[r]) for r in climate.profiles.index}
        self.profiles[atok_name] = climate.target
        self.centroids = data.groupby("source_region")[["lon", "lat"]].mean()
        self.centroids.loc[atok_name] = [atok_lon, atok_lat]

    def climate(self, a, b):
        return float(np.sqrt(((self.profiles[a] - self.profiles[b]) ** 2).sum()))

    def km(self, a, b):
        lon1, lat1, lon2, lat2 = np.radians([*self.centroids.loc[a], *self.centroids.loc[b]])
        return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2
                                                       + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
