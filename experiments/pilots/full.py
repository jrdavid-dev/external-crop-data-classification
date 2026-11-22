"""Feature sets that helped in the pilots, defined once for full-data evaluation and Paoay.

Each builder takes Sentinel-2 reflectance [n, 24, 5] with its clear flags and,
where needed, Sentinel-1 dB [n, 24, 2]; it returns (X, ok, columns).
"""

import numpy as np

from experiments.pilots import features


def s2_harmonic(s2, s2_valid, s1=None):
    return features.harmonic_features(features.stack(s2), s2_valid, form="full")


def s1_raw(s2, s2_valid, s1):
    sar = features.sar_variables(s1)
    return features.harmonic_features(sar[:, :, :2], np.isfinite(sar).all(axis=2), names=("VV", "VH"), form="full")


def s1_ratios(s2, s2_valid, s1):
    sar = features.sar_variables(s1)
    return features.harmonic_features(sar[:, :, 2:], np.isfinite(sar).all(axis=2), names=("VHVV", "CROSSPOL"),
                                      form="amplitude")


def combine(*builders):
    def build(s2, s2_valid, s1=None):
        parts = [b(s2, s2_valid, s1) for b in builders]
        return (np.hstack([p[0] for p in parts]), np.logical_and.reduce([p[1] for p in parts]),
                [c for p in parts for c in p[2]])
    return build


SETS = {
    "s2_harmonic": s2_harmonic,
    "s1_raw_dB": s1_raw,
    "s1_ratios": s1_ratios,
    "s2_harmonic_plus_s1_ratios": combine(s2_harmonic, s1_ratios),
    "s2_harmonic_plus_s1_raw_and_ratios": combine(s2_harmonic, s1_raw, s1_ratios),
}


def copies(build, s2, s2_valid, s1, shifts=(-1, 1), gain=.1, seed=0):
    """Augmented matrices: both sensors shifted together; Sentinel-2 gain only (radar is in dB)."""
    rng = np.random.default_rng(seed)
    result = []
    for shift in shifts:
        shifted_s2, shifted_valid = features.shift_series(s2, s2_valid, shift)
        shifted_s1 = None if s1 is None else np.roll(s1, shift, axis=1)
        result.append(np.nan_to_num(build(shifted_s2, shifted_valid, shifted_s1)[0]))
    if gain:
        result.append(np.nan_to_num(build(features.gain_series(s2, rng, gain), s2_valid, s1)[0]))
    return result


def paoay_pixels(arrays):
    """Paoay study pixels as [n, 24, 5] reflectance, clear flags and [n, 24, 2] radar dB."""
    mask = arrays["paoay_mask"]
    s2 = (arrays["paoay_cube"].astype(float) * .0001)[:, :, mask].transpose(2, 0, 1)
    s1 = arrays["paoay_s1"].astype(float)[:, :, mask].transpose(2, 0, 1)
    return np.argwhere(mask), s2, np.isfinite(s2).all(axis=2), s1


def s2_harmonic_omega(omega):
    def build(s2, s2_valid, s1=None):
        X, ok, columns = features.harmonic_features(features.stack(s2), s2_valid, form="full", omega=omega)
        return X, ok, [f"{c}_w{omega:g}" for c in columns]
    return build
