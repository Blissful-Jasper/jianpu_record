from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "cckw_output"

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "xdg_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


WAVE_TOOLS_DIR = Path("/Users/lipu/Desktop/wave_tools")
if str(WAVE_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(WAVE_TOOLS_DIR))

from filters import CCKWFilter, WaveFilter


OLR_PATH = ROOT / "olr.day.mean.nc"
DEFAULT_START_DATE = "1980-01-01"
DEFAULT_END_DATE = "2022-12-31"
DEFAULT_PERIOD_MIN = 2.5
DEFAULT_PERIOD_MAX = 20.0
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 14
DEFAULT_RECENT_YEARS = 10
DEFAULT_WESTPAC_LON_MIN = 120.0
DEFAULT_WESTPAC_LON_MAX = 180.0

FILTERED_PATH = OUTPUT_DIR / "olr_kelvin_filtered_cckw_1974_2022.nc"
EVENTS_CSV = OUTPUT_DIR / "cckw_top_events.csv"
RECENT_EVENTS_CSV = OUTPUT_DIR / "cckw_recent10_top_events.csv"
RECENT_VISIBLE_EVENTS_CSV = OUTPUT_DIR / "cckw_recent10_satellite_visible_events.csv"
RECENT_WESTPAC_EVENTS_CSV = OUTPUT_DIR / "cckw_recent10_westpac_kelvin_pure_events.csv"
HOVMOLLER_FIG = OUTPUT_DIR / "cckw_hovmoller_strongest_event.png"
SNAPSHOT_FIG = OUTPUT_DIR / "cckw_event_snapshot.png"
RECENT_HOVMOLLER_FIG = OUTPUT_DIR / "cckw_recent10_hovmoller_strongest_event.png"
RECENT_SNAPSHOT_FIG = OUTPUT_DIR / "cckw_recent10_event_snapshot.png"
RECENT_VISIBLE_HOVMOLLER_FIG = OUTPUT_DIR / "cckw_recent10_visible_hovmoller_event.png"
RECENT_VISIBLE_SNAPSHOT_FIG = OUTPUT_DIR / "cckw_recent10_visible_event_snapshot.png"
RECENT_WESTPAC_HOVMOLLER_FIG = OUTPUT_DIR / "cckw_recent10_westpac_hovmoller_event.png"
RECENT_WESTPAC_SNAPSHOT_FIG = OUTPUT_DIR / "cckw_recent10_westpac_event_snapshot.png"
SUMMARY_TXT = OUTPUT_DIR / "cckw_summary.txt"


def _number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def filter_tag(
    start_date: str,
    end_date: str,
    period_days: tuple[float, float],
    wavenumber: tuple[int, int],
) -> str:
    return (
        f"{start_date}_{end_date}"
        f"_p{_number_tag(period_days[0])}-{_number_tag(period_days[1])}"
        f"_k{wavenumber[0]}-{wavenumber[1]}"
    )


def filtered_output_path(
    start_date: str,
    end_date: str,
    period_days: tuple[float, float],
    wavenumber: tuple[int, int],
) -> Path:
    return OUTPUT_DIR / f"olr_kelvin_filtered_cckw_{filter_tag(start_date, end_date, period_days, wavenumber)}.nc"


def wave_filtered_output_path(
    wave_name: str,
    start_date: str,
    end_date: str,
    period_days: tuple[float, float],
    wavenumber: tuple[int, int],
) -> Path:
    return OUTPUT_DIR / f"olr_{wave_name}_filtered_{filter_tag(start_date, end_date, period_days, wavenumber)}.nc"


def recent_window_from_data(time_coord: xr.DataArray, years: int = DEFAULT_RECENT_YEARS) -> tuple[str, str]:
    end = pd.Timestamp(time_coord.max().values).normalize()
    start = pd.Timestamp(year=end.year - years + 1, month=1, day=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def weighted_lat_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean("lat")


def daily_olr_anomaly(olr: xr.DataArray, n_harm: int = 3) -> xr.DataArray:
    wf = WaveFilter()
    clim = olr.groupby("time.dayofyear").mean("time")
    clim_fit = wf.extract_low_harmonics(clim, n_harm=n_harm)
    anomaly = (olr.groupby("time.dayofyear") - clim_fit).transpose("time", "lat", "lon")
    anomaly.name = "olr_anomaly"
    anomaly.attrs.update(
        {
            "long_name": "OLR anomaly after removing low-harmonic daily climatology",
            "units": olr.attrs.get("units", "W/m^2"),
        }
    )
    return anomaly


def fill_missing_olr(olr: xr.DataArray) -> xr.DataArray:
    missing_count = int(olr.isnull().sum().values)
    if missing_count == 0:
        return olr

    print(f"Filling {missing_count} missing OLR values before spectral filtering...")
    filled = olr.interpolate_na(dim="time", method="linear", use_coordinate=False)
    if int(filled.isnull().sum().values) > 0:
        filled = filled.interpolate_na(dim="lon", method="nearest", use_coordinate=False)
    if int(filled.isnull().sum().values) > 0:
        filled = filled.interpolate_na(dim="lat", method="nearest", use_coordinate=False)
    if int(filled.isnull().sum().values) > 0:
        filled = filled.fillna(filled.mean("time"))
    if int(filled.isnull().sum().values) > 0:
        filled = filled.fillna(float(filled.mean().values))

    filled.attrs.update(olr.attrs)
    filled.attrs["missing_value_treatment"] = (
        "NaNs filled with time-linear interpolation before CCKW spectral filtering; "
        "remaining NaNs, if any, filled spatially and by local climatological mean."
    )
    return filled


def run_kelvin_filter(
    olr: xr.DataArray,
    force: bool,
    n_workers: int,
    n_harm: int,
    period_days: tuple[float, float] = (DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX),
    wavenumber: tuple[int, int] = (DEFAULT_K_MIN, DEFAULT_K_MAX),
    output_path: Path | None = None,
) -> xr.DataArray:
    if output_path is None:
        start_date = pd.Timestamp(olr.time.min().values).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(olr.time.max().values).strftime("%Y-%m-%d")
        output_path = filtered_output_path(start_date, end_date, period_days, wavenumber)

    if output_path.exists() and not force:
        return xr.open_dataarray(output_path)

    original_kelvin_spec = CCKWFilter.WAVE_SPECS["kelvin"].copy()
    CCKWFilter.WAVE_SPECS["kelvin"] = {
        "period_days": period_days,
        "wavenumber": wavenumber,
        "equiv_depth": (8.0, 90.0),
        "meridional_mode": None,
        "dispersion_family": "kelvin",
    }
    try:
        wave_filter = CCKWFilter(
            ds=olr,
            sel_dict={"lat": slice(-15, 15)},
            wave_name="kelvin",
            units=olr.attrs.get("units", "W/m^2"),
            spd=1,
            n_workers=n_workers,
            verbose=True,
            n_harm=n_harm,
        )
        kelvin = wave_filter.process()
    finally:
        CCKWFilter.WAVE_SPECS["kelvin"] = original_kelvin_spec

    kelvin.name = "olr_kelvin_cckw"
    kelvin.attrs["description"] = (
        "Convectively coupled Kelvin wave OLR signal filtered with "
        f"wave_tools.filters.CCKWFilter: period {period_days[0]:g}-{period_days[1]:g} days, "
        f"eastward zonal wavenumber {wavenumber[0]}-{wavenumber[1]}, "
        "equivalent depth 8-90 m."
    )

    encoding = {
        kelvin.name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    kelvin.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    return kelvin


def run_wave_filter(
    olr: xr.DataArray,
    wave_name: str,
    period_days: tuple[float, float],
    wavenumber: tuple[int, int],
    force: bool,
    n_workers: int,
    n_harm: int,
    output_path: Path,
    equiv_depth: tuple[float | None, float | None] = (None, None),
    meridional_mode: int | None = None,
    dispersion_family: str = "none",
) -> xr.DataArray:
    if output_path.exists() and not force:
        return xr.open_dataarray(output_path)

    wave_key = wave_name.lower()
    original_spec = CCKWFilter.WAVE_SPECS.get(wave_key, {}).copy()
    CCKWFilter.WAVE_SPECS[wave_key] = {
        "period_days": period_days,
        "wavenumber": wavenumber,
        "equiv_depth": equiv_depth,
        "meridional_mode": meridional_mode,
        "dispersion_family": dispersion_family,
    }
    try:
        wave_filter = CCKWFilter(
            ds=olr,
            sel_dict={"lat": slice(-15, 15)},
            wave_name=wave_key,
            units=olr.attrs.get("units", "W/m^2"),
            spd=1,
            n_workers=n_workers,
            verbose=True,
            n_harm=n_harm,
        )
        wave = wave_filter.process()
    finally:
        if original_spec:
            CCKWFilter.WAVE_SPECS[wave_key] = original_spec
        else:
            CCKWFilter.WAVE_SPECS.pop(wave_key, None)

    wave.name = f"olr_{wave_key}_filtered"
    wave.attrs["description"] = (
        f"{wave_key.upper()} diagnostic OLR signal filtered with "
        f"wave_tools.filters.CCKWFilter: period {period_days[0]:g}-{period_days[1]:g} days, "
        f"zonal wavenumber {wavenumber[0]}-{wavenumber[1]}."
    )
    encoding = {
        wave.name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    wave.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    return wave


def nearest_point_value(da: xr.DataArray, time_value, lon_value: float) -> float:
    return float(da.sel(time=time_value, lon=lon_value, method="nearest").values)


def longitude_box(da: xr.DataArray, center_lon: float, half_width: float = 20.0) -> xr.DataArray:
    lon_rel = ((da.lon - center_lon + 180.0) % 360.0) - 180.0
    return da.where(np.abs(lon_rel) <= half_width, drop=True)


def event_box_metrics(
    anomaly: xr.DataArray,
    olr: xr.DataArray,
    time_value,
    lon_value: float,
    lon_half_width: float = 20.0,
    cold_olr_threshold: float = 210.0,
) -> dict[str, float]:
    anomaly_box = longitude_box(anomaly.sel(time=time_value), lon_value, half_width=lon_half_width)
    olr_box = longitude_box(olr.sel(time=time_value), lon_value, half_width=lon_half_width)
    cold_fraction = float((olr_box <= cold_olr_threshold).mean().values) * 100.0
    strong_anomaly_fraction = float((anomaly_box <= -30.0).mean().values) * 100.0
    return {
        "box_min_olr_w_m2": float(olr_box.min().values),
        "box_min_olr_anomaly_w_m2": float(anomaly_box.min().values),
        "box_cold_olr_fraction_pct": cold_fraction,
        "box_strong_negative_anomaly_fraction_pct": strong_anomaly_fraction,
    }


def identify_events(
    kelvin_eq: xr.DataArray,
    anomaly_eq: xr.DataArray,
    olr_eq: xr.DataArray,
    kelvin_3d: xr.DataArray,
    n_events: int = 10,
    separation_days: int = 10,
    event_start: str | None = None,
    event_end: str | None = None,
    output_csv: Path | None = EVENTS_CSV,
) -> pd.DataFrame:
    if event_start is not None or event_end is not None:
        kelvin_eq = kelvin_eq.sel(time=slice(event_start, event_end))
        anomaly_eq = anomaly_eq.sel(time=slice(event_start, event_end))
        olr_eq = olr_eq.sel(time=slice(event_start, event_end))
        kelvin_3d = kelvin_3d.sel(time=slice(event_start, event_end))

    values = kelvin_eq.values.astype(float).copy()
    times = pd.to_datetime(kelvin_eq.time.values)
    lons = kelvin_eq.lon.values.astype(float)
    rows = []

    for rank in range(1, n_events + 1):
        if not np.isfinite(values).any():
            break
        flat_idx = np.nanargmin(values)
        time_idx, lon_idx = np.unravel_index(flat_idx, values.shape)
        event_time = times[time_idx]
        event_lon = float(lons[lon_idx])
        kelvin_value = float(values[time_idx, lon_idx])
        raw_anomaly_value = nearest_point_value(anomaly_eq, event_time, event_lon)
        raw_olr_value = nearest_point_value(olr_eq, event_time, event_lon)
        lat_slice = kelvin_3d.sel(time=event_time, lon=event_lon, method="nearest")
        event_lat = float(lat_slice.idxmin("lat").values)
        kelvin_3d_min = float(lat_slice.min("lat").values)

        rows.append(
            {
                "rank": rank,
                "date": event_time.strftime("%Y-%m-%d"),
                "longitude_deg_east": event_lon,
                "latitude_of_min_deg_north": event_lat,
                "kelvin_olr_anomaly_w_m2_latmean": kelvin_value,
                "kelvin_olr_anomaly_w_m2_point_min": kelvin_3d_min,
                "unfiltered_olr_anomaly_w_m2_latmean": raw_anomaly_value,
                "raw_olr_w_m2_latmean": raw_olr_value,
                "event_search_start": event_start,
                "event_search_end": event_end,
            }
        )

        day_distance = np.abs((times - event_time).days)
        values[day_distance <= separation_days, :] = np.nan

    events = pd.DataFrame(rows)
    if output_csv is not None:
        events.to_csv(output_csv, index=False)
    return events


def identify_satellite_visible_events(
    kelvin_eq: xr.DataArray,
    anomaly_eq: xr.DataArray,
    olr_eq: xr.DataArray,
    kelvin_3d: xr.DataArray,
    anomaly_3d: xr.DataArray,
    olr_3d: xr.DataArray,
    n_events: int = 10,
    separation_days: int = 10,
    event_start: str | None = None,
    event_end: str | None = None,
    output_csv: Path | None = RECENT_VISIBLE_EVENTS_CSV,
    min_kelvin_strength: float = 22.0,
    min_raw_anomaly_strength: float = 25.0,
    max_raw_olr: float = 220.0,
) -> pd.DataFrame:
    if event_start is not None or event_end is not None:
        kelvin_eq = kelvin_eq.sel(time=slice(event_start, event_end))
        anomaly_eq = anomaly_eq.sel(time=slice(event_start, event_end))
        olr_eq = olr_eq.sel(time=slice(event_start, event_end))
        kelvin_3d = kelvin_3d.sel(time=slice(event_start, event_end))
        anomaly_3d = anomaly_3d.sel(time=slice(event_start, event_end))
        olr_3d = olr_3d.sel(time=slice(event_start, event_end))

    kelvin_values = kelvin_eq.values.astype(float)
    anomaly_values = anomaly_eq.values.astype(float)
    olr_values = olr_eq.values.astype(float)

    kelvin_strength = -kelvin_values
    anomaly_strength = -anomaly_values
    cold_strength = 240.0 - olr_values
    score = kelvin_strength + 0.50 * anomaly_strength + 0.25 * cold_strength

    candidate_mask = (
        (kelvin_strength >= min_kelvin_strength)
        & (anomaly_strength >= min_raw_anomaly_strength)
        & (olr_values <= max_raw_olr)
        & np.isfinite(score)
    )
    score = np.where(candidate_mask, score, np.nan)

    times = pd.to_datetime(kelvin_eq.time.values)
    lons = kelvin_eq.lon.values.astype(float)
    rows = []

    for rank in range(1, n_events + 1):
        if not np.isfinite(score).any():
            break
        flat_idx = np.nanargmax(score)
        time_idx, lon_idx = np.unravel_index(flat_idx, score.shape)
        event_time = times[time_idx]
        event_lon = float(lons[lon_idx])
        kelvin_value = float(kelvin_values[time_idx, lon_idx])
        raw_anomaly_value = float(anomaly_values[time_idx, lon_idx])
        raw_olr_value = float(olr_values[time_idx, lon_idx])

        lat_slice = kelvin_3d.sel(time=event_time, lon=event_lon, method="nearest")
        event_lat = float(lat_slice.idxmin("lat").values)
        kelvin_3d_min = float(lat_slice.min("lat").values)
        box_metrics = event_box_metrics(anomaly_3d, olr_3d, event_time, event_lon)

        rows.append(
            {
                "rank": rank,
                "date": event_time.strftime("%Y-%m-%d"),
                "longitude_deg_east": event_lon,
                "latitude_of_min_deg_north": event_lat,
                "satellite_visibility_score": float(score[time_idx, lon_idx]),
                "kelvin_olr_anomaly_w_m2_latmean": kelvin_value,
                "kelvin_olr_anomaly_w_m2_point_min": kelvin_3d_min,
                "unfiltered_olr_anomaly_w_m2_latmean": raw_anomaly_value,
                "raw_olr_w_m2_latmean": raw_olr_value,
                **box_metrics,
                "event_search_start": event_start,
                "event_search_end": event_end,
            }
        )

        day_distance = np.abs((times - event_time).days)
        score[day_distance <= separation_days, :] = np.nan

    events = pd.DataFrame(rows)
    if output_csv is not None:
        events.to_csv(output_csv, index=False)
    return events


def lon_range_mask(lons: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lons = np.asarray(lons, dtype=float) % 360.0
    lon_min = lon_min % 360.0
    lon_max = lon_max % 360.0
    if lon_min <= lon_max:
        return (lons >= lon_min) & (lons <= lon_max)
    return (lons >= lon_min) | (lons <= lon_max)


def identify_westpac_kelvin_candidates(
    kelvin_eq: xr.DataArray,
    anomaly_eq: xr.DataArray,
    olr_eq: xr.DataArray,
    mjo_eq: xr.DataArray,
    er_eq: xr.DataArray,
    kelvin_3d: xr.DataArray,
    anomaly_3d: xr.DataArray,
    olr_3d: xr.DataArray,
    n_events: int = 10,
    separation_days: int = 10,
    event_start: str | None = None,
    event_end: str | None = None,
    lon_min: float = DEFAULT_WESTPAC_LON_MIN,
    lon_max: float = DEFAULT_WESTPAC_LON_MAX,
    output_csv: Path | None = RECENT_WESTPAC_EVENTS_CSV,
    min_kelvin_strength: float = 20.0,
    min_raw_anomaly_strength: float = 20.0,
    max_raw_olr: float = 225.0,
    min_purity_ratio: float = 0.65,
) -> pd.DataFrame:
    if event_start is not None or event_end is not None:
        time_slice = slice(event_start, event_end)
        kelvin_eq = kelvin_eq.sel(time=time_slice)
        anomaly_eq = anomaly_eq.sel(time=time_slice)
        olr_eq = olr_eq.sel(time=time_slice)
        mjo_eq = mjo_eq.sel(time=time_slice)
        er_eq = er_eq.sel(time=time_slice)
        kelvin_3d = kelvin_3d.sel(time=time_slice)
        anomaly_3d = anomaly_3d.sel(time=time_slice)
        olr_3d = olr_3d.sel(time=time_slice)

    kelvin_values = kelvin_eq.values.astype(float)
    anomaly_values = anomaly_eq.values.astype(float)
    olr_values = olr_eq.values.astype(float)
    mjo_abs = np.abs(mjo_eq.values.astype(float))
    er_abs = np.abs(er_eq.values.astype(float))

    kelvin_strength = -kelvin_values
    anomaly_strength = -anomaly_values
    cold_strength = 240.0 - olr_values
    purity_ratio = kelvin_strength / (1.0 + mjo_abs + er_abs)
    score = (
        kelvin_strength
        + 0.50 * anomaly_strength
        + 0.25 * cold_strength
        + 12.0 * np.clip(purity_ratio, 0.0, 3.0)
        - 0.35 * (mjo_abs + er_abs)
    )

    lon_mask = lon_range_mask(kelvin_eq.lon.values, lon_min, lon_max)[np.newaxis, :]
    candidate_mask = (
        lon_mask
        & (kelvin_strength >= min_kelvin_strength)
        & (anomaly_strength >= min_raw_anomaly_strength)
        & (olr_values <= max_raw_olr)
        & (purity_ratio >= min_purity_ratio)
        & np.isfinite(score)
    )
    score = np.where(candidate_mask, score, np.nan)

    times = pd.to_datetime(kelvin_eq.time.values)
    lons = kelvin_eq.lon.values.astype(float)
    rows = []

    for rank in range(1, n_events + 1):
        if not np.isfinite(score).any():
            break
        flat_idx = np.nanargmax(score)
        time_idx, lon_idx = np.unravel_index(flat_idx, score.shape)
        event_time = times[time_idx]
        event_lon = float(lons[lon_idx])

        lat_slice = kelvin_3d.sel(time=event_time, lon=event_lon, method="nearest")
        event_lat = float(lat_slice.idxmin("lat").values)
        kelvin_3d_min = float(lat_slice.min("lat").values)
        box_metrics = event_box_metrics(anomaly_3d, olr_3d, event_time, event_lon)

        rows.append(
            {
                "rank": rank,
                "date": event_time.strftime("%Y-%m-%d"),
                "longitude_deg_east": event_lon,
                "latitude_of_min_deg_north": event_lat,
                "westpac_kelvin_score": float(score[time_idx, lon_idx]),
                "kelvin_purity_ratio": float(purity_ratio[time_idx, lon_idx]),
                "mjo_abs_w_m2_latmean": float(mjo_abs[time_idx, lon_idx]),
                "er_abs_w_m2_latmean": float(er_abs[time_idx, lon_idx]),
                "kelvin_olr_anomaly_w_m2_latmean": float(kelvin_values[time_idx, lon_idx]),
                "kelvin_olr_anomaly_w_m2_point_min": kelvin_3d_min,
                "unfiltered_olr_anomaly_w_m2_latmean": float(anomaly_values[time_idx, lon_idx]),
                "raw_olr_w_m2_latmean": float(olr_values[time_idx, lon_idx]),
                **box_metrics,
                "westpac_lon_min": lon_min,
                "westpac_lon_max": lon_max,
                "event_search_start": event_start,
                "event_search_end": event_end,
            }
        )

        day_distance = np.abs((times - event_time).days)
        score[day_distance <= separation_days, :] = np.nan

    events = pd.DataFrame(rows)
    if output_csv is not None:
        events.to_csv(output_csv, index=False)
    return events


def recenter_longitude(da: xr.DataArray, center_lon: float) -> xr.DataArray:
    lon_rel = ((da.lon - center_lon + 180.0) % 360.0) - 180.0
    out = da.assign_coords(lon_rel=("lon", lon_rel.values)).swap_dims({"lon": "lon_rel"})
    out = out.drop_vars("lon").sortby("lon_rel")
    return out


def robust_vlim(*arrays: xr.DataArray, percentile: float = 98.0, minimum: float = 5.0) -> float:
    vals = []
    for arr in arrays:
        data = np.asarray(arr.values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size:
            vals.append(np.nanpercentile(np.abs(data), percentile))
    if not vals:
        return minimum
    return max(minimum, float(np.nanmax(vals)))


def speed_to_deg_per_day(speed_mps: float) -> float:
    return speed_mps * 86400.0 / 111320.0


def add_wave_speed_guides(
    ax,
    event_time: pd.Timestamp,
    days_before: int,
    days_after: int,
) -> None:
    guide_specs = [
        ("Kelvin 12 m/s", 12.0, "k", "--"),
        ("MJO 5 m/s", 5.0, "tab:green", ":"),
        ("ER/Rossby -5 m/s", -5.0, "tab:purple", "-."),
    ]
    delta_days = np.linspace(-days_before, days_after, 101)
    dates = event_time + pd.to_timedelta(delta_days, unit="D")
    for label, speed, color, linestyle in guide_specs:
        x = speed_to_deg_per_day(speed) * delta_days
        ax.plot(x, dates, color=color, lw=1.4, ls=linestyle, alpha=0.9, label=label)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.82)


def plot_hovmoller(
    anomaly_eq: xr.DataArray,
    kelvin_eq: xr.DataArray,
    event: pd.Series,
    days_before: int = 25,
    days_after: int = 25,
    save_path: Path = HOVMOLLER_FIG,
    title_prefix: str = "Strongest CCKW event in NOAA daily OLR",
    add_guides: bool = True,
) -> None:
    event_time = pd.Timestamp(event["date"])
    event_lon = float(event["longitude_deg_east"])
    time_slice = slice(event_time - pd.Timedelta(days=days_before), event_time + pd.Timedelta(days=days_after))

    raw = recenter_longitude(anomaly_eq.sel(time=time_slice), event_lon)
    filt = recenter_longitude(kelvin_eq.sel(time=time_slice), event_lon)
    vlim_raw = robust_vlim(raw, percentile=98.0, minimum=20.0)
    vlim_filt = robust_vlim(filt, percentile=98.0, minimum=8.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, constrained_layout=True)
    panels = [
        (axes[0], raw, vlim_raw, "Unfiltered OLR anomaly"),
        (axes[1], filt, vlim_filt, "CCKW-filtered OLR"),
    ]

    for ax, data, vlim, title in panels:
        mesh = ax.contourf(
            data.lon_rel.values,
            pd.to_datetime(data.time.values),
            data.values,
            levels=np.linspace(-vlim, vlim, 25),
            cmap="RdBu_r",
            extend="both",
        )
        ax.axhline(event_time, color="0.15", lw=1.0, ls="--")
        ax.axvline(0, color="0.15", lw=1.0, ls="--")
        ax.plot(0, event_time, marker="*", ms=13, color="gold", mec="black", mew=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Longitude relative to {event_lon:.1f}E (degrees)")
        ax.set_xlim(-180, 180)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        if add_guides and ax is axes[0]:
            add_wave_speed_guides(ax, event_time, days_before, days_after)
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, shrink=0.86)
        cbar.set_label("W m$^{-2}$")

    axes[0].set_ylabel("Date")
    axes[0].yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.suptitle(
        (
            f"{title_prefix}: "
            f"{event_time:%Y-%m-%d}, {event_lon:.1f}E"
        ),
        fontsize=14,
    )
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_snapshot(
    anomaly: xr.DataArray,
    kelvin: xr.DataArray,
    event: pd.Series,
    save_path: Path = SNAPSHOT_FIG,
    title_prefix: str = "OLR fields on strongest CCKW date",
) -> None:
    event_time = pd.Timestamp(event["date"])
    event_lon = float(event["longitude_deg_east"])
    raw = anomaly.sel(time=event_time, lat=slice(-20, 20))
    filt = kelvin.sel(time=event_time, lat=slice(-20, 20))
    vlim_raw = robust_vlim(raw, percentile=98.0, minimum=20.0)
    vlim_filt = robust_vlim(filt, percentile=98.0, minimum=8.0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], raw, vlim_raw, "Unfiltered OLR anomaly"),
        (axes[1], filt, vlim_filt, "CCKW-filtered OLR"),
    ]
    for ax, data, vlim, title in panels:
        mesh = ax.contourf(
            data.lon.values,
            data.lat.values,
            data.values,
            levels=np.linspace(-vlim, vlim, 25),
            cmap="RdBu_r",
            extend="both",
        )
        ax.axvline(event_lon, color="0.15", lw=1.0, ls="--")
        ax.axhline(0, color="0.15", lw=0.8)
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=12)
        ax.set_ylim(float(data.lat.min()), float(data.lat.max()))
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.015, shrink=0.9)
        cbar.set_label("W m$^{-2}$")
    axes[-1].set_xlabel("Longitude (degrees east)")
    fig.suptitle(f"{title_prefix}: {event_time:%Y-%m-%d}", fontsize=14)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _event_text(label: str, events: pd.DataFrame) -> str:
    strongest = events.iloc[0]
    return f"""{label}:
  date: {strongest['date']}
  longitude: {strongest['longitude_deg_east']:.1f}E
  latitude of 3D minimum at that date/lon: {strongest['latitude_of_min_deg_north']:.1f}N
  Kelvin OLR anomaly, lat-mean: {strongest['kelvin_olr_anomaly_w_m2_latmean']:.2f} W m-2
  Kelvin OLR anomaly, point min: {strongest['kelvin_olr_anomaly_w_m2_point_min']:.2f} W m-2
  unfiltered OLR anomaly, lat-mean: {strongest['unfiltered_olr_anomaly_w_m2_latmean']:.2f} W m-2
  raw OLR, lat-mean: {strongest['raw_olr_w_m2_latmean']:.2f} W m-2
"""


def write_summary(
    events: pd.DataFrame,
    recent_events: pd.DataFrame | None = None,
    visible_events: pd.DataFrame | None = None,
    westpac_events: pd.DataFrame | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    period_days: tuple[float, float] = (DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX),
    wavenumber: tuple[int, int] = (DEFAULT_K_MIN, DEFAULT_K_MAX),
    filtered_path: Path = FILTERED_PATH,
    events_csv: Path = EVENTS_CSV,
    recent_events_csv: Path | None = RECENT_EVENTS_CSV,
    visible_events_csv: Path | None = RECENT_VISIBLE_EVENTS_CSV,
    westpac_events_csv: Path | None = RECENT_WESTPAC_EVENTS_CSV,
    hovmoller_fig: Path = HOVMOLLER_FIG,
    snapshot_fig: Path = SNAPSHOT_FIG,
    recent_hovmoller_fig: Path | None = RECENT_HOVMOLLER_FIG,
    recent_snapshot_fig: Path | None = RECENT_SNAPSHOT_FIG,
    visible_hovmoller_fig: Path | None = RECENT_VISIBLE_HOVMOLLER_FIG,
    visible_snapshot_fig: Path | None = RECENT_VISIBLE_SNAPSHOT_FIG,
    westpac_hovmoller_fig: Path | None = RECENT_WESTPAC_HOVMOLLER_FIG,
    westpac_snapshot_fig: Path | None = RECENT_WESTPAC_SNAPSHOT_FIG,
) -> None:
    recent_block = ""
    if recent_events is not None and not recent_events.empty:
        recent_block = "\n" + _event_text("Strongest recent-decade event", recent_events)
    visible_block = ""
    if visible_events is not None and not visible_events.empty:
        visible_block = "\n" + _event_text("Most satellite-visible recent-decade candidate", visible_events)
    westpac_block = ""
    if westpac_events is not None and not westpac_events.empty:
        westpac_block = "\n" + _event_text("West-Pacific Kelvin-pure recent-decade candidate", westpac_events)

    text = f"""CCKW OLR analysis summary
=========================

Input OLR:
  {OLR_PATH}

Wave tools:
  {WAVE_TOOLS_DIR}

Filter:
  wave_tools.filters.CCKWFilter, wave_name='kelvin'
  analysis time: {start_date} to {end_date}
  period: {period_days[0]:g}-{period_days[1]:g} days
  zonal wavenumber: {wavenumber[0]}-{wavenumber[1]}, eastward
  equivalent depth: 8-90 m
  latitude band: 15S-15N

{_event_text("Strongest full-period event", events)}{recent_block}{visible_block}{westpac_block}

Outputs:
  filtered NetCDF: {filtered_path}
  full-period top events CSV: {events_csv}
  full-period Hovmoller comparison: {hovmoller_fig}
  full-period event snapshot: {snapshot_fig}
  recent-decade top events CSV: {recent_events_csv}
  recent-decade Hovmoller comparison: {recent_hovmoller_fig}
  recent-decade event snapshot: {recent_snapshot_fig}
  satellite-visible recent-decade events CSV: {visible_events_csv}
  satellite-visible recent-decade Hovmoller comparison: {visible_hovmoller_fig}
  satellite-visible recent-decade event snapshot: {visible_snapshot_fig}
  West-Pacific Kelvin-pure recent-decade events CSV: {westpac_events_csv}
  West-Pacific Kelvin-pure recent-decade Hovmoller comparison: {westpac_hovmoller_fig}
  West-Pacific Kelvin-pure recent-decade event snapshot: {westpac_snapshot_fig}
"""
    SUMMARY_TXT.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter NOAA OLR for convectively coupled Kelvin waves.")
    parser.add_argument("--force-filter", action="store_true", help="recompute the Kelvin-filtered NetCDF")
    parser.add_argument("--n-workers", type=int, default=4, help="thread workers for latitude filtering")
    parser.add_argument("--n-harm", type=int, default=3, help="annual-cycle harmonics removed before filtering")
    parser.add_argument("--top-n", type=int, default=10, help="number of separated events to report")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="analysis start date")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="analysis end date")
    parser.add_argument("--period-min", type=float, default=DEFAULT_PERIOD_MIN, help="minimum Kelvin period in days")
    parser.add_argument("--period-max", type=float, default=DEFAULT_PERIOD_MAX, help="maximum Kelvin period in days")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN, help="minimum eastward zonal wavenumber")
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX, help="maximum eastward zonal wavenumber")
    parser.add_argument("--recent-years", type=int, default=DEFAULT_RECENT_YEARS, help="years at the end of the data used for recent-event ranking")
    parser.add_argument("--westpac-lon-min", type=float, default=DEFAULT_WESTPAC_LON_MIN, help="western Pacific candidate longitude minimum")
    parser.add_argument("--westpac-lon-max", type=float, default=DEFAULT_WESTPAC_LON_MAX, help="western Pacific candidate longitude maximum")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "xdg_cache").mkdir(parents=True, exist_ok=True)

    period_days = (args.period_min, args.period_max)
    wavenumber = (args.k_min, args.k_max)
    output_path = filtered_output_path(args.start_date, args.end_date, period_days, wavenumber)
    tag = filter_tag(args.start_date, args.end_date, period_days, wavenumber)
    events_csv = OUTPUT_DIR / f"cckw_top_events_{tag}.csv"
    recent_events_csv = OUTPUT_DIR / f"cckw_recent{args.recent_years}_top_events_{tag}.csv"
    visible_events_csv = OUTPUT_DIR / f"cckw_recent{args.recent_years}_satellite_visible_events_{tag}.csv"
    westpac_events_csv = OUTPUT_DIR / f"cckw_recent{args.recent_years}_westpac_kelvin_pure_events_{tag}.csv"
    hovmoller_fig = OUTPUT_DIR / f"cckw_hovmoller_strongest_event_{tag}.png"
    snapshot_fig = OUTPUT_DIR / f"cckw_event_snapshot_{tag}.png"
    recent_hovmoller_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_hovmoller_strongest_event_{tag}.png"
    recent_snapshot_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_event_snapshot_{tag}.png"
    visible_hovmoller_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_visible_hovmoller_event_{tag}.png"
    visible_snapshot_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_visible_event_snapshot_{tag}.png"
    westpac_hovmoller_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_westpac_hovmoller_event_{tag}.png"
    westpac_snapshot_fig = OUTPUT_DIR / f"cckw_recent{args.recent_years}_westpac_event_snapshot_{tag}.png"
    mjo_path = wave_filtered_output_path("mjo", args.start_date, args.end_date, (30.0, 100.0), (1, 5))
    er_path = wave_filtered_output_path("er", args.start_date, args.end_date, (9.0, 72.0), (-10, -1))

    ds = xr.open_dataset(OLR_PATH)
    olr = (
        ds["olr"]
        .sortby("lat")
        .sel(time=slice(args.start_date, args.end_date), lat=slice(-15, 15))
        .transpose("time", "lat", "lon")
    )
    olr.load()
    olr = fill_missing_olr(olr)

    print("Computing unfiltered OLR anomaly...")
    anomaly = daily_olr_anomaly(olr, n_harm=args.n_harm)

    print("Computing or loading CCKW-filtered OLR...")
    kelvin = run_kelvin_filter(
        olr,
        force=args.force_filter,
        n_workers=args.n_workers,
        n_harm=args.n_harm,
        period_days=period_days,
        wavenumber=wavenumber,
        output_path=output_path,
    )

    print("Computing or loading diagnostic MJO-filtered OLR for event purity...")
    mjo = run_wave_filter(
        olr,
        wave_name="mjo",
        period_days=(30.0, 96.0),
        wavenumber=(1, 5),
        force=args.force_filter,
        n_workers=args.n_workers,
        n_harm=args.n_harm,
        output_path=mjo_path,
    )

    print("Computing or loading diagnostic ER/Rossby-filtered OLR for event purity...")
    er = run_wave_filter(
        olr,
        wave_name="er",
        period_days=(6.25, 48.0),
        wavenumber=(-10, -1),
        force=args.force_filter,
        n_workers=args.n_workers,
        n_harm=args.n_harm,
        output_path=er_path,
        equiv_depth=(8.0, 90.0),
        meridional_mode=1,
        dispersion_family="er",
    )

    anomaly_eq = weighted_lat_mean(anomaly)
    kelvin_eq = weighted_lat_mean(kelvin)
    mjo_eq = weighted_lat_mean(mjo)
    er_eq = weighted_lat_mean(er)
    olr_eq = weighted_lat_mean(olr)
    recent_start, recent_end = recent_window_from_data(kelvin_eq.time, years=args.recent_years)

    print("Identifying strongest CCKW events for the full period...")
    events = identify_events(
        kelvin_eq=kelvin_eq,
        anomaly_eq=anomaly_eq,
        olr_eq=olr_eq,
        kelvin_3d=kelvin,
        n_events=args.top_n,
        output_csv=events_csv,
    )

    print(f"Identifying strongest CCKW events for recent {args.recent_years} years: {recent_start} to {recent_end}...")
    recent_events = identify_events(
        kelvin_eq=kelvin_eq,
        anomaly_eq=anomaly_eq,
        olr_eq=olr_eq,
        kelvin_3d=kelvin,
        n_events=args.top_n,
        event_start=recent_start,
        event_end=recent_end,
        output_csv=recent_events_csv,
    )

    print(f"Identifying satellite-visible CCKW candidates for recent {args.recent_years} years...")
    visible_events = identify_satellite_visible_events(
        kelvin_eq=kelvin_eq,
        anomaly_eq=anomaly_eq,
        olr_eq=olr_eq,
        kelvin_3d=kelvin,
        anomaly_3d=anomaly,
        olr_3d=olr,
        n_events=args.top_n,
        event_start=recent_start,
        event_end=recent_end,
        output_csv=visible_events_csv,
    )

    print(
        "Identifying West-Pacific Kelvin-pure candidates "
        f"({args.westpac_lon_min:g}E-{args.westpac_lon_max:g}E) for recent {args.recent_years} years..."
    )
    westpac_events = identify_westpac_kelvin_candidates(
        kelvin_eq=kelvin_eq,
        anomaly_eq=anomaly_eq,
        olr_eq=olr_eq,
        mjo_eq=mjo_eq,
        er_eq=er_eq,
        kelvin_3d=kelvin,
        anomaly_3d=anomaly,
        olr_3d=olr,
        n_events=args.top_n,
        event_start=recent_start,
        event_end=recent_end,
        lon_min=args.westpac_lon_min,
        lon_max=args.westpac_lon_max,
        output_csv=westpac_events_csv,
    )

    strongest = events.iloc[0]
    recent_strongest = recent_events.iloc[0]
    visible_strongest = visible_events.iloc[0]
    westpac_strongest = westpac_events.iloc[0]
    print("\nFull-period events:")
    print(events.to_string(index=False))
    print(f"\nRecent {args.recent_years}-year events:")
    print(recent_events.to_string(index=False))
    print(f"\nRecent {args.recent_years}-year satellite-visible candidates:")
    print(visible_events.to_string(index=False))
    print(f"\nRecent {args.recent_years}-year West-Pacific Kelvin-pure candidates:")
    print(westpac_events.to_string(index=False))

    print("Plotting full-period Hovmoller comparison...")
    plot_hovmoller(
        anomaly_eq,
        kelvin_eq,
        strongest,
        save_path=hovmoller_fig,
        title_prefix="Strongest full-period CCKW event in NOAA daily OLR",
    )

    print("Plotting full-period event snapshot...")
    plot_snapshot(
        anomaly,
        kelvin,
        strongest,
        save_path=snapshot_fig,
        title_prefix="OLR fields on strongest full-period CCKW date",
    )

    print("Plotting recent-decade Hovmoller comparison...")
    plot_hovmoller(
        anomaly_eq,
        kelvin_eq,
        recent_strongest,
        save_path=recent_hovmoller_fig,
        title_prefix=f"Strongest recent {args.recent_years}-year CCKW event in NOAA daily OLR",
    )

    print("Plotting recent-decade event snapshot...")
    plot_snapshot(
        anomaly,
        kelvin,
        recent_strongest,
        save_path=recent_snapshot_fig,
        title_prefix=f"OLR fields on strongest recent {args.recent_years}-year CCKW date",
    )

    print("Plotting satellite-visible candidate Hovmoller comparison...")
    plot_hovmoller(
        anomaly_eq,
        kelvin_eq,
        visible_strongest,
        save_path=visible_hovmoller_fig,
        title_prefix=f"Satellite-visible recent {args.recent_years}-year CCKW candidate in NOAA daily OLR",
    )

    print("Plotting satellite-visible candidate snapshot...")
    plot_snapshot(
        anomaly,
        kelvin,
        visible_strongest,
        save_path=visible_snapshot_fig,
        title_prefix=f"OLR fields on satellite-visible recent {args.recent_years}-year CCKW candidate",
    )

    print("Plotting West-Pacific Kelvin-pure candidate Hovmoller comparison...")
    plot_hovmoller(
        anomaly_eq,
        kelvin_eq,
        westpac_strongest,
        save_path=westpac_hovmoller_fig,
        title_prefix=f"West-Pacific Kelvin-pure recent {args.recent_years}-year CCKW candidate",
    )

    print("Plotting West-Pacific Kelvin-pure candidate snapshot...")
    plot_snapshot(
        anomaly,
        kelvin,
        westpac_strongest,
        save_path=westpac_snapshot_fig,
        title_prefix=f"OLR fields on West-Pacific Kelvin-pure recent {args.recent_years}-year CCKW candidate",
    )

    write_summary(
        events,
        recent_events=recent_events,
        visible_events=visible_events,
        westpac_events=westpac_events,
        start_date=args.start_date,
        end_date=args.end_date,
        period_days=period_days,
        wavenumber=wavenumber,
        filtered_path=output_path,
        events_csv=events_csv,
        recent_events_csv=recent_events_csv,
        visible_events_csv=visible_events_csv,
        westpac_events_csv=westpac_events_csv,
        hovmoller_fig=hovmoller_fig,
        snapshot_fig=snapshot_fig,
        recent_hovmoller_fig=recent_hovmoller_fig,
        recent_snapshot_fig=recent_snapshot_fig,
        visible_hovmoller_fig=visible_hovmoller_fig,
        visible_snapshot_fig=visible_snapshot_fig,
        westpac_hovmoller_fig=westpac_hovmoller_fig,
        westpac_snapshot_fig=westpac_snapshot_fig,
    )
    print(f"Done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
