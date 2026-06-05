# -*- coding: utf-8 -*-
"""Compare spatial fields produced by WaveFilter and CCKWFilter.

This script runs the legacy ``WaveFilter`` and the newer ``CCKWFilter`` on the
same input data, then compares their time-std spatial fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from filters import WaveFilter, CCKWFilter

SUPPORTED_WAVES = ("kelvin", "er", "mrg", "ig", "mjo", "td")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spatial wave-filter fields from WaveFilter and CCKWFilter.",
    )
    parser.add_argument("--input", required=True, help="Input NetCDF file path.")
    parser.add_argument("--var", default="olr", help="Variable name inside the NetCDF file.")
    parser.add_argument(
        "--waves",
        nargs="+",
        default=["mjo", "kelvin"],
        choices=SUPPORTED_WAVES,
        help="Wave types to compare.",
    )
    parser.add_argument("--time-start", default="1979-01-01", help="Start date.")
    parser.add_argument("--time-end", default="2020-12-31", help="End date.")
    parser.add_argument("--lat-min", type=float, default=-25.0, help="Minimum latitude.")
    parser.add_argument("--lat-max", type=float, default=25.0, help="Maximum latitude.")
    parser.add_argument("--spd", type=int, default=1, help="Samples per day.")
    parser.add_argument(
        "--n-harm",
        type=int,
        default=3,
        help="Harmonics retained in WaveFilter climatology removal.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for WaveFilter.")
    parser.add_argument("--n-workers", type=int, default=4, help="Workers for CCKWFilter.")
    parser.add_argument(
        "--out-dir",
        default="./wave_filter_compare_outputs",
        help="Directory for plots and summary tables.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution in legacy WaveFilter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress from CCKWFilter.",
    )
    return parser.parse_args()


def load_data(
    input_path: str,
    var_name: str,
    time_start: str,
    time_end: str,
    lat_min: float,
    lat_max: float,
) -> xr.DataArray:
    ds = xr.open_dataset(input_path)
    da = ds[var_name]
    lat_slice = slice(lat_min, lat_max) if lat_min <= lat_max else slice(lat_max, lat_min)
    da = da.sel(time=slice(time_start, time_end), lat=lat_slice).sortby("lat")
    return da.transpose("time", "lat", "lon")


def run_legacy_filter(
    data: xr.DataArray,
    wave_name: str,
    spd: int,
    n_harm: int,
    use_parallel: bool,
    n_jobs: int,
) -> xr.DataArray:
    legacy = WaveFilter()
    return legacy.extract_wave_signal(
        data,
        wave_name=wave_name,
        obs_per_day=spd,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
        n_harm=n_harm,
    )


def run_cckw_filter(
    data: xr.DataArray,
    wave_name: str,
    spd: int,
    n_workers: int,
    verbose: bool,
) -> xr.DataArray:
    filt = CCKWFilter(
        ds=data,
        wave_name=wave_name,
        units=data.attrs.get("units", "unknown"),
        spd=spd,
        n_workers=n_workers,
        verbose=verbose,
    )
    return filt.process()


def compute_spatial_metrics(legacy_std: xr.DataArray, cckw_std: xr.DataArray) -> dict[str, float]:
    diff = legacy_std - cckw_std
    legacy_vals = legacy_std.values.ravel()
    cckw_vals = cckw_std.values.ravel()
    diff_vals = diff.values.ravel()
    valid = np.isfinite(legacy_vals) & np.isfinite(cckw_vals)
    if not np.any(valid):
        corr = np.nan
        rmse = np.nan
        mean_bias = np.nan
        mean_abs_diff = np.nan
    else:
        corr = float(np.corrcoef(legacy_vals[valid], cckw_vals[valid])[0, 1])
        rmse = float(np.sqrt(np.mean(diff_vals[valid] ** 2)))
        mean_bias = float(np.mean(diff_vals[valid]))
        mean_abs_diff = float(np.mean(np.abs(diff_vals[valid])))
    return {
        "legacy_mean_std": float(legacy_std.mean().item()),
        "cckw_mean_std": float(cckw_std.mean().item()),
        "mean_bias": mean_bias,
        "mean_abs_diff": mean_abs_diff,
        "rmse": rmse,
        "pattern_corr": corr,
    }


def plot_spatial_comparison(
    legacy_std: xr.DataArray,
    cckw_std: xr.DataArray,
    wave_name: str,
    time_start: str,
    time_end: str,
    out_path: Path,
) -> None:
    diff = legacy_std - cckw_std
    vmax = float(np.nanmax([legacy_std.max().item(), cckw_std.max().item()]))
    dmax = float(np.nanmax(np.abs(diff.values)))
    if not np.isfinite(dmax) or dmax == 0.0:
        dmax = 1.0e-8

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        constrained_layout=True,
        facecolor="white",
        dpi=180,
    )

    legacy_std.plot.contourf(
        ax=axes[0],
        levels=21,
        cmap="Spectral_r",
        vmin=0.0,
        vmax=vmax,
        add_colorbar=True,
    )
    axes[0].set_title(f"Legacy WaveFilter STD\n{wave_name.upper()}")

    cckw_std.plot.contourf(
        ax=axes[1],
        levels=21,
        cmap="Spectral_r",
        vmin=0.0,
        vmax=vmax,
        add_colorbar=True,
    )
    axes[1].set_title(f"CCKWFilter STD\n{wave_name.upper()}")

    diff.plot.contourf(
        ax=axes[2],
        levels=21,
        cmap="RdBu_r",
        vmin=-dmax,
        vmax=dmax,
        add_colorbar=True,
    )
    axes[2].set_title("Difference\nLegacy - CCKW")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        f"Spatial STD Comparison for {wave_name.upper()} ({time_start} to {time_end})",
        fontsize=14,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_fields(legacy_std: xr.DataArray, cckw_std: xr.DataArray, wave_name: str, out_dir: Path) -> None:
    diff = (legacy_std - cckw_std).rename(f"{wave_name}_std_diff")
    ds = xr.Dataset(
        {
            f"{wave_name}_legacy_std": legacy_std,
            f"{wave_name}_cckw_std": cckw_std,
            f"{wave_name}_legacy_minus_cckw": diff,
        }
    )
    ds.to_netcdf(out_dir / f"{wave_name}_spatial_std_comparison.nc")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(
        args.input,
        args.var,
        args.time_start,
        args.time_end,
        args.lat_min,
        args.lat_max,
    )
    print(f"Loaded data: {data.shape} for {args.time_start} to {args.time_end}")

    summary_rows: list[dict[str, float | str]] = []
    for wave_name in args.waves:
        print(f"\n=== Comparing wave: {wave_name} ===")
        legacy_filtered = run_legacy_filter(
            data,
            wave_name=wave_name,
            spd=args.spd,
            n_harm=args.n_harm,
            use_parallel=not args.no_parallel,
            n_jobs=args.n_jobs,
        )
        cckw_filtered = run_cckw_filter(
            data,
            wave_name=wave_name,
            spd=args.spd,
            n_workers=args.n_workers,
            verbose=args.verbose,
        )

        legacy_std = legacy_filtered.std("time").rename(f"{wave_name}_legacy_std")
        cckw_std = cckw_filtered.std("time").rename(f"{wave_name}_cckw_std")

        metrics = compute_spatial_metrics(legacy_std, cckw_std)
        metrics["wave"] = wave_name
        summary_rows.append(metrics)

        plot_spatial_comparison(
            legacy_std,
            cckw_std,
            wave_name,
            args.time_start,
            args.time_end,
            out_dir / f"{wave_name}_spatial_std_compare.png",
        )
        save_fields(legacy_std, cckw_std, wave_name, out_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "wave_filter_spatial_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
