#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced EOF Analysis for Atmospheric Data with Land Masking
Created on Mon Jul 21 2025
Enhanced version with better structure and land masking capability
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, List, Optional, Union, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from global_land_mask import globe
import warnings
warnings.filterwarnings('ignore')


class EOFAnalyzer:
    """
    A comprehensive class for EOF analysis of atmospheric data with masking capabilities.
    """
    
    def __init__(self, apply_land_mask: bool = True, ocean_only: bool = True, mask_resolution: str = 'c'):
        """
        Initialize the EOF analyzer.
        
        Parameters:
        -----------
        apply_land_mask : bool
            Whether to apply land/ocean masking
        ocean_only : bool
            If True, keep only ocean points; if False, keep only land points
        mask_resolution : str
            Resolution for global-land-mask ('c', 'l', 'i', 'h', 'f')
            'c': coarse ~50km, 'l': low ~10km, 'i': intermediate ~2km, 
            'h': high ~400m, 'f': full ~100m (recommended: 'c' for global studies)
        """
        self.apply_land_mask = apply_land_mask
        self.ocean_only = ocean_only
        self.mask_resolution = mask_resolution
        self.land_mask = None
        self.eof_results = {}
        
    def create_land_mask(self, data: xr.DataArray, resolution: str = 'c') -> xr.DataArray:
        """
        Create land/ocean mask for the data using global-land-mask.
        
        Parameters:
        -----------
        data : xr.DataArray
            Input data array with lat/lon coordinates
        resolution : str
            Resolution of the mask ('c' for coarse ~50km, 'l' for low ~10km, 
            'i' for intermediate ~2km, 'h' for high ~400m, 'f' for full ~100m)
            
        Returns:
        --------
        xr.DataArray
            Boolean mask where True indicates ocean (if ocean_only=True) or land (if ocean_only=False)
        """
        # Get coordinate names (handle different naming conventions)
        lat_name = None
        lon_name = None
        
        for coord in data.coords:
            if coord.lower() in ['lat', 'latitude', 'y']:
                lat_name = coord
            elif coord.lower() in ['lon', 'longitude', 'x']:
                lon_name = coord
                
        if lat_name is None or lon_name is None:
            raise ValueError("Cannot find latitude/longitude coordinates in data")
        
        # Get coordinate values
        lats = data[lat_name].values
        lons = data[lon_name].values
        
        # Create meshgrid for vectorized operation
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Use global-land-mask to determine land/ocean
        # globe.is_land returns True for land, False for ocean
        land_mask_2d = globe.is_land(lat_grid, lon_grid)
        
        # Create xarray DataArray for the mask
        mask_coords = {lat_name: lats, lon_name: lons}
        land_mask_da = xr.DataArray(
            land_mask_2d, 
            coords=mask_coords, 
            dims=[lat_name, lon_name]
        )
        
        # Apply ocean_only logic
        if self.ocean_only:
            # True for ocean (where land_mask is False)
            self.land_mask = ~land_mask_da
        else:
            # True for land (where land_mask is True)
            self.land_mask = land_mask_da
            
        return self.land_mask
    
    def extract_low_harmonics(self, data: xr.DataArray, n_harm: int = 3, dim: str = 'dayofyear') -> xr.DataArray:
        """
        Extract low-order harmonics from daily climatology using FFT.
        
        Parameters:
        -----------
        data : xr.DataArray
            Input climatology data
        n_harm : int
            Number of harmonics to retain
        dim : str
            Dimension along which to apply FFT
            
        Returns:
        --------
        xr.DataArray
            Smoothed climatology with retained harmonics
        """
        # Fourier transform
        z_fft = np.fft.rfft(data, axis=data.get_axis_num(dim))
        
        # Keep only low-order harmonics
        z_fft_filtered = z_fft.copy()
        if n_harm < z_fft.shape[data.get_axis_num(dim)]:
            z_fft_filtered[n_harm,:,:] *= 0.5  # Reduce n_harm order amplitude by half
            z_fft_filtered[(n_harm+1):,:,:] = 0  # Zero out higher harmonics
        
        # Inverse FFT
        smoothed_data = np.fft.irfft(z_fft_filtered, n=data.sizes[dim], 
                                   axis=data.get_axis_num(dim)).real
        
        # Maintain xarray format
        coords = {k: v for k, v in data.coords.items()}
        dims = data.dims
        attrs = {
            "smoothing": f"FFT: {n_harm} harmonics retained",
            "information": "Smoothed daily climatological averages",
            "units": data.attrs.get("units", ""),
            "long_name": f"Daily Climatology: {n_harm} harmonics retained",
        }
        
        return xr.DataArray(smoothed_data, coords=coords, dims=dims, attrs=attrs)
    
    def compute_eof(self, data: np.ndarray) -> Dict:
        """
        Compute EOF analysis using SVD.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data matrix (variables x observations)
            
        Returns:
        --------
        Dict
            Dictionary containing EOF results
        """
        # Handle missing values
        valid_mask = ~np.any(np.isnan(data), axis=0)
        data_valid = data[:, valid_mask]
        
        # Perform SVD
        u, s, v = np.linalg.svd(data_valid, full_matrices=False)
        
        # Calculate EOF patterns and PC time series
        eof_patterns = u.T  # EOF patterns
        pc_series = np.dot(eof_patterns, data_valid)  # Principal components
        
        # Calculate eigenvalues and explained variance
        nt = data_valid.shape[1]
        eigenvalues = s**2 / nt
        explained_variance = eigenvalues / np.sum(eigenvalues) * 100
        
        # Estimate degrees of freedom for North test
        L = 1  # one-lag autocorrelation
        phi_L, phi_0, dof = self._estimate_dof(data_valid, L)
        eigenvalue_errors = explained_variance * np.sqrt(2 / dof)
        
        return {
            'eof_patterns': eof_patterns,
            'pc_series': pc_series,
            'eigenvalues': eigenvalues,
            'explained_variance': explained_variance,
            'eigenvalue_errors': eigenvalue_errors,
            'degrees_of_freedom': dof,
            'valid_mask': valid_mask,
            'phi_0': phi_0,
            'phi_L': phi_L
        }
    
    def _estimate_dof(self, data: np.ndarray, L: int = 1) -> Tuple[float, float, float]:
        """
        Estimate degrees of freedom for North test.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        L : int
            Lag for autocorrelation calculation
            
        Returns:
        --------
        Tuple[float, float, float]
            phi_L, phi_0, degrees_of_freedom
        """
        nt = data.shape[1]
        
        # Calculate lag-L autocorrelation
        B = 0
        for k in range(L, nt - L):
            B += np.sum(data[:, k] * data[:, k + L])
        
        phi_L = B / (nt - 2 * L)
        phi_0 = np.sum(data**2) / nt
        r_L = phi_L / phi_0
        
        # Degrees of freedom
        dof = (1 - r_L**2) / (1 + r_L**2) * nt
        
        return phi_L, phi_0, dof
    
    def adjust_eof_signs(self, eof_results: Dict, pressure_levels: np.ndarray) -> Dict:
        """
        Adjust EOF signs based on physical conventions.
        
        Parameters:
        -----------
        eof_results : Dict
            EOF analysis results
        pressure_levels : np.ndarray
            Pressure levels for sign convention
            
        Returns:
        --------
        Dict
            Adjusted EOF results
        """
        eof_patterns = eof_results['eof_patterns'].copy()
        pc_series = eof_results['pc_series'].copy()
        
        # Adjust EOF1 sign (convention: positive mean)
        if np.mean(eof_patterns[0, :]) < 0:
            eof_patterns[0, :] *= -1
            pc_series[0, :] *= -1
            print('EOF1 sign adjusted')
        
        # Adjust EOF2 sign (convention: negative in upper levels)
        if len(pressure_levels) > 1:
            ilev_half = len(pressure_levels) // 2
            if np.mean(eof_patterns[1, :ilev_half]) < 0:
                # Keep as is
                print('EOF2 sign maintained')
            else:
                eof_patterns[1, :] *= -1
                pc_series[1, :] *= -1
                print('EOF2 sign adjusted')
        
        eof_results['eof_patterns'] = eof_patterns
        eof_results['pc_series'] = pc_series
        
        return eof_results
    
    def analyze_data(self, filepath: str, variable: str = 'w', 
                    time_slice: slice = slice('1979', '2021'),
                    lat_slice: slice = slice(-15, 15),
                    level_slice: slice = slice(1000, 100),
                    n_harmonics: int = 3) -> Dict:
        """
        Complete EOF analysis pipeline.
        
        Parameters:
        -----------
        filepath : str
            Path to the NetCDF file
        variable : str
            Variable name to analyze
        time_slice : slice
            Time period to analyze
        lat_slice : slice
            Latitude range
        level_slice : slice
            Pressure level range
        n_harmonics : int
            Number of harmonics to retain in climatology
            
        Returns:
        --------
        Dict
            Complete analysis results
        """
        print("Loading data...")
        ds = xr.open_dataset(filepath).sortby('latitude').sortby('level')
        
        # Select data
        data = getattr(ds, variable).sel(
            latitude=lat_slice, 
            time=time_slice,
            level=level_slice
        )
        
        print(f"Original data shape: {data.shape}")
        
        # Apply land mask if requested
        if self.apply_land_mask:
            print("Applying land/ocean mask using global-land-mask...")
            mask = self.create_land_mask(data.isel(time=0, level=0), 
                                       resolution=self.mask_resolution)
            
            # Expand mask to all dimensions
            mask_expanded = mask.broadcast_like(data)
            data = data.where(mask_expanded)
            
            mask_type = "Ocean" if self.ocean_only else "Land"
            print(f"Data shape after {mask_type.lower()} masking: {data.shape}")
            valid_points = (~np.isnan(data.isel(time=0, level=0))).sum().values
            print(f"Valid {mask_type.lower()} points: {valid_points}")
            total_points = data.isel(time=0, level=0).size
            print(f"Masking efficiency: {valid_points/total_points*100:.1f}% points retained")
        
        # Calculate climatology and anomalies
        print("Calculating climatology and anomalies...")
        clim = data.groupby('time.dayofyear').mean(dim='time')
        clim_smoothed = self.extract_low_harmonics(clim, n_harm=n_harmonics)
        anomalies = data.groupby('time.dayofyear') - clim_smoothed
        
        # Reshape for EOF analysis
        print("Reshaping data for EOF analysis...")
        nt, nlev, nlat, nlon = anomalies.shape
        data_reshaped = anomalies.values.transpose(1, 0, 2, 3).reshape(
            nlev, nt * nlat * nlon, order='F')
        
        # Perform EOF analysis
        print("Performing EOF analysis...")
        eof_results = self.compute_eof(data_reshaped)
        
        
        # Store results
        self.eof_results = {
            **eof_results,
            'pressure_levels': data.level.values,
            'original_data': data,
            'anomalies': anomalies,
            'climatology': clim_smoothed,
            'n_harmonics': n_harmonics,
            'mask_applied': self.apply_land_mask,
            'ocean_only': self.ocean_only if self.apply_land_mask else None
        }
        
        print("EOF analysis completed!")
        return self.eof_results
    
    def plot_eof_profiles(self, n_modes: int = 4, figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot EOF vertical profiles.
        
        Parameters:
        -----------
        n_modes : int
            Number of EOF modes to plot
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.eof_results:
            raise ValueError("No EOF results available. Run analyze_data first.")
        
        # Setup
        eof_patterns = self.eof_results['eof_patterns']
        explained_var = self.eof_results['explained_variance']
        pressure_levels = self.eof_results['pressure_levels']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 14})
        
        # Colors and labels
        colors = ['#2E8B57', '#FF8C00', '#4169E1', '#DC143C']  # SeaGreen, DarkOrange, RoyalBlue, Crimson
        
        # Plot EOF profiles
        for i in range(min(n_modes, len(explained_var))):
            label = f'EOF{i+1} ({explained_var[i]:.1f}%)'
            ax.plot(eof_patterns[i, :], pressure_levels, 
                   label=label, color=colors[i], linewidth=3, marker='o', markersize=4)
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_ylim(pressure_levels.max(), pressure_levels.min())  # Invert y-axis
        ax.set_xlim(-0.6, 0.6)
        ax.set_yscale("log")
        
        # Set pressure level ticks
        major_ticks = np.array([1000, 850, 700,600, 500,400, 300, 200, 100])
        valid_ticks = major_ticks[
            (major_ticks >= pressure_levels.min()) & 
            (major_ticks <= pressure_levels.max())
        ]
        ax.set_yticks(valid_ticks)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        
        # Labels and styling
        ax.set_xlabel('Normalized EOF Amplitude', fontsize=16, fontweight='bold')
        ax.set_ylabel('Pressure (hPa)', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Tick formatting
        ax.tick_params(which='both', direction='inout', length=6, width=1.2)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Title
        mask_info = ""
        if self.eof_results['mask_applied']:
            mask_info = f" ({'Ocean Only' if self.eof_results['ocean_only'] else 'Land Only'})"
        
        ax.set_title(f'EOF Vertical Profiles - Tropical Vertical Velocity{mask_info}', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Add info box
        info_text = f"Harmonics: {self.eof_results['n_harmonics']}\nTotal Variance: {np.sum(explained_var[:n_modes]):.1f}%"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_pc_timeseries(self, n_modes: int = 4, figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot principal component time series.
        
        Parameters:
        -----------
        n_modes : int
            Number of PC modes to plot
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.eof_results:
            raise ValueError("No EOF results available. Run analyze_data first.")
        
        pc_series = self.eof_results['pc_series']
        explained_var = self.eof_results['explained_variance']
        
        # Create time axis (assuming monthly data)
        original_data = self.eof_results['original_data']
        time_axis = original_data.time
        
        fig, axes = plt.subplots(n_modes, 1, figsize=figsize, sharex=True)
        if n_modes == 1:
            axes = [axes]
        
        colors = ['#2E8B57', '#FF8C00', '#4169E1', '#DC143C']
        
        for i in range(min(n_modes, len(explained_var))):
            # Reshape PC to match time dimension
            pc_reshaped = pc_series[i, :].reshape(original_data.shape[1:])
            pc_mean = np.nanmean(pc_reshaped, axis=(1, 2))  # Average over lat/lon
            
            axes[i].plot(time_axis, pc_mean, color=colors[i], linewidth=2)
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[i].set_ylabel(f'PC{i+1}\n({explained_var[i]:.1f}%)', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        axes[-1].set_xlabel('Time', fontsize=14, fontweight='bold')
        
        mask_info = ""
        if self.eof_results['mask_applied']:
            mask_info = f" ({'Ocean Only' if self.eof_results['ocean_only'] else 'Land Only'})"
            
        fig.suptitle(f'Principal Component Time Series{mask_info}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


# Example usage and main execution
def main():
    """
    Example usage of the EOFAnalyzer class with global-land-mask.
    """
    # File path (adjust as needed)
    file_path = "/Datadisk/ERA5/monthly1.0x1.0/1950_2023/vvel.monthly.1950_2023.nc"
    
    print("="*60)
    print("ENHANCED EOF ANALYSIS WITH GLOBAL-LAND-MASK")
    print("="*60)
    
    # Example 1: Ocean-only analysis
    print("\n1. OCEAN-ONLY ANALYSIS")
    print("-" * 30)
    analyzer_ocean = EOFAnalyzer(
        apply_land_mask=True, 
        ocean_only=True, 
        mask_resolution='c'  # Coarse resolution for global studies
    )
    
    try:
        results_ocean = analyzer_ocean.analyze_data(
            filepath=file_path,
            variable='w',
            time_slice=slice('1979', '2021'),
            lat_slice=slice(-15, 15),
            level_slice=slice(100, 1000),
            n_harmonics=3
        )
        
        # Create plots for ocean analysis
        fig1_ocean = analyzer_ocean.plot_eof_profiles(
            n_modes=4, 
            save_path='eof_profiles_ocean_global_mask.png'
        )
        
        # Example 2: Land-only analysis  
        print("\n2. LAND-ONLY ANALYSIS")
        print("-" * 30)
        analyzer_land = EOFAnalyzer(
            apply_land_mask=True, 
            ocean_only=False, 
            mask_resolution='c'
        )
        
        results_land = analyzer_land.analyze_data(
            filepath=file_path,
            variable='w',
            time_slice=slice('1979', '2021'),
            lat_slice=slice(-15, 15),
            level_slice=slice(100, 1000),
            n_harmonics=3
        )
        
        # Create plots for land analysis
        fig1_land = analyzer_land.plot_eof_profiles(
            n_modes=4, 
            save_path='eof_profiles_land_global_mask.png'
        )
        
        # Example 3: Global analysis (no masking)
        print("\n3. GLOBAL ANALYSIS (NO MASKING)")
        print("-" * 35)
        analyzer_global = EOFAnalyzer(apply_land_mask=False)
        
        results_global = analyzer_global.analyze_data(
            filepath=file_path,
            variable='w',
            time_slice=slice('1979', '2021'),
            lat_slice=slice(-15, 15),
            level_slice=slice(100, 1000),
            n_harmonics=3
        )
        
        fig1_global = analyzer_global.plot_eof_profiles(
            n_modes=4, 
            save_path='eof_profiles_global.png'
        )
        
        plt.show()
        
        # Comparative summary
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS SUMMARY")
        print("="*70)
        
        analyses = [
            ("Ocean Only", results_ocean, analyzer_ocean),
            ("Land Only", results_land, analyzer_land), 
            ("Global", results_global, analyzer_global)
        ]
        
        for name, results, analyzer in analyses:
            print(f"\n{name} Analysis:")
            print(f"  Masking: {'Yes' if analyzer.apply_land_mask else 'No'}")
            if analyzer.apply_land_mask:
                domain = 'Ocean' if analyzer.ocean_only else 'Land'
                print(f"  Domain: {domain} only")
                print(f"  Mask resolution: {analyzer.mask_resolution}")
            print(f"  Harmonics: {results['n_harmonics']}")
            print(f"  EOF1 variance: {results['explained_variance'][0]:.1f}%")
            print(f"  EOF2 variance: {results['explained_variance'][1]:.1f}%")
            print(f"  First 4 modes: {np.sum(results['explained_variance'][:4]):.1f}%")
        
        print("\n" + "="*70)
        print("GLOBAL-LAND-MASK FEATURES:")
        print("- Fast and accurate land/ocean identification")
        print("- Multiple resolution options available") 
        print("- No external data files required")
        print("- Optimized for global climate studies")
        print("="*70)
        
    except FileNotFoundError:
        print(f"\nERROR: File not found: {file_path}")
        print("Please adjust the file path in the main() function.")
        print("\nFor testing, you can create synthetic data:")
        print("```python")
        print("# Create test data")
        print("analyzer = EOFAnalyzer(apply_land_mask=True, ocean_only=True)")
        print("# ... continue with synthetic data ...")
        print("```")
    except ImportError as e:
        print(f"\nERROR: Missing required package: {e}")
        print("Please install global-land-mask:")
        print("pip install global-land-mask")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        print("Please check your data file and parameters.")


if __name__ == "__main__":
    main()
