from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss


def make_coords(grid: str = "regular"):
    time = xr.DataArray(
        np.asarray([0.0, 1.0]),
        dims=("time",),
        coords={"time": [0.0, 1.0]},
        name="time",
        attrs={"axis": "T"},
    )
    level = xr.DataArray(
        np.asarray([700.0, 500.0, 300.0]),
        dims=("level",),
        coords={"level": [700.0, 500.0, 300.0]},
        name="level",
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )

    if grid == "regular":
        lat_values = np.asarray([67.5, 22.5, -22.5, -67.5])
    elif grid == "gaussian":
        nodes, _ = leggauss(4)
        lat_values = np.rad2deg(np.arcsin(nodes[::-1]))
    else:
        raise ValueError("grid must be 'regular' or 'gaussian'")

    latitude = xr.DataArray(
        lat_values,
        dims=("latitude",),
        coords={"latitude": lat_values},
        name="latitude",
        attrs={"units": "degrees_north", "axis": "Y", "standard_name": "latitude"},
    )
    longitude = xr.DataArray(
        np.asarray([0.0, 90.0, 180.0, 270.0]),
        dims=("longitude",),
        coords={"longitude": [0.0, 90.0, 180.0, 270.0]},
        name="longitude",
        attrs={"units": "degrees_east", "axis": "X", "standard_name": "longitude"},
    )
    return time, level, latitude, longitude


def pressure_field(time, level, latitude, longitude, name: str = "pressure"):
    shape = (time.size, level.size, latitude.size, longitude.size)
    values = np.broadcast_to(level.values[None, :, None, None], shape)
    return xr.DataArray(
        values.astype(float),
        dims=("time", "level", "latitude", "longitude"),
        coords={
            "time": time.values,
            "level": level.values,
            "latitude": latitude.values,
            "longitude": longitude.values,
        },
        name=name,
        attrs={"units": "Pa", "standard_name": "pressure"},
    )


def surface_pressure(time, latitude, longitude, values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full((time.size, latitude.size, longitude.size), float(values))
    elif values.ndim == 2:
        values = np.broadcast_to(values[None, :, :], (time.size, latitude.size, longitude.size))
    elif values.ndim != 3:
        raise ValueError("Surface pressure values must be scalar, 2D, or 3D.")

    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time.values,
            "latitude": latitude.values,
            "longitude": longitude.values,
        },
        name="ps",
        attrs={"units": "Pa", "standard_name": "surface_pressure"},
    )


def full_field(time, level, latitude, longitude, values, name: str, units: str = ""):
    values = np.asarray(values, dtype=float)
    shape = (time.size, level.size, latitude.size, longitude.size)
    if values.shape != shape:
        values = np.broadcast_to(values, shape)
    return xr.DataArray(
        values,
        dims=("time", "level", "latitude", "longitude"),
        coords={
            "time": time.values,
            "level": level.values,
            "latitude": latitude.values,
            "longitude": longitude.values,
        },
        name=name,
        attrs={"units": units},
    )


def zonal_field(time, level, latitude, values, name: str):
    values = np.asarray(values, dtype=float)
    shape = (time.size, level.size, latitude.size)
    if values.shape != shape:
        values = np.broadcast_to(values, shape)
    return xr.DataArray(
        values,
        dims=("time", "level", "latitude"),
        coords={"time": time.values, "level": level.values, "latitude": latitude.values},
        name=name,
    )


def seba_dataset(time, level, latitude, longitude, u, v, omega, temperature):
    pressure = pressure_field(time, level, latitude, longitude)
    return xr.Dataset(
        data_vars={
            "u_wind": u,
            "v_wind": v,
            "omega": omega,
            "temperature": temperature,
            "pressure": pressure,
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
