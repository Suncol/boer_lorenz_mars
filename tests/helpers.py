from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

from mars_exact_lec.common.grid_weights import longitude_weights
from mars_exact_lec.constants_mars import MARS


def make_coords(grid: str = "regular", *, ntime: int = 2, time_dtype: str = "numeric"):
    if ntime < 1:
        raise ValueError("ntime must be at least 1")

    if time_dtype == "numeric":
        time_values = np.arange(float(ntime), dtype=float)
    elif time_dtype == "datetime":
        time_values = np.arange(ntime).astype("timedelta64[h]") + np.datetime64("2001-01-01T00:00:00")
    else:
        raise ValueError("time_dtype must be 'numeric' or 'datetime'")

    time = xr.DataArray(
        time_values,
        dims=("time",),
        coords={"time": time_values},
        name="time",
        attrs={"axis": "T", **({"units": "hours"} if time_dtype == "numeric" else {})},
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
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
        name=name,
        attrs={"units": "Pa", "standard_name": "pressure"},
    )


def surface_pressure(time, latitude, longitude, values):
    values = np.asarray(values, dtype=float)
    shape = (time.size, latitude.size, longitude.size)
    if values.ndim == 0:
        values = np.full(shape, float(values))
    elif values.ndim == 2:
        values = np.broadcast_to(values[None, :, :], shape)
    elif values.ndim != 3:
        raise ValueError("Surface pressure values must be scalar, 2D, or 3D.")
    elif values.shape != shape:
        values = np.broadcast_to(values, shape)

    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        name="ps",
        attrs={"units": "Pa", "standard_name": "surface_pressure"},
    )


def surface_geopotential(time, latitude, longitude, values, name: str = "phis"):
    values = np.asarray(values, dtype=float)
    shape = (time.size, latitude.size, longitude.size)
    if values.ndim == 0:
        values = np.full(shape, float(values))
    elif values.ndim == 2:
        values = np.broadcast_to(values[None, :, :], shape)
    elif values.ndim != 3:
        raise ValueError("Surface geopotential values must be scalar, 2D, or 3D.")
    elif values.shape != shape:
        values = np.broadcast_to(values, shape)

    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        name=name,
        attrs={"units": "m2 s-2", "standard_name": "surface_geopotential"},
    )


def surface_zonal_field(time, latitude, values, name: str = "surface_zonal"):
    values = np.asarray(values, dtype=float)
    shape = (time.size, latitude.size)
    if values.ndim == 0:
        values = np.full(shape, float(values))
    elif values.ndim == 1:
        values = np.broadcast_to(values[None, :], shape)
    elif values.ndim != 2:
        raise ValueError("Surface-zonal values must be scalar, 1D, or 2D.")
    elif values.shape != shape:
        values = np.broadcast_to(values, shape)

    return xr.DataArray(
        values,
        dims=("time", "latitude"),
        coords={"time": time, "latitude": latitude},
        name=name,
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
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
        name=name,
        attrs={"units": units},
    )


def temperature_from_theta_values(
    time,
    level,
    latitude,
    longitude,
    theta_values,
    name: str = "temperature",
):
    theta_values = np.asarray(theta_values, dtype=float)
    shape = (time.size, level.size, latitude.size, longitude.size)
    if theta_values.shape != shape:
        theta_values = np.broadcast_to(theta_values, shape)
    pressure = pressure_field(time, level, latitude, longitude)
    temperature_values = theta_values * (pressure.values / MARS.p00) ** MARS.kappa
    return full_field(
        time,
        level,
        latitude,
        longitude,
        temperature_values,
        name=name,
        units="K",
    )


def zonal_field(time, level, latitude, values, name: str):
    values = np.asarray(values, dtype=float)
    shape = (time.size, level.size, latitude.size)
    if values.shape != shape:
        values = np.broadcast_to(values, shape)
    return xr.DataArray(
        values,
        dims=("time", "level", "latitude"),
        coords={"time": time, "level": level, "latitude": latitude},
        name=name,
    )


def surface_zonal_mean(field):
    weights = longitude_weights(field.coords["longitude"], normalize=True)
    return field.weighted(weights).sum(dim="longitude")


def broadcast_surface_zonal(field_zonal, longitude, name: str | None = None):
    field = field_zonal.expand_dims(longitude=longitude).transpose("time", "latitude", "longitude")
    if name is not None:
        field.name = name
    return field


def finite_reference_profile(solution, *, time_index: int = 0):
    theta_reference = np.asarray(solution.theta_reference.isel(time=time_index).values, dtype=float)
    pi_reference = np.asarray(solution.pi_reference.isel(time=time_index).values, dtype=float)
    mass_reference = np.asarray(solution.mass_reference.isel(time=time_index).values, dtype=float)
    interface_pressure = np.asarray(solution.reference_interface_pressure.isel(time=time_index).values, dtype=float)
    interface_geopotential = np.asarray(solution.reference_interface_geopotential.isel(time=time_index).values, dtype=float)

    valid_samples = (
        np.isfinite(theta_reference)
        & np.isfinite(pi_reference)
        & np.isfinite(mass_reference)
    )
    valid_interfaces = np.isfinite(interface_pressure) & np.isfinite(interface_geopotential)

    return {
        "theta_reference": theta_reference[valid_samples],
        "pi_reference": pi_reference[valid_samples],
        "mass_reference": mass_reference[valid_samples],
        "reference_interface_pressure": interface_pressure[valid_interfaces],
        "reference_interface_geopotential": interface_geopotential[valid_interfaces],
    }


def surface_mass_from_pi_s(pi_s, reference_top_pressure, cell_area, *, constants=MARS):
    return ((pi_s - reference_top_pressure) * cell_area).sum(dim=("latitude", "longitude")) / constants.g


def pressure_inside_reference_layer(phi_target, phi_bottom, p_bottom, theta_layer, *, constants=MARS):
    phi_target = np.asarray(phi_target, dtype=float)
    exner_bottom = (float(p_bottom) / constants.p00) ** constants.kappa
    exner = exner_bottom - (phi_target - float(phi_bottom)) / (constants.cp * float(theta_layer))
    exner = np.maximum(exner, 0.0)
    return constants.p00 * np.power(exner, 1.0 / constants.kappa)


def reference_layer_mass_from_interfaces(
    p_bottom,
    p_top,
    phi_bottom,
    phi_top,
    theta_layer,
    phis,
    cell_area,
    *,
    constants=MARS,
):
    phis_values = np.asarray(phis.values if hasattr(phis, "values") else phis, dtype=float)
    area_values = np.asarray(cell_area.values if hasattr(cell_area, "values") else cell_area, dtype=float)

    local_bottom = np.full_like(phis_values, float(p_top), dtype=float)
    full_mask = phis_values <= float(phi_bottom)
    if np.any(full_mask):
        local_bottom[full_mask] = float(p_bottom)

    partial_mask = (phis_values > float(phi_bottom)) & (phis_values < float(phi_top))
    if np.any(partial_mask):
        local_bottom[partial_mask] = pressure_inside_reference_layer(
            phis_values[partial_mask],
            float(phi_bottom),
            float(p_bottom),
            float(theta_layer),
            constants=constants,
        )

    return float(np.sum((local_bottom - float(p_top)) * area_values) / constants.g)


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
