from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

from mars_exact_lec.common.integrals import pressure_level_edges
from mars_exact_lec.common.grid_weights import longitude_weights
from mars_exact_lec.constants_mars import MARS


REFERENCE_CASE_LEVEL_VALUES = np.arange(900.0, 99.0, -50.0)
REFERENCE_CASE_NLAT = 8
REFERENCE_CASE_NLON = 8


def _equal_area_latitude_bounds(nlat: int) -> np.ndarray:
    sin_edges = np.linspace(1.0, -1.0, nlat + 1)
    upper = np.rad2deg(np.arcsin(np.clip(sin_edges[:-1], -1.0, 1.0)))
    lower = np.rad2deg(np.arcsin(np.clip(sin_edges[1:], -1.0, 1.0)))
    return np.column_stack([lower, upper])


def _equal_area_latitude_centers(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = _equal_area_latitude_bounds(nlat)
    sin_lower = np.sin(np.deg2rad(bounds[:, 0]))
    sin_upper = np.sin(np.deg2rad(bounds[:, 1]))
    centers = np.rad2deg(np.arcsin(np.clip(0.5 * (sin_lower + sin_upper), -1.0, 1.0)))
    return centers, bounds


def _gaussian_latitude_bounds(nlat: int) -> np.ndarray:
    _, weights = leggauss(nlat)
    north_edges = np.empty(nlat, dtype=float)
    south_edges = np.empty(nlat, dtype=float)
    edge = 1.0
    for idx, weight in enumerate(weights[::-1]):
        north_edges[idx] = edge
        edge = edge - weight
        south_edges[idx] = edge

    upper = np.rad2deg(np.arcsin(np.clip(north_edges, -1.0, 1.0)))
    lower = np.rad2deg(np.arcsin(np.clip(south_edges, -1.0, 1.0)))
    return np.column_stack([lower, upper])


def _longitude_centers_and_bounds(nlon: int) -> tuple[np.ndarray, np.ndarray]:
    width = 360.0 / float(nlon)
    centers = np.arange(float(nlon), dtype=float) * width
    bounds = np.column_stack([centers - 0.5 * width, centers + 0.5 * width])
    return centers, bounds


def make_coords(
    grid: str = "regular",
    *,
    ntime: int = 2,
    time_dtype: str = "numeric",
    level_values=None,
    nlat: int | None = None,
    nlon: int | None = None,
):
    if ntime < 1:
        raise ValueError("ntime must be at least 1")
    if nlat is not None and nlat < 2:
        raise ValueError("nlat must be at least 2")
    if nlon is not None and nlon < 2:
        raise ValueError("nlon must be at least 2")

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
    if level_values is None:
        level_values = np.asarray([700.0, 500.0, 300.0], dtype=float)
    else:
        level_values = np.asarray(level_values, dtype=float)
    level = xr.DataArray(
        level_values,
        dims=("level",),
        coords={"level": level_values},
        name="level",
        attrs={"units": "Pa", "axis": "Z", "standard_name": "pressure"},
    )

    if grid == "regular":
        if nlat is None:
            lat_values = np.asarray([67.5, 22.5, -22.5, -67.5], dtype=float)
        else:
            lat_values, _ = _equal_area_latitude_centers(nlat)
    elif grid == "gaussian":
        node_count = 4 if nlat is None else int(nlat)
        nodes, _ = leggauss(node_count)
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
    if nlon is None:
        lon_values = np.asarray([0.0, 90.0, 180.0, 270.0], dtype=float)
    else:
        lon_values, _ = _longitude_centers_and_bounds(int(nlon))
    longitude = xr.DataArray(
        lon_values,
        dims=("longitude",),
        coords={"longitude": lon_values},
        name="longitude",
        attrs={"units": "degrees_east", "axis": "X", "standard_name": "longitude"},
    )
    return time, level, latitude, longitude


def reference_case_coords(grid: str = "regular", *, ntime: int = 2, time_dtype: str = "numeric"):
    return make_coords(
        grid=grid,
        ntime=ntime,
        time_dtype=time_dtype,
        level_values=REFERENCE_CASE_LEVEL_VALUES,
        nlat=REFERENCE_CASE_NLAT,
        nlon=REFERENCE_CASE_NLON,
    )


def reference_case_theta_profile(level, *, base: float = 180.0, step: float = 5.0) -> np.ndarray:
    return base + step * np.arange(level.size, dtype=float)


def reference_case_theta_field_values(
    time,
    level,
    latitude,
    longitude,
    *,
    base: float,
    step: float,
    time_offsets,
    lat_amplitude: float,
    lon_amplitude: float,
) -> np.ndarray:
    time_offsets = np.asarray(time_offsets, dtype=float)
    if time_offsets.ndim == 0:
        time_offsets = np.full(time.size, float(time_offsets))
    if time_offsets.shape != (time.size,):
        raise ValueError("time_offsets must be scalar or have shape (time,).")

    base_profile = reference_case_theta_profile(level, base=base, step=step)[None, :, None, None]
    centered_latitude = np.linspace(-1.0, 1.0, latitude.size, dtype=float)[None, None, :, None]
    centered_longitude = np.linspace(-1.0, 1.0, longitude.size, dtype=float)[None, None, None, :]
    return (
        base_profile
        + time_offsets[:, None, None, None]
        + float(lat_amplitude) * centered_latitude
        + float(lon_amplitude) * centered_longitude
    )


def reference_case_surface_pressure_values(
    latitude,
    longitude,
    *,
    base: float = 910.0,
    lon_drop: float = 420.0,
    lat_drop: float = 80.0,
) -> np.ndarray:
    lat_fraction = np.linspace(0.0, 1.0, latitude.size, dtype=float)[:, None]
    lon_fraction = np.linspace(0.0, 1.0, longitude.size, dtype=float)[None, :]
    return base - lon_drop * lon_fraction - lat_drop * lat_fraction


def reference_case_surface_geopotential_values(
    latitude,
    longitude,
    *,
    base: float = 0.0,
    lat_range: float = 360.0,
    lon_range: float = 540.0,
) -> np.ndarray:
    lat_fraction = np.linspace(0.0, 1.0, latitude.size, dtype=float)[:, None]
    lon_fraction = np.linspace(0.0, 1.0, longitude.size, dtype=float)[None, :]
    return base + lat_range * lat_fraction + lon_range * lon_fraction


def reference_case_terrain_anomaly_values(latitude, longitude, amplitude: float) -> np.ndarray:
    lat_centered = np.linspace(-1.0, 1.0, latitude.size, dtype=float)[:, None]
    lon_centered = np.linspace(-1.0, 1.0, longitude.size, dtype=float)[None, :]
    pattern = lon_centered + 0.35 * lat_centered
    pattern = pattern / np.max(np.abs(pattern))
    return amplitude * pattern


def reference_case_surface_time_series(surface_2d, time_offsets) -> np.ndarray:
    surface_2d = np.asarray(surface_2d, dtype=float)
    if surface_2d.ndim != 2:
        raise ValueError("surface_2d must be two-dimensional.")

    time_offsets = np.asarray(time_offsets, dtype=float)
    if time_offsets.ndim == 0:
        time_offsets = np.asarray([float(time_offsets)], dtype=float)
    if time_offsets.ndim != 1:
        raise ValueError("time_offsets must be scalar or one-dimensional.")
    return surface_2d[None, :, :] + time_offsets[:, None, None]


def reference_case_level_bounds(level) -> xr.DataArray:
    edges = pressure_level_edges(level)
    return xr.DataArray(
        np.column_stack([edges.values[:-1], edges.values[1:]]),
        dims=("level", "bounds"),
        coords={"level": level.values, "bounds": [0, 1]},
        name="level_bounds",
        attrs={"units": "Pa"},
    )


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
