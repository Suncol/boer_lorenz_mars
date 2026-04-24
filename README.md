### Description
    -----------
    A collection of tools to compute the Spectral Energy Budget of a dry hydrostatic
    Atmosphere (SEBA). This package is developed for application to global numerical
    simulations of General Circulation Models (GCMs). SEBA is implemented based on the
    formalism developed by Augier and Lindborg (2013) and includes the Helmholtz decomposition
    into the rotational and divergent kinetic energy contributions to the nonlinear energy
    fluxes introduced by Li et al. (2023). The Spherical Harmonic Transforms are carried out
    with the high-performance SHTns C library. The analysis supports data sampled on a
    regular (equally spaced in longitude and latitude) or Gaussian (equally spaced in
    longitude, latitudes located at roots of ordinary Legendre polynomial of degree nlat)
    horizontal grids. The vertical grid can be arbitrary; if data is not sampled on
    pressure levels, it is interpolated to isobaric levels before the analysis.

    References:
    -----------
    Augier, P., and E. Lindborg (2013), A new formulation of the spectral energy budget
    of the atmosphere, with application to two high-resolution general circulation models,
    J. Atmos. Sci., 70, 2293–2308, https://doi.org/10.1175/JAS-D-12-0281.1.

    Li, Z., J. Peng, and L. Zhang, 2023: Spectral Budget of Rotational and Divergent Kinetic
    Energy in Global Analyses.  J. Atmos. Sci., 80, 813–831,
    https://doi.org/10.1175/JAS-D-21-0332.1.

    Schaeffer, N. (2013). Efficient spherical harmonic transforms aimed at pseudospectral
    numerical simulations, Geochem. Geophys. Geosyst., 14, 751– 758,
    https://doi.org/10.1002/ggge.20071.

## Examples

This example demonstrates how to compute and visualize **spectral energy diagnostics and nonlinear energy transfers** using **SEBA** with reanalysis or model data.

### 1. Load atmospheric data

```python
import xarray as xr
from seba.seba import EnergyBudget

# Define dataset path and model
model = "ERA5"
resolution = "025deg"
data_path = "/path/to/simulations/"
date_time = "20200128"

# Load atmospheric 3D fields and surface pressure
file_name = "/path/to/simulations/data.nc"

# Define external surface variables. Optional: read from input model data if available, override if specified
sfc_pres = xr.open_dataset("/path/to/simulations/surface_data.nc")["sfc_pressure"]
```

---

### 2. Create an energy budget object

```python
# Create budget object from data path or directly from loaded xarray.Dataset
budget = EnergyBudget(file_name, truncation=511, ps=sfc_pres)
```

The `truncation` parameter sets the **spectral resolution** of the spherical harmonic transform used internally.
Default `None`: truncation is based on Gaussian grid resolution, e.g., n512

---

### 3. Compute energy diagnostics

```python
dataset_energy = budget.energy_diagnostics()
```

This computes **Horizontal kinetic energy (HKE)**, **Vertical kinetic energy (VKE)**, **Rotational/Divergent kinetic energy ([R/D]KE)**, and **Available potential energy (APE)** spectra for each time and pressure level.

---

### 4. Visualize energy spectra

```python
layers = {"Troposphere": [250e2, 450e2], "Stratosphere": [50e2, 250e2]}
dataset_energy.visualize_energy(model=model, layers=layers, fig_name=f"figures/energy_spectra.pdf")
```

This produces a subplot per layer of the energy spectra. Variables can be specified via the `variables` argument, e.g., `variables=['hke', 'vke', 'ape']`.

---

### 5. Compute and visualize nonlinear energy fluxes

```python
dataset_fluxes = budget.cumulative_energy_fluxes()

layers = {
    "Lower troposphere": [450e2, 850e2],
    "Free troposphere": [250e2, 450e2],
    "Stratosphere": [20e2, 250e2],
}
y_limits = {
    "Stratosphere": [-0.6, 1.0],
    "Free troposphere": [-0.6, 1.0],
    "Lower troposphere": [-0.9, 1.5],
}

dataset_fluxes.visualize_fluxes(
    model=model,
    variables=["pi_hke+pi_ape", "pi_hke", "pi_ape"],
    layers=layers, y_limits=y_limits,
    fig_name=f"figures/energy_fluxes.pdf"
)
```

This generates layer-wise flux plots showing **Cumulative nonlinear spectral energy transfers**.

---

### 6. Cross-section visualization

```python
dataset_fluxes.visualize_sections(
    variables=["cad", "vfd_dke"],
    y_limits=[1000., 98.],
    fig_name=f"figures/fluxes_section.pdf"
)
```

Plots **vertical cross-sections** of specific HKE budget terms (e.g., conversion from APE to HKE, vertical flux divergence) as a function of pressure and horizontal wavenumber.

## Live SEBA validation

The live Mars/SEBA cross-validation path in `tests/test_seba_validation.py` runs against an
installed SEBA runtime, not just the repository source tree. This matters because
`seba.seba` imports both the compiled `numeric_tools` extension and `shtns`.

On macOS/Homebrew with the repository-managed uv environment:

```bash
source .venv/bin/activate
brew install fftw pkg-config
export PKG_CONFIG_PATH="$(brew --prefix fftw)/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CPPFLAGS="-I$(brew --prefix fftw)/include ${CPPFLAGS}"
export LDFLAGS="-L$(brew --prefix fftw)/lib ${LDFLAGS}"
uv pip install --no-binary shtns shtns
export MESON="$PWD/.venv/bin/meson"
export NINJA="$PWD/.venv/bin/ninja"
uv pip install -e .
```

This project keeps `mars_exact_lec` source-imported during local test runs; the installed
editable runtime is only required for `seba` itself. Pinning `MESON` and `NINJA` to
`.venv/bin` keeps the editable loader from capturing temporary uv build-environment tool paths.

If you prefer not to export those tool paths, the supported fallback is:

```bash
uv pip install --no-build-isolation -e .
```

If your toolchain cannot provide OpenMP-compatible flags and libraries, disable OpenMP for
SEBA's own `numeric_tools` build and reinstall:

```bash
uv pip install --no-build-isolation -Csetup-args=-Dopenmp=false -e .
```

Once the environment is ready, run the live validation path with:

```bash
source .venv/bin/activate
python -m pytest -m live_seba -rs tests/test_seba_validation.py
```

The validation helper `seba_energy_components_per_level()` exposes SEBA spectral
`rke`, `dke`, `hke`, `vke`, and `ape` components after summing over horizontal
wavenumber. These are cross-check fields from the SEBA branch; they are not Boer
exact four-reservoir diagnostics and should not be interpreted as equivalent to
the Mars exact LEC APE/KE reservoirs.

## Mars exact LEC notes

The `mars_exact_lec` exact/topography-aware branch now distinguishes two different objects:

- `Theta` remains the sharp cell-center above-ground mask used for exact-Boer representative fields,
  validity masks, and derivative-domain masking.
- `TopographyAwareMeasure` provides the shared finite-volume partial-cell measure used by the
  exact branch for rigorous diagnostics over uneven topography.

Public exact-Boer APIs call the mask argument `theta_mask` to avoid confusing the above-ground
coverage field with physical potential temperature `theta`. Positional mask arguments remain
supported, but keyword calls should use `theta_mask=...`; deprecated `theta=...` mask keywords
are accepted only through a warning-backed compatibility path.

Exact Boer diagnostics now default to the measure-aware finite-volume path. If you pass `ps`,
the public API auto-constructs a consistent `TopographyAwareMeasure` and uses shared
partial-cell weights, `p_s_eff`, and measure-aware representative means by default. The
explicit `measure=` argument is still available as an advanced override when you want to reuse
an already-constructed measure or control `clip/raise` behavior directly.

```python
from mars_exact_lec.boer.reservoirs import kinetic_energy_zonal
from mars_exact_lec.common import build_mass_integrator

integrator = build_mass_integrator(level, latitude, longitude)
result = kinetic_energy_zonal(u, v, theta_mask, integrator, ps=ps)
```

If you prefer to build the measure yourself, the exact branch still supports that path:

```python
from mars_exact_lec.common import TopographyAwareMeasure, build_mass_integrator

integrator = build_mass_integrator(level, latitude, longitude)
measure = TopographyAwareMeasure.from_surface_pressure(
    level,
    ps,
    integrator,
    surface_pressure_policy="raise",
)
```

If an exact diagnostic does not receive either `ps` or an explicit `measure`, it now raises a
clear error instead of silently falling back to the old whole-cell mass measure. The default
`surface_pressure_policy` is `"raise"`; use `"clip"` explicitly when you want to truncate
columns that extend below the deepest model interface.

If you pass both `measure=` and `ps=`, they must come from the same raw surface-pressure field.
The exact branch treats the explicit `measure` as authoritative and uses `ps` only to validate
that provenance. This check is intentionally strict: in `clip` mode, two different raw `ps`
fields are still considered incompatible even if they happen to clip to the same `p_s_eff`.
If you want a different `surface_pressure_policy`, rebuild the `measure`; passing a new
`surface_pressure_policy=` alongside an explicit `measure` does not override the policy stored
inside that measure.

Reference-state solvers require explicit `phis` for topographic exact calculations. Omit it
only when intentionally solving a flat-surface problem, and spell that intent with
`assume_flat_surface=True`. Use the explicit solver names `Koehler1986ReferenceState` for the
fixed-isentrope Koehler (1986) path and `FiniteVolumeReferenceState` for the legacy finite-volume
path. `KoehlerReferenceState` is retained only as a deprecated, warning-backed compatibility
alias to `FiniteVolumeReferenceState`.

When `surface_pressure_policy="clip"`, the exact branch computes diagnostics on the truncated
model pressure domain rather than the full atmospheric column implied by the raw `ps`. This is
surfaced explicitly in attrs on measure-aware exact outputs:

- `surface_pressure_policy = "clip"`
- `domain = "truncated_to_model_pressure_domain"`
- `not_exact_full_atmosphere = True`

For symmetry, the `raise` path also writes domain metadata:

- `surface_pressure_policy = "raise"`
- `domain = "full_model_pressure_domain"`
- `not_exact_full_atmosphere = False`

These attrs are intentionally machine-readable. They are preserved on exact reservoir and
conversion diagnostics, reference-state outputs driven by the same measure, and downstream
per-area normalization helpers.

Among the exact conversion terms, `C_K2` is the most sensitive to topography-boundary
discretization. Its public outputs now carry additional machine-readable provenance attrs:

- `ck2_discretization`
- `ck2_geopotential_source`
- `ck2_geopotential_mode`
- `ck2_geopotential_reconstruction_allowed`
- `ck2_geopotential_reconstruction_approximate`
- `ck2_vertical_integral`
- `ck2_reconstruction`
- `ck2_bottom_pressure`
- `ck2_horizontal_boundary_correction`
- `ck2_pressure_term`
- `ck2_zonal_mean`
- `ck2_derivative_mask`

These attrs document that `C_K2` is evaluated with a cut-cell finite-volume discretization.
The implementation reconstructs `Phi*` to the layer top and effective bottom pressure,
forms a trapezoidal pressure-layer integral, applies the Leibniz lower-boundary correction
for moving or sloping cut cells, and zonally averages the resulting full-layer-normalized
terms with cell-area weights. By default, the reconstruction uses pressure-linear interpolation
from an explicit level-center `Phi*` field. If your GCM provides interface geopotential, pass
`interface_geopotential=` with dims `(time, level_edge, latitude, longitude)`; `C_K2` will
compute `Phi*` directly on the raw interface faces and use `phis` for partial-cell surface
bottoms when available. Without `phis`, partial bottoms are reconstructed pressure-linearly
from interface faces, or from interface plus center geopotential when both
`interface_geopotential` and `geopotential` are supplied.

High-accuracy `C_K2` should use native GCM `interface_geopotential` or `geopotential`.
Hydrostatic reconstruction from `temperature`, `pressure`, `ps`, and `phis` is an
approximate fallback and is disabled by default. Use `geopotential_mode="strict"` for
the default no-reconstruction path, `geopotential_mode="hydrostatic"` when you explicitly
accept hydrostatic reconstruction/filling, or `geopotential_mode="fill"` when you only
want log-pressure filling of gaps in an explicit level-center `geopotential` field.
The legacy `allow_geopotential_reconstruction=True` spelling is accepted only through a
warning-backed compatibility path. Approximate outputs mark their provenance with
`ck2_geopotential_source`, `ck2_geopotential_mode`, and
`ck2_geopotential_reconstruction_approximate=True`.

Koehler (1986) reference-state preprocessing defaults to
`monotonic_policy="reject"`. Non-monotonic potential-temperature columns are
not silently repaired by the public solver; pass `monotonic_policy="repair"`
only when you explicitly accept the monotonic-envelope modification. Repair
diagnostics are exposed as `monotonic_violations` and `monotonic_repairs` on
the returned `ReferenceStateSolution`.

Low-level Boer diagnostics continue to return global integrals in `J` and `W`. To compare against
paper figures reported in `J m-2` or `W m-2`, normalize them explicitly with
`mars_exact_lec.common.to_per_area()` or `mars_exact_lec.common.normalize_dataset_per_area()`.
