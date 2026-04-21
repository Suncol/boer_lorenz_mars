project('seba', 'fortran', 'c',
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>=1.1.0',
  default_options: [
    'warning_level=1'
  ]
)

# --- Python and NumPy setup ---
py = import('python').find_installation(pure: false)
python_bindir = run_command(py, '-c', 'import os, sys; print(os.path.realpath(os.path.dirname(sys.executable)))').stdout().strip()
stable_path = run_command(py, '-c', '''
import os
import sys

build_bindir = os.path.realpath(os.path.dirname(sys.executable))
entries = []

venv = os.environ.get('VIRTUAL_ENV')
if venv:
    entries.append(os.path.join(venv, 'bin'))

for entry in os.environ.get('PATH', '').split(os.pathsep):
    if not entry:
        continue
    if os.path.realpath(entry) == build_bindir:
        continue
    if entry not in entries:
        entries.append(entry)

print(os.pathsep.join(entries))
''').stdout().strip()
numpy_dep = dependency('numpy', required: true)
fc = find_program('gfortran', 'gfortran-15', 'gfortran-14', 'gfortran-13', 'gfortran-12', required: true)

# --- OpenMP support ---
option_openmp = get_option('openmp')
openmp_dep = dependency('openmp', required: option_openmp, language: 'fortran')

# --- Fortran extension: numeric_tools ---
mod_name = 'numeric_tools'
fortran_source = 'src/seba/fortran_libs/numeric_tools.f90'

# infer extension suffix
mod_suffix = run_command(py, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))').stdout().strip()

numeric_tools_command = ['env']
numeric_tools_command += ['PATH=' + stable_path]
numeric_tools_command += ['FC=' + fc.full_path()]
if option_openmp
  openmp_cc = find_program('gcc-15', 'gcc-14', 'gcc-13', 'gcc-12', 'gcc', required: true)
  numeric_tools_command += [
    'CC=' + openmp_cc.full_path(),
    'CFLAGS=-fopenmp',
    'FFLAGS=-fopenmp',
    'LDFLAGS=-fopenmp -lgomp',
  ]
endif
numeric_tools_command += ['f2py', '-c', '-m', mod_name, '@INPUT@', '--quiet']

numeric_tools = custom_target(
  mod_name,
  input: fortran_source,
  output: mod_name + mod_suffix,
  command: numeric_tools_command,
  install: true,
  install_dir: join_paths(py.get_install_dir(), 'seba')
)


# --- Installing Python sources ---
py.install_sources([
  'src/seba/__init__.py',
  'src/seba/constants.py',
  'src/seba/io_tools.py',
  'src/seba/kinematics.py',
  'src/seba/seba.py',
  'src/seba/spectral_analysis.py',
  'src/seba/spherical_harmonics.py',
  'src/seba/thermodynamics.py',
  'src/seba/tools.py',
  'src/seba/visualization.py',
  'src/seba/inter_scale_transfers.py'
], subdir: 'seba')

py.install_sources([
  'src/seba/cm_data/__init__.py'
], subdir: 'seba/cm_data')

# --- Installing data files ---
cm_data_files = files(
  'src/seba/cm_data/bcw.cm',
  'src/seba/cm_data/cet_d02.cm',
  'src/seba/cm_data/cet_d10.cm',
  'src/seba/cm_data/cet_d13.cm'
)

install_data(cm_data_files, install_dir: join_paths(py.get_install_dir(), 'seba', 'cm_data'))

# --- Summary ---
message('✔ Configuration for SEBA complete.')
message('Python version:      ' + py.version())
message('Install location:    ' + py.get_install_dir())
message('NumPy version:       ' + numpy_dep.version())
message('OpenMP enabled:      ' + openmp_dep.found().to_string())
