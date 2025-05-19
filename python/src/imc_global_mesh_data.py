"""Global mesh data."""

ncells = -1
source_cells = 1
xsize = -1.0
dx = -1.0
body_source = True
# Cell arrays

cellpos = -1.0
temp = -1.0
temp0 = -1.0
radtemp = -1.0
radtemp0 = -1.0
nrgdep = -1.0
nrgscattered = -1.0
nrgdep_from_scattering = -1.0
sigma_a = -1.0
sigma_s = -1.0
sigma_t = -1.0
fleck = -1.0
beta = -1.0

nrg_leaked = 0.0
radnrgdens = -1.0
matnrgdens = -1.0
# Nodal arrays

nodepos = -1.0

# Boundary conditions (reflecting or vacuum)
left_bc = None
right_bc = None
