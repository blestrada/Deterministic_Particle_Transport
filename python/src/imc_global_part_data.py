"""Global particle data."""

# Particle Parameters
Nx = -1
Nmu = -1
Nt = -1

# Number of particles added per timestep (RN)
n_input = -1
n_max = -1
n_census = -1

# Global list of particle properties
n_particles = 0
n_scattered_particles = 0
max_array_size = 14_000_000
particle_prop = []
scattered_particles = []

# Census grid
census_grid = []

# Some parameters
# mode : random numbers or no random numbers (rn or nrn)
# scattering: using analog scattering or implicit scattering - used only in NRN mode (analog or implicit)
problem_type = None
mode = None
scattering = None