"""Global particle data."""

# Particle Parameters
Nx = None
Ny = None
Nmu = None
N_omega = None
Nt = None
Ng = None
surface_Ny = None
surface_Nmu = None
surface_N_omega = None
surface_Nt = None

# Number of particles added per timestep (RN)
n_input = -1
n_census = -1

# Global list of particle properties
n_particles = 0
n_scattered_particles = 0
max_array_size = 6_000_000
particle_prop = []
scattered_particles = []

# Census grid 
census_particles = []

# Some parameters
# mode : random numbers or no random numbers (rn or nrn)
# scattering: using analog scattering or implicit scattering - used only in NRN mode (analog or implicit)
problem_type = None
mode = None
scattering = None