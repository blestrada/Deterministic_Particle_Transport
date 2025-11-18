import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

# Get data from the gnuplot file.
# Load the data file
file_path = 'marshak_wave_imc'

# Read the data, skipping the first row if it's a header
data = pd.read_csv(file_path, sep='\s+', skiprows=1, header=None)

# Assign column names based on the data header or assume generic names
data.columns = ['x', 'Te', 'Tr', 'Tanalytic', 'Column5', 'Column6']  # Adjust based on actual data columns

# Create figure for Material temperature plot
plt.figure(figsize=(7, 6))
# Plot columns 1 and 2 as in Gnuplot
plt.plot(data['x'], data['Te'], color='red', label=r'IMC',linewidth=1)
plt.xlabel('x - cm')
plt.ylabel('Tm - keV')
plt.ylim(0.0, 1.1)


# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu2.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='blue', label=r"$N_{\mu}=2$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu4.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='green', label=r"$N_{\mu}=4$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='orange', label=r"$N_{\mu}=8$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu16.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='purple', label=r"$N_{\mu}=16$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu32.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='lime', label=r"$N_{\mu}=32$", linewidth=1, alpha=1)



#####################
#                   #
# Testing N_t       #
#                   #
#####################


# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='blue', label=r"$N_{t}=1$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt2_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='green', label=r"$N_{t}=2$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt3_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='orange', label=r"$N_{t}=3$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt4_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='purple', label=r"$N_{t}=4$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx1_nt5_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='cyan', label=r"$N_{t}=5$", linewidth=1, alpha=1)


#####################
#                   #
# Testing N_x       #
#                   #
#####################


# # Open the output file
# fname = open("MarshakWave_nx1_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='blue', label=r"$N_{x}=1$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx2_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='green', label=r"$N_{x}=2$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx3_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='orange', label=r"$N_{x}=3$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx4_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='purple', label=r"$N_{x}=4$", linewidth=1, alpha=1)

# # Open the output file
# fname = open("MarshakWave_nx5_nt1_nmu8.out", "rb")

# # Times corresponding to data in the output file
# times = [r"$t$ = 0.3 sh"]

# time_line = fname.readline().decode().strip()  # Read the time line
# xdata = pickle.load(fname)      # cellpos
# temp = pickle.load(fname)       # temperature
# #radnrgdens_dpt = pickle.load(fname) # radnrgdens

# # Create figure for the material temperature for DPT
# plt.plot(xdata, temp, '-' , color='cyan', label=r"$N_{x}=5$", linewidth=1, alpha=1)


#####################
#                   #
# Testing N_x=5, N_t=5, Nmu=8       #
#                   #
#####################
# Open the output file
fname = open("MarshakWave_nx5_nt5_nmu8.out", "rb")

# Times corresponding to data in the output file
times = [r"$t$ = 0.3 sh"]

time_line = fname.readline().decode().strip()  # Read the time line
xdata = pickle.load(fname)      # cellpos
temp = pickle.load(fname)       # temperature
#radnrgdens_dpt = pickle.load(fname) # radnrgdens

# Create figure for the material temperature for DPT
plt.plot(xdata, temp, '-' , color='blue', label=r"$N_{x}=5, N_{t}=5, N_{\mu}=8$", linewidth=1, alpha=1)

plt.legend()
plt.show()