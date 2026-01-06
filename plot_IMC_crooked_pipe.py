import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Load your CSV
# df = pd.read_csv("output_DPT_nx2_ny2_nt10_nmu2_nphi16.csv")
df = pd.read_csv('temperature_history.csv')
# df = pd.read_csv("temperature_history_IMC_1e6.csv")
df_imc = pd.read_csv("temperature_history_ts0.3_sigma2000_imc.csv")
surface_temp=0.3
# Define the indices of interest
point_1_indices = [
    (0,0), (0,1), (0,2), (0, 3), (0, 4),
    (1,0), (1,1), (1,2), (1, 3), (1, 4),
    (2,0), (2,1), (2,2), (2, 3), (2, 4),
    (3,0), (3,1), (3,2), (3, 3), (3, 4),
    (4,0), (4,1), (4,2), (4, 3), (4, 4)
]
point_2_indices = [
    (35,0), (35,1), (35,2), (35, 3), (35, 4),
    (36,0), (36,1), (36,2), (36, 3), (36, 4),
    (37,0), (37,1), (37,2), (37, 3), (37, 4),
    (38,0), (38,1), (38,2), (38, 3), (38, 4),
    (39,0), (39,1), (39,2), (39, 3), (39, 4)
]
point_3_indices = [
    (51,28), (51,29), (51,30), (51,31), (51,32),
    (52,28), (52,29), (52,30), (52,31), (52,32),
    (53,28), (53,29), (53,30), (53,31), (53,32),
    (54,28), (54,29), (54,30), (54,31), (54,32),
    (55,28), (55,29), (55,30), (55,31), (55,32),
]
point_4_indices = [
    (70,0), (70,1), (70,2), (70, 3), (70, 4),
    (71,0), (71,1), (71,2), (71, 3), (71, 4),
    (72,0), (72,1), (72,2), (72, 3), (72, 4),
    (73,0), (73,1), (73,2), (73, 3), (73, 4),
    (74,0), (74,1), (74,2), (74, 3), (74, 4)
]
point_5_indices = [
    (101,0), (101,1), (101,2), (101, 3), (101, 4),
    (102,0), (102,1), (102,2), (102, 3), (103, 4),
    (103,0), (103,1), (103,2), (103, 3), (104, 4),
    (104,0), (104,1), (104,2), (104, 3), (104, 4),
    (105,0), (105,1), (105,2), (105, 3), (104, 4)
]

point_1 = [0.25, 0.00]
point_2 = [2.75, 0.0]
point_3 = [3.5, 1.25]
point_4 = [4.25, 0.0]
point_5 = [6.75, 0.0]

points = np.array([point_1, point_2, point_3, point_4, point_5])
x_coords = points[:, 0]
y_coords = points[:, 1]

x_edges = np.array([
    0.00000, 1.00000e-01, 2.00000e-01, 3.00000e-01, 4.00000e-01, 5.00000e-01, 6.00000e-01, 7.00000e-01,
    8.00000e-01, 9.00000e-01, 1.00000, 1.10000, 1.20000, 1.30000, 1.40000, 1.50000, 1.60000, 1.70000,
    1.80000, 1.90000, 2.00000, 2.10000, 2.20000, 2.30000, 2.40000, 2.43283, 2.45511, 2.47022, 2.48048,
    2.48743, 2.49215, 2.49535, 2.49753, 2.49900, 2.50000, 2.60000, 2.70000, 2.80000, 2.90000, 3.00000,
    3.00100, 3.00247, 3.00465, 3.00785, 3.01257, 3.01952, 3.02978, 3.04489, 3.06717, 3.10000, 3.20000,
    3.30000, 3.40000, 3.50000, 3.60000, 3.70000, 3.80000, 3.90000, 3.93283, 3.95511, 3.97022, 3.98048,
    3.98743, 3.99215, 3.99535, 3.99753, 3.99900, 4.00000, 4.10000, 4.20000, 4.30000, 4.40000, 4.50000,
    4.50100, 4.50247, 4.50465, 4.50785, 4.51257, 4.51952, 4.52978, 4.54489, 4.56717, 4.60000, 4.70000,
    4.80000, 4.90000, 5.00000, 5.10000, 5.20000, 5.30000, 5.40000, 5.50000, 5.60000, 5.70000, 5.80000,
    5.90000, 6.00000, 6.10000, 6.20000, 6.30000, 6.40000, 6.50000, 6.60000, 6.70000, 6.80000, 6.90000, 7.00000
])

y_edges = np.array([
    0.00000, 1.00000e-01, 2.00000e-01, 3.00000e-01, 4.00000e-01, 5.00000e-01,
    5.01000e-01, 5.02474e-01, 5.04646e-01, 5.07849e-01, 5.12568e-01,
    5.19525e-01, 5.29778e-01, 5.44891e-01, 5.67167e-01, 6.00000e-01,
    7.00000e-01, 8.00000e-01, 9.00000e-01, 9.32833e-01, 9.55109e-01,
    9.70222e-01, 9.80475e-01, 9.87432e-01, 9.92151e-01, 9.95354e-01,
    9.97526e-01, 9.99000e-01, 1.00000, 1.10000, 1.20000, 1.30000,
    1.40000, 1.50000, 1.50100, 1.50247, 1.50465, 1.50785, 1.51257,
    1.51952, 1.52978, 1.54489, 1.56717, 1.60000, 1.70000, 1.80000,
    1.90000, 2.00000
])

def get_grid_indices(coords, edges):
    """
    Calculates the grid cell indices for a set of coordinates.
    The cell index i is defined by the interval [edges[i], edges[i+1]).
    The valid indices are [0, len(edges) - 2].
    """
    # Number of cells is len(edges) - 1
    max_index = len(edges) - 2

    # Use side='right' to include the left boundary (edges[i])
    # The result i' is the index where the value is inserted *after* all existing equal elements.
    # For edges[i] <= x < edges[i+1], i' will be i+1.
    search_indices = np.searchsorted(edges, coords, side='right')

    # The cell index is i = i' - 1
    cell_indices = search_indices - 1

    # Clip the indices to ensure they are within the valid range [0, max_index]
    # This handles points outside the defined edge boundaries.
    indices_clipped = np.clip(cell_indices, 0, max_index)
    return indices_clipped

# Calculate indices
x_indices = get_grid_indices(x_coords, x_edges)
y_indices = get_grid_indices(y_coords, y_edges)

# Combine and present results
results = []
for i, point in enumerate(points):
    results.append({
        "point": point.tolist(),
        "x_index": int(x_indices[i]),
        "y_index": int(y_indices[i])
    })

print(results)



def get_avg_temp(df, indices):
    index_set = set(indices)
    mask = df.apply(lambda row: (row["x_idx"], row["y_idx"]) in index_set, axis=1)
    df_filtered = df[mask]
    return df_filtered.groupby("time")["temp"].mean().reset_index()

def get_avg_rad_temp(df, indices):
    index_set = set(indices)
    mask = df.apply(lambda row: (row["x_idx"], row["y_idx"]) in index_set, axis=1)
    df_filtered = df[mask]
    return df_filtered.groupby("time")["radtemp"].mean().reset_index()



# --- Build temperature grid at final time ---
final_time = df["time"].max()
df_final = df[df["time"] == final_time]

# Pivot into 2D array
temp_grid = df_final.pivot(index="x_idx", columns="y_idx", values="temp").values
rad_temp_grid = df_final.pivot(index="x_idx", columns="y_idx", values="radtemp").values
# --- Plot ---

plt.figure()
pc = plt.pcolormesh(x_edges,
                    y_edges,
                    temp_grid.T,   # Transpose to match orientation
                    cmap="inferno",
                    shading="flat")
plt.colorbar(pc, label="Temperature [keV]")
plt.clim(vmin=0.00, vmax=surface_temp)
plt.xlabel("z")
plt.ylabel("r")
plt.xlim(x_edges[0], x_edges[-1])
plt.ylim(0,2)
plt.axis('scaled')
plt.title(f"Temperature at t={final_time}")
plt.tight_layout
plt.show()

plt.figure()
pc = plt.pcolormesh(x_edges,
                    y_edges,
                    rad_temp_grid.T,   # Transpose to match orientation
                    cmap="inferno",
                    shading="flat")
plt.colorbar(pc, label="Temperature [keV]")
plt.clim(vmin=0.00, vmax=surface_temp)
plt.xlabel("z")
plt.ylabel("r")
plt.xlim(x_edges[0], x_edges[-1])
plt.ylim(0,2)
plt.axis('scaled')
plt.title(f"Radiation Temperature at t={final_time}")
plt.tight_layout
plt.show()


# Make an animation of the material temperature over time.

# --- Animation setup ---
# times = sorted(df["time"].unique())  # all time points
# temp_min, temp_max = 0.05, 0.3       # fixed color scale for consistency

# fig, ax = plt.subplots(figsize=(6, 5))

# pc = ax.pcolormesh(
#     x_edges,
#     y_edges,
#     np.zeros((len(y_edges)-1, len(x_edges)-1)),  # placeholder array
#     cmap="inferno",
#     shading="flat"
# )
# cbar = fig.colorbar(pc, ax=ax, label="Temperature [keV]")
# pc.set_clim(vmin=temp_min, vmax=temp_max)

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_xlim(x_edges[0], x_edges[-1])
# ax.set_ylim(0, 2)
# ax.set_aspect("equal")
# title = ax.set_title("")

# def update(frame):
#     t = times[frame]
#     df_t = df[df["time"] == t]
#     temp_grid = df_t.pivot(index="x_idx", columns="y_idx", values="temp").values
#     pc.set_array(temp_grid.T.ravel())  # update color values
#     title.set_text(f"Temperature at t={t:.3e}")
#     return pc, title

# ani = animation.FuncAnimation(
#     fig, update, frames=len(times), blit=False, interval=200
# )

# # Save as mp4 (requires ffmpeg installed)
# ani.save("temperature_evolution.mp4", writer="ffmpeg", dpi=150)

# plt.close(fig)  # close figure to avoid extra static plot


# Plot Fiducial Points

# Get averages for both sets of points
avg_temp_point1 = get_avg_temp(df, point_1_indices)
avg_temp_point2 = get_avg_temp(df, point_2_indices)
avg_temp_point3 = get_avg_temp(df, point_3_indices)
avg_temp_point4 = get_avg_temp(df, point_4_indices)
avg_temp_point5 = get_avg_temp(df, point_5_indices)

# avg_radtemp_point1 = get_avg_rad_temp(df, point_1_indices)
# avg_radtemp_point2 = get_avg_rad_temp(df, point_2_indices)
# avg_radtemp_point3 = get_avg_rad_temp(df, point_3_indices)
# avg_radtemp_point4 = get_avg_rad_temp(df, point_4_indices)
# avg_radtemp_point5 = get_avg_rad_temp(df, point_5_indices)

avg_temp_point1_imc = get_avg_temp(df_imc, point_1_indices)
avg_temp_point2_imc = get_avg_temp(df_imc, point_2_indices)
avg_temp_point3_imc = get_avg_temp(df_imc, point_3_indices)
avg_temp_point4_imc = get_avg_temp(df_imc, point_4_indices)
avg_temp_point5_imc = get_avg_temp(df_imc, point_5_indices)

# avg_radtemp_point1_imc = get_avg_rad_temp(df_imc, point_1_indices)
# avg_radtemp_point2_imc = get_avg_rad_temp(df_imc, point_2_indices)
# avg_radtemp_point3_imc = get_avg_rad_temp(df_imc, point_3_indices)
# avg_radtemp_point4_imc = get_avg_rad_temp(df_imc, point_4_indices)
# avg_radtemp_point5_imc = get_avg_rad_temp(df_imc, point_5_indices)
print(f'finished getting averages')
plt.figure(figsize=(8,6))
markersize=2.0

# DPT temp
plt.plot(avg_temp_point1["time"], avg_temp_point1["temp"], 
         marker="+", color="blue", label="DPT", markersize=markersize)
plt.plot(avg_temp_point2["time"], avg_temp_point2["temp"], 
         marker="+", color="blue", markersize=markersize)
plt.plot(avg_temp_point3["time"], avg_temp_point3["temp"], 
         marker="+", color="blue", markersize=markersize )
plt.plot(avg_temp_point4["time"], avg_temp_point4["temp"], 
         marker="+", color="blue", markersize=markersize)
plt.plot(avg_temp_point5["time"], avg_temp_point5["temp"], 
         marker="+", color="blue", markersize=markersize)

# DPT rad temp
# plt.plot(avg_radtemp_point1["time"], avg_radtemp_point1["radtemp"], 
#          marker="+", color="blue", label="DPT Tr", )
# plt.plot(avg_radtemp_point2["time"], avg_radtemp_point2["radtemp"], 
#          marker="+", color="blue", )
# plt.plot(avg_radtemp_point3["time"], avg_radtemp_point3["radtemp"], 
#          marker="+", color="blue", )
# plt.plot(avg_radtemp_point4["time"], avg_radtemp_point4["radtemp"], 
#          marker="+", color="blue", )
# plt.plot(avg_radtemp_point5["time"], avg_radtemp_point5["radtemp"], 
#          marker="+", color="blue", )

# IMC temp
plt.plot(avg_temp_point1_imc["time"], avg_temp_point1_imc["temp"], 
         marker="x", color="red", label="IMC", markersize=markersize )
plt.plot(avg_temp_point2_imc["time"], avg_temp_point2_imc["temp"], 
         marker="x", color="red", markersize=markersize)
plt.plot(avg_temp_point3_imc["time"], avg_temp_point3_imc["temp"], 
         marker="x", color="red", markersize=markersize)
plt.plot(avg_temp_point4_imc["time"], avg_temp_point4_imc["temp"], 
         marker="x", color="red", markersize=markersize)
plt.plot(avg_temp_point5_imc["time"], avg_temp_point5_imc["temp"], 
         marker="x", color="red", markersize=markersize)
# IMC rad temp
# plt.plot(avg_radtemp_point1_imc["time"], avg_radtemp_point1_imc["radtemp"], 
#          marker="x", color="blue", label="IMC Tr", )
# plt.plot(avg_radtemp_point2_imc["time"], avg_radtemp_point2_imc["radtemp"], 
#          marker="x", color="blue", )
# plt.plot(avg_radtemp_point3_imc["time"], avg_radtemp_point3_imc["radtemp"], 
#          marker="x", color="blue", )
# plt.plot(avg_radtemp_point4_imc["time"], avg_radtemp_point4_imc["radtemp"], 
#          marker="x", color="blue", )
# plt.plot(avg_radtemp_point5_imc["time"], avg_radtemp_point5_imc["radtemp"], 
#          marker="x", color="blue", )



plt.xlabel("Time [sh]")
plt.ylabel(" Material Temperature [keV]")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-3, 100.0)
plt.ylim(0.0, surface_temp)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

plot_thermal_energy = True

if plot_thermal_energy:
    num_x_idx = len(x_edges) - 1
    num_y_idx = len(y_edges) - 1

    thin_opacity = 2.0
    thick_opacity = 200.0
    opacity = np.zeros((num_x_idx, num_y_idx))
    heat_capacity = np.zeros((num_x_idx, num_y_idx))
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    area = np.outer(dx, dy)

    # Make thin and thick masks
    for i in range(num_x_idx):
        for j in range(num_y_idx):
            x_left   = x_edges[i]
            x_right  = x_edges[i+1]
            y_bot    = y_edges[j]
            y_top    = y_edges[j+1]

            # Region 1: Bottom left
            if x_right <= 2.5 and y_top <= 0.5:
                opacity[i, j] = thin_opacity
                    

            # Region 2: Vertical left gap
            if 2.5 <= x_left <= 3.0 and y_top <= 1.5:
                opacity[i, j] = thin_opacity
                
            # Region 3: Horizontal middle strip
            if 3.0 <= x_left <= 4.0 and (y_bot < 1.5 and y_top >= 1.00):
                opacity[i, j] = thin_opacity
                    

            # Region 4: Vertical right gap
            if 4.0 <= x_left <= 4.5 and y_top <= 1.5:
                opacity[i, j] = thin_opacity
                

            # Region 5: Bottom right
            if x_left >= 4.5 and y_top <= 0.5:
                opacity[i, j] = thin_opacity
                
    thin_cells = (opacity == thin_opacity)
    opacity[~thin_cells] = thick_opacity 
    thick_cells = ~thin_cells

    # Set heat capacities
    heat_capacity[thick_cells] = 1.0 # jk/cm^3-keV
    heat_capacity[thin_cells] = 1e-3 # jk/cm^3-keV

    

    # Convert indices to integers
    df['x_idx'] = df['x_idx'].astype(int)
    df['y_idx'] = df['y_idx'].astype(int)

    df_imc['x_idx'] = df_imc['x_idx'].astype(int)
    df_imc['y_idx'] = df_imc['y_idx'].astype(int)

    # Map the 2D array properties (heat_capacity and area) to the DataFrame rows
    # We use .at[] for fast, individual lookups
    def get_cell_property(row, prop_array):
        # Explicitly cast the index values to standard Python integers (int)
        x_i = int(row['x_idx'])
        y_i = int(row['y_idx'])
        return prop_array[x_i, y_i]

    df['heat_capacity'] = df.apply(lambda row: get_cell_property(row, heat_capacity), axis=1)
    df['area'] = df.apply(lambda row: get_cell_property(row, area), axis=1)

    df_imc['heat_capacity'] = df_imc.apply(lambda row: get_cell_property(row, heat_capacity), axis=1)
    df_imc['area'] = df_imc.apply(lambda row: get_cell_property(row, area), axis=1)

    # Assign cell type ('thin' or 'thick')
    def get_cell_type(row):
        # Explicitly cast to integer for NumPy indexing
        x_i = int(row['x_idx'])
        y_i = int(row['y_idx'])
        
        # thin_cells is a boolean NumPy array, which requires integer indices
        if thin_cells[x_i, y_i]:
            return 'thin'
        else:
            return 'thick'

    df['cell_type'] = df.apply(get_cell_type, axis=1)
    df_imc['cell_type'] = df_imc.apply(get_cell_type, axis=1)

    # Calculate thermal energy for each cell
    # thermal energy = temp * heat_capacity * area
    df_imc['thermal_energy'] = df_imc['temp'] * df_imc['heat_capacity'] * df_imc['area']
    df['thermal_energy'] = df['temp'] * df["heat_capacity"] * df["area"]
    # --- 3. Aggregation and Plotting ---

    # Sum the thermal energy by time and cell type
    summed_energy_dpt = df.groupby(['time', 'cell_type'])['thermal_energy'].sum().reset_index()
    summed_energy_imc = df_imc.groupby(['time', 'cell_type'])['thermal_energy'].sum().reset_index()
    # Pivot the data for easier plotting
    plot_data_dpt = summed_energy_dpt.pivot(index='time', columns='cell_type', values='thermal_energy')
    plot_data_imc = summed_energy_imc.pivot(index='time', columns='cell_type', values='thermal_energy')
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data_dpt.index, plot_data_dpt['thick'], label='DPT', marker='o', markersize=3)
    plt.plot(plot_data_imc.index, plot_data_imc['thick'], label='IMC_1e6', marker='s', markersize=3)
    plt.xlabel('Time (sh)')
    plt.ylabel('Thermal Energy (jrk)')
    plt.title('Crooked Pipe Thermal energy in thick part')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.savefig('thermal_energy_vs_time.png')
    print("Plot saved as 'thermal_energy_vs_time.png'")