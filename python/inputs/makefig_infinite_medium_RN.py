import numpy as np
import matplotlib.pyplot as plt
import pickle

output_file = 'infinite_medium_one_cell_RN.out'

data = []
with open(output_file, "rb") as f:
    while True:
        try:
            t = pickle.load(f)  # Load time
            temp = pickle.load(f)  # Load mesh.temp[0]
            radtemp = pickle.load(f)  # Load mesh.radtemp[0]
            
            print(f"Loaded: time={t}, temp={temp}, radtemp={radtemp}")  # Debugging
            print(f"Types: time={type(t)}, temp={type(temp)}, radtemp={type(radtemp)}")  # Check types
            
            data.append((t, temp, radtemp))
        except EOFError:
            print("End of file reached.")
            break


# Unpack for plotting
times, temps, radtemps = zip(*data)
print(len(times))
print(len(temps))
print(len(radtemps))
times = np.array(times, dtype=float)
temps = np.array(temps, dtype=float)
radtemps = np.array(radtemps, dtype=float)

plt.figure()
plt.plot(times, temps, label="Temperature")
plt.plot(times, radtemps, label="Radiation Temperature")
plt.legend()
plt.show()
