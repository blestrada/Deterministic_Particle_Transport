import pickle
import matplotlib.pyplot as plt

# Open the output file
fname = open("SuOlson4.out", "rb")

# Create figure and axis
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()

# Loop to read and plot data from the file
times = ["0.01 ns", "0.1 ns", "1 ns"]  # Corresponds to your plot times
for i in range(3):
    fname.readline()  # Skip the time line
    xdata = pickle.load(fname)  # cellpos
    temp = pickle.load(fname)   # temp
    radtemp = pickle.load(fname)  # radtemp

    plt.plot(xdata, temp, label=f"Temp @ {times[i]}", color='blue')
    plt.plot(xdata, radtemp, label=f"RadTemp @ {times[i]}", color='red')

# Set axis ticks, labels, and limits
ax.xaxis.set_ticks([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
plt.xlim(0, 20.0)
plt.ylim(0, 1.2)
ax.xaxis.set_label_text("x - cm")
ax.yaxis.set_label_text("T - keV")

# Add legend
plt.legend(loc=9, numpoints=1, frameon=False)

# Save figure
plt.savefig("SuOlson4.png", bbox_inches="tight", dpi=900)

# Optionally show the plot
# plt.show()

# Close the file
fname.close()
