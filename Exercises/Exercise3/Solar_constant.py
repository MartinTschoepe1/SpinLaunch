import matplotlib.pyplot as plt

# Data
planets = ["Merkur", "Venus", "Erde", "Mars", "Ceres", "Jupiter", "Saturn", "Uranus", "Neptun"]
distances = [0.387, 0.723, 1.000, 1.524, 2.766, 5.204, 9.582, 19.201, 30.047]
solar_energy = [9126.6, 2601.3, 1361, 589.2, 179, 50.50, 14.99, 3.71, 1.51]
# energy_comparison = [6.706, 1.911, 1, 0.433, 0.131, 0.037, 0.011, 0.0027, 0.00111]

# Plot
fig, ax1 = plt.subplots()

ax1.set_xlabel('Distance from Sun (AE)')
ax1.set_ylabel('Average Solar Energy (W/m²)', color='tab:blue')
ax1.plot(distances, distances, color='tab:red', marker='o')
ax1.set_xscale('log')
ax1.set_yscale('log')
# Set y range from 0.1 to default
ax1.set_ylim(0.00465, *ax1.get_ylim()[1:])

fig.tight_layout()
plt.title('Average Solar Energy and Distance from Sun for Planets')
plt.show()