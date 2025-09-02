import math
import matplotlib.pyplot as plt

# Volume of one chestnut bun (m^3)
bun_volume = 1e-4

# Astronomical Unit (m)
AU = 1.496e11

# Approximate radius of Solar System (to Neptune, ~30 AU)
R = 30 * AU

# Volume of Solar System (sphere approximation)
solar_system_volume = (4/3) * math.pi * (R**3)

def doubling_time(target_volume, initial_volume=1e-4, interval=5):
    n = math.ceil(math.log2(target_volume / initial_volume))
    time_minutes = n * interval
    return n, time_minutes

# Calculate for Solar System
n_intervals, total_minutes = doubling_time(solar_system_volume, bun_volume)

days = total_minutes / (60 * 24)
years = days / 365

print(f"Solar System Volume: {solar_system_volume:.2e} m^3")
print(f"Doublings needed: {n_intervals}")
print(f"Time: {total_minutes:.2e} minutes ({days:.2e} days â‰ˆ {years:.2e} years)")

# Visualization (first 200 doublings for clarity)
volumes = [bun_volume * (2**i) for i in range(200)]
times = [i*5 for i in range(200)]

plt.plot(times, volumes)
plt.yscale("log")
plt.xlabel("Time (minutes)")
plt.ylabel("Total volume of buns (mÂ³, log scale)")
plt.title("Exponential Growth of Chestnut Buns")
plt.grid(True)
plt.show()
