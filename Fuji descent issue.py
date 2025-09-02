import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
csv_path = "mtfuji_data.csv"
np.set_printoptions(suppress=True)
fuji = np.loadtxt(csv_path, delimiter=",", skiprows=1)

# Create folder to save plots
output_dir = "fuji_plots"
os.makedirs(output_dir, exist_ok=True)

# Problem 2: Gradient at a point
def gradient_at_point(current_point):
    if current_point == 0:
        return 0
    return fuji[current_point,3] - fuji[current_point-1,3]

# Problem 3: Next point calculation
def next_point(current_point, alpha=0.2):
    grad = gradient_at_point(current_point)
    next_pt = current_point - alpha * grad
    next_pt = int(round(next_pt))
    next_pt = max(0, min(next_pt, len(fuji)-1))
    return next_pt

# Problem 4: Go down the mountain
def descend_mountain(start_point=136, alpha=0.2):
    path = [start_point]
    current = start_point
    while True:
        nxt = next_point(current, alpha)
        if nxt == current:
            break
        path.append(nxt)
        current = nxt
    return path

# Problem 1: Visualize elevation
plt.figure(figsize=(10,4))
plt.plot(fuji[:,0], fuji[:,3], label='Elevation')
plt.xlabel('Point Number')
plt.ylabel('Elevation [m]')
plt.title('Mt. Fuji Cross Section')
plt.grid(True)
plt.legend()
plt.savefig(f"{output_dir}/cross_section.png")
plt.close()

# Problem 5: Example descent visualization
path = descend_mountain()
plt.figure(figsize=(10,4))
plt.plot(fuji[:,0], fuji[:,3], label='Elevation')
plt.scatter(path, fuji[path,3], color='red', label='Descent Path')
plt.xlabel('Point Number')
plt.ylabel('Elevation [m]')
plt.title('Descent from Mt. Fuji')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/descent_example.png")
plt.close()

# Problem 6 & 7: Multiple initial points
start_points = [50, 100, 136, 180, 250]
paths = [descend_mountain(start, alpha=0.2) for start in start_points]
plt.figure(figsize=(10,4))
plt.plot(fuji[:,0], fuji[:,3], label='Elevation')
for i, path in enumerate(paths):
    plt.scatter(path, fuji[path,3], label=f'Start {start_points[i]}')
plt.xlabel('Point Number')
plt.ylabel('Elevation [m]')
plt.title('Descent from Various Initial Points')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/multiple_starts.png")
plt.close()

# Problem 8: Hyperparameter effect
alphas = [0.05, 0.2, 0.5]
plt.figure(figsize=(10,4))
plt.plot(fuji[:,0], fuji[:,3], label='Elevation')
for alpha in alphas:
    path = descend_mountain(start_point=136, alpha=alpha)
    plt.scatter(path, fuji[path,3], label=f'alpha={alpha}')
plt.xlabel('Point Number')
plt.ylabel('Elevation [m]')
plt.title('Descent with Different Alphas')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/alpha_effect.png")
plt.close()
