import matplotlib.pyplot as plt
import numpy as np  # For more efficient calculations

def sorori_shinzaemon(n_days=100):
    """
    Calculates the number of rice grains received each day and cumulatively
    according to the Sorori Shinzaemon legend.

    Parameters:
    n_days (int): The number of days to calculate for.

    Returns:
    tuple: A tuple containing two NumPy arrays:
        - daily_grains (ndarray): Number of grains received each day.
        - cumulative_grains (ndarray): Cumulative number of grains received.
        Returns None if n_days is not a positive integer
    """

    if not isinstance(n_days, int) or n_days <= 0:
        print("n_days must be a positive integer.")
        return None

    daily_grains = np.array([2**day for day in range(n_days)])
    cumulative_grains = np.cumsum(daily_grains)

    return daily_grains, cumulative_grains

# --- Problem 1: 100 days ---
days_100 = 100
grains_daily_100, grains_cumulative_100 = sorori_shinzaemon(days_100)

if grains_daily_100 is not None:
    print(f"Rice grains on Day {days_100}: {grains_daily_100[-1]:,}")
    print(f"Total rice grains after {days_100} days: {grains_cumulative_100[-1]:,}")

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, days_100 + 1), grains_daily_100, label='Daily Rice Grains')
    plt.title("Daily Rice Grains (100 Days)")
    plt.xlabel("Day")
    plt.ylabel("Rice Grains")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, days_100 + 1), grains_cumulative_100, label='Cumulative Rice Grains')
    plt.title("Cumulative Rice Grains (100 Days)")
    plt.xlabel("Day")
    plt.ylabel("Rice Grains")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Problem 2: 81 days ---
days_81 = 81
grains_daily_81, grains_cumulative_81 = sorori_shinzaemon(days_81)

if grains_daily_81 is not None:
    print(f"\nRice grains on Day {days_81}: {grains_daily_81[-1]:,}")
    print(f"Total rice grains after {days_81} days: {grains_cumulative_81[-1]:,}")

    # Plotting (Simplified)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, days_81 + 1), grains_cumulative_81, label=f'Cumulative Rice Grains ({days_81} Days)')
    plt.title(f"Cumulative Rice Grains ({days_81} Days)")
    plt.xlabel("Day")
    plt.ylabel("Rice Grains")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
