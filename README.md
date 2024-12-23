Q1:
THICKNESS = 0.00008  # meters

num_folds = 43
folded_thickness = THICKNESS * 2 ** num_folds

print("Thickness after {} folds: {} meters".format(num_folds, folded_thickness))



Q2:
print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

Q3:
THICKNESS = 0.00008  # meters
num_folds = 43

current_thickness = THICKNESS
for _ in range(num_folds):
  current_thickness *= 2

print("Thickness after {} folds: {} meters".format(num_folds, current_thickness))

Q4:
start = time.time()
THICKNESS = 0.00008  # meters
num_folds = 43

current_thickness = THICKNESS
for _ in range(num_folds):
  current_thickness *= 2

elapsed_time = time.time() - start

print("for loop time: {}[s]".format(elapsed_time))

Q5:

THICKNESS = 0.00008  # meters
num_folds = 43

thicknesses = [THICKNESS]  # Initial thickness
for _ in range(num_folds):
  current_thickness = thicknesses[-1] * 2  # Access last element
  thicknesses.append(current_thickness)

print("Length of thicknesses list:", len(thicknesses))  # Verify 44 values

Q6:

import matplotlib.pyplot as plt

THICKNESS = 0.00008  # meters
num_folds = 43

thicknesses = [THICKNESS]
for _ in range(num_folds):
  current_thickness = thicknesses[-1] * 2
  thicknesses.append(current_thickness)

# Generate x-axis values (number of folds)
folds = range(num_folds + 1)

plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness (meters)")
plt.plot(folds, thicknesses)
plt.show()

# Explanation: Thickness increases exponentially with each fold.
