import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

mean1 = [-3, 0]
cov1 = [[1.0, 0.8], [0.8, 1.0]]
data1 = np.random.multivariate_normal(mean1, cov1, 500)

plt.scatter(data1[:,0], data1[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot of Problem 1")
plt.show()

plt.hist(data1[:,0], bins=20, alpha=0.5, label='X')
plt.hist(data1[:,1], bins=20, alpha=0.5, label='Y')
plt.xlim(-7, 4)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Problem 1")
plt.legend()
plt.show()

mean2 = [0, -3]
cov2 = [[1.0, 0.8], [0.8, 1.0]]
data2 = np.random.multivariate_normal(mean2, cov2, 500)

plt.scatter(data1[:,0], data1[:,1], label='0')
plt.scatter(data2[:,0], data2[:,1], label='1')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot combining Problem 1 and 4")
plt.legend()
plt.show()

combined_data = np.vstack((data1, data2))
labels = np.hstack((np.zeros(500), np.ones(500)))
labeled_data = np.hstack((combined_data, labels.reshape(-1,1)))

print("Combined and labeled data shape:", labeled_data.shape)
print(labeled_data[:10])
