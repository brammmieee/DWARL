import numpy as np

import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()

# Generate fake data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the data
ax.plot(x, y)
plt.show()