"""Example 07 - Colored and sized scatter plot."""
import matplotlib.pyplot as plt

# Data
x = [1, 3, 5, 7, 9]
y = [10, 5, 15, 8, 12]
sizes = [30, 80, 120, 40, 60]
colors = ['red', 'green', 'blue', 'yellow', 'purple']

# Creating a scatter plot with varied sizes and colors
plt.scatter(x, y, s=sizes, c=colors)
plt.title('Colored and Sized Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
