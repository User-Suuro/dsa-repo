"""Example 06 - Percentage-labeled pie chart."""
import matplotlib.pyplot as plt

# Data for proportions
sizes = [15, 30, 45, 10]

# Creating a pie chart with percentage labels
plt.pie(sizes,
        labels=['Category A', 'Category B', 'Category C', 'Category D'],
        autopct='%1.1f%%')
plt.title('Percentage-labeled Pie Chart')
plt.show()
