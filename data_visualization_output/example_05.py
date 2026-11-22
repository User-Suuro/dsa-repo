"""Example 05 - Exploded pie chart."""
import matplotlib.pyplot as plt

# Data for proportions
sizes = [20, 35, 25, 20]

# Offset for each slice (0.1 pulls Category B out of the pie)
explode = (0, 0.1, 0, 0)

# Creating an exploded pie chart
plt.pie(sizes, explode=explode,
        labels=['Category A', 'Category B', 'Category C', 'Category D'])
plt.title('Exploded Pie Chart')
plt.show()
