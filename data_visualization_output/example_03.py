"""Example 03 - Grouped bar graph."""
import matplotlib.pyplot as plt
import numpy as np

# Defining categories and their corresponding values for two groups
categories = ['Category A', 'Category B', 'Category C']
values1 = [15, 24, 30]
values2 = [20, 18, 25]

# Setting the width of the bars
bar_width = 0.35

# Calculating bar positions for both groups
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

# Creating the first set of bars (Group 1)
plt.bar(bar_positions1, values1, width=bar_width, label='Group 1', color='skyblue')

# Creating the second set of bars (Group 2) next to the first set
plt.bar(bar_positions2, values2, width=bar_width, label='Group 2', color='orange')

# Adding labels to the axes
plt.xlabel('Categories')
plt.ylabel('Values')

# Centering the category names under each pair of bars
plt.xticks(bar_positions1 + bar_width / 2, categories)

# Adding a title to the graph
plt.title('Grouped Bar Graph')

# Displaying a legend to identify the groups
plt.legend()

# Showing the plot
plt.show()
