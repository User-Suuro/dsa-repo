"""Example 04 - Stacked bar graph."""
import matplotlib.pyplot as plt

# Defining categories and values for two groups
categories = ['Category A', 'Category B', 'Category C']
values1 = [15, 24, 30]
values2 = [20, 18, 25]

# Creating the first set of bars (Group 1) without any offset
plt.bar(categories, values1, label='Group 1', color='skyblue')

# Creating the second set of bars (Group 2) plotted with 'bottom'
# set to the values of Group 1.
# This makes Group 2 bars stacked on top of Group 1 bars.
plt.bar(categories, values2, bottom=values1, label='Group 2', color='orange')

# Adding labels to the axes
plt.xlabel('Categories')
plt.ylabel('Values')

plt.title('Stacked Bar Graph')
plt.legend()
plt.show()
