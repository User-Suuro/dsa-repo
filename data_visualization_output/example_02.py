"""Example 02 - Horizontal bar graph."""
import matplotlib.pyplot as plt

salesrep = ['Ana', 'Juan', 'Lita', 'Rosa', 'Ato']
totalsales = [240, 430, 320, 125, 340]

plt.barh(salesrep, totalsales)
plt.title('Sales')
# In a horizontal bar graph the axes are swapped:
# the values run along x and the categories along y.
plt.xlabel('totalsales')
plt.ylabel('salesrep')
plt.show()
