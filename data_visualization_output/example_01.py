"""Example 01 - Bar graph (vertical)."""
import matplotlib.pyplot as plt

salesrep = ['Ana', 'Juan', 'Lita', 'Rosa', 'Ato']
totalsales = [240, 430, 320, 125, 340]

plt.bar(salesrep, totalsales)
plt.title('Sales')
plt.xlabel('salesrep')
plt.ylabel('totalsales')
plt.show()
