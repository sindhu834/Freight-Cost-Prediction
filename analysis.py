import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

plt.scatter(df['Distance'], df['cost'])
plt.xlabel('Distance')
plt.ylabel('Cost')
plt.title('Distance vs Cost')
plt.show()

plt.scatter(df['Weight'], df['cost'])
plt.xlabel('Weight')
plt.ylabel('Cost')
plt.title('Weight vs Cost')
plt.show()