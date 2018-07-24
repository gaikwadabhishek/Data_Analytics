import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
url = '/home/administrator/Iris.csv'

df = pd.read_csv(url)

print(df.shape)

print(df.head(5))

print(df. tail(5))

print(df.dtypes)

print(df.describe())
'''
df =pd.DataFrame({'col1':np.random.randn(100),'col2':np.random.randn(100)})

df.hist(layout=(1,2))

plt.show()
'''
header = ['sepLength','sepWidth','petLength','petWidth','class']
df.columns = header

df.hist()
plt.show()

df.boxplot()

'''
plt.title('Iris Info')
plt.ylabel('Y axis')
plt.xlabel('X axis') 
'''
plt.show()

x = df.
