from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = '/home/administrator/Iris.csv'
header = ['sepLength','sepWidth','petLength','petWidth','class']

df = pd.read_csv(url, names = ['sep_length','sep_width','pet_length','pet_width','class'])
#scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
df.columns = header
scatter_matrix(df, alpha=0.2, diagonal = 'class')
plt.show()
