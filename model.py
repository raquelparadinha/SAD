import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('WXAgg')

dataset = pd.read_csv('Dataset_Trabalho.csv', sep=';')
print(dataset)

sns.set_theme(style='dark')

sns.countplot(x='Target', data=dataset)
plt.title(f'Classes distribution')
plt.xlabel('Classes')
plt.show()

# sns.distplot(dataset['Target'])
# plt.show()

