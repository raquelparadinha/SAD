import numpy as np
import matplotlib.pyplot as plt
import plotext as plx
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.utils import resample

dataset = pd.read_csv('Dataset_Trabalho.csv', sep=';')
print(dataset.describe())


ax = sns.countplot(x='Target', data=dataset)

# Extract the counts from the countplot
x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
counts = [rect.get_height() for rect in ax.patches]
print("Distribution [Dropout, Graduate, Enrolled]: ", counts)

# Create an ASCII bar plot using Plotext
plx.bar(x_labels, counts)
plx.title('Classes distribution')
plx.xlabel('Classes')
plx.ylabel('Count')

# Display the countplot in the terminal
plx.show()

df_majority = dataset[(dataset['Target']=='Graduate')] 
df_intermediary = dataset[(dataset['Target']=='Dropout')]
df_minority = dataset[(dataset['Target']=='Enrolled')] 

# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= int(counts[1])) # to match majority class

df_intermediary_upsampled = resample(df_intermediary, 
                                 replace=True,    # sample with replacement
                                 n_samples= int(counts[1])) # to match majority class

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_intermediary_upsampled, df_majority, df_minority_upsampled])
print(df_upsampled)
ax = sns.countplot(x='Target', data=df_upsampled)

# Extract the counts from the countplot
x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
counts = [rect.get_height() for rect in ax.patches]
print("Distribution [Dropout, Graduate, Enrolled]: ", counts)

# Create an ASCII bar plot using Plotext
plx.bar(x_labels, counts)
plx.title('Classes distribution resampled')
plx.xlabel('Classes')
plx.ylabel('Count')

# Display the countplot in the terminal
plx.show()