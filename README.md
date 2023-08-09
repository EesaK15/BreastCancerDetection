# BreastCancerDetection
"Breast Cancer Classification using K-Nearest Neighbors: Employ machine learning and K-Nearest Neighbors classifiers to analyze breast cancer data, distinguishing between malignant and benign cells

### A benign tumor is made up of cells that don't threaten to invade other tissues.
### Malignant tumors are made of cancer cells that can grow uncontrollably and invade nearby tissues

### Importing Libraries & Dataset
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data.keys())
```

### Assigning Values
```
print(data['feature_names']) #  x values
print(data['target_names']) # y values

X = data['data'] # Just holds numerical values
Y = data['target']

```

### Training the Data
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2) # Training on 80 percent data and testing on 20
clf = KNeighborsClassifier()
clf.fit(x_train, y_train) # This takes a random 80 percent of the data to train and the rest is testing
print(clf.score(x_test,y_test)) # it is going to look at test data, and create a predicition and will compare to y_test
# Score returns accuracy

```
### Creating A Prediction
```
print(data['feature_names'])
# To find the number of feature_names

print(len(data['feature_names']))

```
### Creating A Prediction Continued
```
import random
x_new = np.array(random.sample(range(0,50), 30)) # 0 - 50 for 30 values
print(data['target_names'][clf.predict([x_new])])

```
### Creating A Data Frame and Checking Correlations
```
# Define the Column Data

column_data = np.concatenate([data['data'], data['target'][:,None]], axis = 1)
print(column_data)

column_names = np.concatenate([data['feature_names'],['Class']])
df = pd.DataFrame(column_data, columns = column_names)

print(df.head(5))

df.corr()
```
### Data Visualization - Creating HeatMaps
```
sns.heatmap(df.corr(), cmap = 'coolwarm', annot_kws = {'fontsize':8})
plt.tight_layout()
plt.show()

# The more blue or dark blue, the more negativley correlated it is 
# The more a negative feature occurs, the more malignant mean

```
