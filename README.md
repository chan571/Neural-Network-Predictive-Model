
## Load libraries
```
import urllib.request
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
```

## Fetch data from the url and loads it into a NumPy array 
```
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data)
dataset
```
<img width="581" alt="Screenshot 2024-02-26 at 1 35 52 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/65c2a62a-b113-476a-ad0b-4cb91eb9fd72">


## Preparing data for analysis
```
# Drop categorical variables
dataset = np.delete(dataset, [11, 12], axis=1)

# Combined last 7 columns into one to classify 7 Types of Steel Plates Faults
def get_fault_type(row):
    for idx, value in enumerate(row[-7:]):
        if value == 1:
            return idx + 1
    return None

fault_types = []
for row in dataset:
    fault_type = get_fault_type(row)
    fault_types.append(fault_type)

fault_types = np.array(fault_types)

# Concatenate fault_types as a new column to the dataset
dataset = np.column_stack((dataset, fault_types))
```

Next, normalization of the dataset
```
data_scale = StandardScaler().fit_transform(dataset)

```
## Check the size of the dataset
```
print(dataset.shape)
```
<img width="103" alt="Screenshot 2024-02-26 at 1 39 21 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/fcfdab89-4d7a-4450-9d32-01a2b0f695f5">


## Multi-layer Perceptron Regressor
The initially randomly chosen number of layers is 3, and the number of nodes is 5. 
The testing size is 25%.

```
X_train, X_test, y_train, y_test = train_test_split (data_scale[: , 0:25], data_scale[:, 25], train_size= 0.25, random_state=1)

# Train a neural network model, MLP regressor, using training data
reg = MLPRegressor (hidden_layer_sizes= (3, 5), random_state=1). fit(X_train, y_train)
```

Obtain predictions from the trained model for the testing dataset. 
Then, assess the predictions and goodness of fit for regression on the testing data.

```
prediction_result = reg.predict(X_test)
print(prediction_result)
```
<img width="558" alt="Screenshot 2024-02-26 at 2 18 56 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/a5cb9f0a-b925-4304-bf1b-3d707d24e565">

```
# R squared result
errors = reg.score(X_test, y_test)/ len(y_test)
print(errors)
```
<img width="210" alt="Screenshot 2024-02-26 at 2 17 06 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/d6c83a33-23b9-4182-988d-6aa384e5feac">



## Multi-layer Perceptron Classifier

```
# Train a neural network model, MLP classifier, using training data
X_train, X_test, y_train, y_test = train_test_split (dataset[:, 0:25], dataset[:, 25], test_size= 0.25, random_state=1)
clf = MLPClassifier(solver= 'sgd', alpha= 1e-5, hidden_layer_sizes= (3, 5), random_state= 1)
clf.fit(X_train, y_train)
```

Selects a specific row from the testing dataset, passes it to the trained classifier model.
```
print(clf.predict (X_test[2:3, :]))
```
<img width="37" alt="Screenshot 2024-02-26 at 2 53 35 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/ebf0aba0-0626-47b9-8ab2-aa75b1029f99">

Find prediction accuracy. 
```
# Classification, accuracy result
print(clf.score(X_test,y_test))
```
<img width="173" alt="Screenshot 2024-02-26 at 2 55 41 PM" src="https://github.com/chan571/Neural-Network-Predictive-Model/assets/157858508/6fa90fb9-fe48-4e80-bdd4-bb2d71bdf778">

## Conclusion
The regression model shows a very low goodness of fit, with a R-squared value of 0.00016694889970054935. 
This suggests that the model has weak ability to predict the relationship between predictors and the outcome, as well as to explain the variation observed.
In this scenario, a classification approach is preferable since the output is a categorical variable. 
Although the prediction accuracy is also low, 0.21193415637860083, it achieves better performance overall compared to the regression model.


