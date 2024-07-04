## Predict Credit Approvals using Machine Learning

This project uses machine learning techniques to analyze and predict credit approvals. The dataset is preprocessed to handle missing values and categorical data, and several machine learning algorithms are applied to build predictive models.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)

## Installation
To get started with this project, clone the repository and install the necessary packages:
```bash
git clone https://github.com/anessrb/Credit-Machine-Learning.git
cd Credit-Machine-Learning
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is `train_u6lujuX_CVtuZ9i.csv`. Make sure the dataset is in the same directory as your project files.

## Data Preprocessing
The dataset is read and missing values are handled as follows:

```python
import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Check for missing values
df.isnull().sum().sort_values(ascending=False)

# Split the data into categorical and numerical
cat_data = []
num_data = []
for i, c in enumerate(df.dtypes):
    if c == 'object':
        cat_data.append(df.iloc[:, i])
    else:
        num_data.append(df.iloc[:, i])
cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

# Fill missing values in categorical data with the mode
cat_data = cat_data.apply(lambda x: x.fillna(x.mode().iloc[0]))

# Fill missing values in numerical data with the previous value
num_data.fillna(method='bfill', inplace=True)
```

## Exploratory Data Analysis
Several visualizations are created to understand the distribution and relationships in the data.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable
sns.countplot(df['Loan_Status'])
plt.show()

# Plot credit history vs loan status
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.show()

# Plot gender vs loan status
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.show()
```

## Model Training and Evaluation
Three machine learning algorithms are applied: Logistic Regression, K-Nearest Neighbors, and Decision Tree. The data is split into training and testing sets, and the models are evaluated.

```python
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split the data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

# Define the models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(max_depth=1, random_state=42),
}

# Train and evaluate the models
def train_test_eval(models, X_train, Y_train, X_test, Y_test):
    for name, model in models.items():
        model.fit(X_train, Y_train)
        accuracy = accuracy_score(Y_test, model.predict(X_test))
        print(f'{name}: Accuracy = {accuracy:.2f}')

train_test_eval(models, X_train, Y_train, X_test, Y_test)
```

## Saving the Model
The final model is saved using the `pickle` module for future use.

```python
import pickle

# Train the final model
classifier = LogisticRegression()
classifier.fit(X, Y)

# Save the model
pickle.dump(classifier, open('model.pkl', 'wb'))
```

## Usage
To use the trained model, load it using `pickle` and make predictions on new data.

```python
# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Make predictions
predictions = model.predict(new_data)
```

## Conclusion
This project demonstrates how to preprocess data, perform exploratory data analysis, train and evaluate multiple machine learning models, and save the final model for future use.

For any questions or issues, please contact [anessrabia34@gmail.com].
