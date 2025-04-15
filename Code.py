pip install category_encoders

import calendar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import datetime
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error, accuracy_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use(style='seaborn-v0_8-notebook')
%matplotlib inline


from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt
drive.mount('dataset', force_remount=True)
link = '/content/dataset/MyDrive/ML Project/Walmart.csv'
data = pd.read_csv(link)

data.info()
#entire dataset information

data.columns  = data.columns.str.lower()
data.rename({'holiday_flag': 'is_holiday'}, axis = 1, inplace = True)
data.columns
#Renaming colunms as lower case letters

data.isnull().sum()
#check if dataset has any null values

data.duplicated().sum()
#check if data has any duplicates

data.head()

# correct data format of the 'date' column
data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
#data.date=pd.to_datetime(data.date,format = "%d-%m-%Y")
#data['date'] = pd.to_datetime(data['date'], format = "%d-%m-%Y")

# Create a new column "year" containing the year
data['year'] = data['date'].dt.year

# Create a new column "quarter" containing the season number
data['quarter'] = data['date'].dt.quarter

def get_season(quarter):

    if quarter == 1:
        return 'Winter'
    elif quarter == 2:
        return 'Spring'
    elif quarter == 3:
        return 'Summer'
    else:
        return 'Autumn'

# Create a new column "season" containing the season
data['season'] = data['quarter'].apply(get_season)

# Create a new column "month" containing the month number
data['month'] = data['date'].dt.month

# Create a new column "month_name" containing the month names
data['month_name'] = data['date'].dt.month_name()

# Create a new column "week" containing the week number
data['week'] = data['date'].dt.isocalendar().week

# Create a new column "day_of_week" containing the day names
data['day_of_week'] = data['date'].dt.day_name()

print(data.columns)

data

data['week'] = data['week'].astype('int32')

data.describe()

data.head()

data[['weekly_sales', 'temperature', 'fuel_price', 'unemployment', 'cpi']].describe()

columns = ['weekly_sales', 'temperature', 'fuel_price', 'unemployment', 'cpi']
plt.figure(figsize=(8, 8))
for i,col in enumerate(columns):
    plt.subplot(3, 2, i+1)
    sns.histplot(data = data, x = col, kde = True, bins = 15, color = 'black')
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=data, x='is_holiday', ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=data, x='season', ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=data, x='month', ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(3, 4))
sns.countplot(data=data, x='day_of_week', ax=ax)
plt.show()

data.day_of_week.value_counts()

plt.figure(figsize=(16, 6))
sns.countplot(data = data, x = 'store')
plt.show()

plt.figure(figsize = (5, 3))
sns.barplot(data = data,
            x = 'is_holiday',
            y = 'weekly_sales',
            estimator = np.mean,
            ci = False)

# Add labels and title
plt.title('Average Sales by Holidays')
plt.xlabel('Is Holiday', size = 12)
plt.ylabel('Average Sales', size = 12)
plt.show()

plt.figure(figsize = (10, 4))
sns.barplot(data = data,
            x = 'is_holiday',
            y = 'weekly_sales',
            estimator = np.sum,
            ci = False)

# Add labels and title
plt.title('Total Sales by Holidays')
plt.xlabel('Is Holiday', size = 12)
plt.ylabel('Total Sales', size = 12)
plt.show()

gb_store = data.groupby('store')['weekly_sales'].sum().sort_values(ascending = False)
plt.figure(figsize = (12, 4))
sns.barplot(data = data,
            x = 'store',
            y = 'weekly_sales',
            order = gb_store.index,
            ci = False)

# Add labels and title
plt.title('Total Sales in each Store', size = 20)
plt.xlabel('Store', size = 15)
plt.ylabel('Total Sales', size = 15)
plt.show()

plt.figure(figsize = (7, 4))
plt.scatter(data.sort_values('fuel_price')['fuel_price'], data.sort_values('fuel_price')['weekly_sales'], color='b', edgecolors='black')
plt.title('Fuel Price')

plt.scatter(data.sort_values('temperature').temperature, data.sort_values('temperature').weekly_sales, color = 'gray', edgecolors='black')

plt.scatter(data.sort_values('cpi').cpi, data.sort_values('cpi').weekly_sales, color = 'r', edgecolors='black')

plt.scatter(data.sort_values('unemployment').unemployment, data.sort_values('unemployment').weekly_sales, color = 'g', edgecolors='black')

import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Scatter plot for Fuel Price
axes[0, 0].scatter(data.sort_values('fuel_price')['fuel_price'], data.sort_values('fuel_price')['weekly_sales'], color='b', edgecolors='black')
axes[0, 0].set_title('Fuel Price')

# Scatter plot for Temperature
axes[0, 1].scatter(data.sort_values('temperature')['temperature'], data.sort_values('temperature')['weekly_sales'], color='gray', edgecolors='black')
axes[0, 1].set_title('Temperature')

# Scatter plot for CPI
axes[1, 0].scatter(data.sort_values('cpi')['cpi'], data.sort_values('cpi')['weekly_sales'], color='r', edgecolors='black')
axes[1, 0].set_title('CPI')

# Scatter plot for Unemployment
axes[1, 1].scatter(data.sort_values('unemployment')['unemployment'], data.sort_values('unemployment')['weekly_sales'], color='g', edgecolors='black')
axes[1, 1].set_title('Unemployment')

# Adjust layout
plt.tight_layout()

plt.show()

# Let's calculate the Pearson Correlation Coefficient and P-value of 'fuel_price' and 'weekly_sales':
categorical_columns = ['temperature', 'fuel_price', 'unemployment', 'cpi']
target_column = 'weekly_sales'
for cat_column in categorical_columns:
  pearson_coef, p_value = stats.pearsonr(data[cat_column], data['weekly_sales'])
  print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

from scipy import stats
import pandas as pd

# Assuming 'data' is your DataFrame

numerical_columns = ['temperature', 'fuel_price', 'unemployment', 'cpi']
target_column = 'weekly_sales'

for num_column in numerical_columns:
    pearson_coef, p_value = stats.pearsonr(data[num_column], data[target_column])
    print(f"The Pearson Correlation Coefficient between {num_column} and {target_column} is {pearson_coef:.4f} with a P-value of {p_value:.4f}")

data.head()

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Assuming 'data' is your DataFrame
categorical_columns = ['is_holiday', 'year', 'season', 'month_name', 'day_of_week']
target_column = 'weekly_sales'

# Create a function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()  # Sum twice to get the total count
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# Calculate Cramér's V for each categorical variable
for cat_column in categorical_columns:
    confusion_matrix = pd.crosstab(data[cat_column], data[target_column])
    v_value = cramers_v(confusion_matrix)
    print(f"Cramér's V for {cat_column} vs. {target_column}: {v_value:.4f}")

data_bkp = data.copy()

data_prep = data_bkp.copy()

data_prep.drop(['date', 'year', 'quarter', 'month', 'day_of_week'], axis = 1, inplace = True)

data_prep.dtypes

data_prep['store'] = data_prep['store'].astype('object')
data_prep['is_holiday'] = data_prep['is_holiday'].astype('object')
data_prep['week'] = data_prep['week'].astype('object')

data_prep.dtypes

cols = ['fuel_price', 'temperature', 'cpi', 'unemployment']
plt.figure(figsize=(16,10))
for i,col in enumerate(cols):
    print(i, col)
    plt.subplot(3,2,i+1)
    sns.boxplot(data_prep, x = col, color = 'red')
plt.show()

print('Number of data rows: ', data_prep.shape[0])

data_prep.drop(data_prep[data_prep['temperature'] < 7].index, axis = 0, inplace = True)

data_prep.drop(data_prep[data_prep['unemployment'] < 4.4].index, axis = 0, inplace = True)
data_prep.drop(data_prep[data_prep['unemployment'] > 11].index, axis = 0, inplace = True)

cols = ['temperature', 'fuel_price', 'cpi', 'unemployment']
plt.figure(figsize=(16,10))
for i, col in enumerate(cols):
    print(i, col)
    plt.subplot(3,2,i+1)
    sns.boxplot(data_prep, x = col, color = 'g')
plt.show()

print('Number of data rows: ', data_prep.shape[0])

X = data_prep.drop('weekly_sales', axis = 1)
y = data_prep['weekly_sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

print('Shape of data      : ', X.shape)
print('Shape of train data: ', X_train.shape)
print('Shape of test data : ', X_test.shape)
print('Shape of test data : ', y_train.shape)
print('Shape of test data : ', y_test.shape)

num_features = data_prep.select_dtypes('number').columns.to_list()
num_features.remove('weekly_sales')

cat_features = data_prep.select_dtypes('object').columns.to_list()

print(f'Numerical Features : {num_features}')
print(f'Categorical Features: {cat_features}')

# data transformation pipeline
preprocessor = ColumnTransformer([
                                  ('num_features', StandardScaler(), num_features),
                                  ('cat_features', BinaryEncoder(), cat_features),
                                ])

# Fitting the training data
preprocessor.fit(X_train)

# Transform the training data
X_train_transformed = preprocessor.transform(X_train)

# Transform the testing data
X_test_transformed = preprocessor.transform(X_test)

data_prep

'''def model_evaluation(estimator, Training_Testing, X, y):

    '''#This function is used to evaluate the model through RMSE and R2'''

    # Y predict of X train or X test
    #predict_data = estimator.predict(X)

    #print(f'{Training_Testing} Accuracy: \n')
    #rmse = {round(np.sqrt(mean_squared_error(y, predict_data)), 2)}
    #print(f'-> Root Mean Squared Error: {round(np.sqrt(mean_squared_error(y, predict_data)), 2)}')
    #print(f'-> R-Squere score Training: {round(r2_score(y, predict_data) * 100, 2)} % \n')'''

def model_evaluation(estimator, Training_Testing, X, y):

    ''' This function is used to evaluate the model through RMSE and R2'''

    # Y predict of X train or X test
    predict_data = estimator.predict(X)

    # Calculate RMSE
    rmse = round(np.sqrt(mean_squared_error(y, predict_data)), 2)

    # Calculate the range of the target variable
    target_range = np.max(y) - np.min(y)

    # Calculate normalized RMSE
    normalized_rmse = rmse / target_range
    R2= {round(r2_score(y, predict_data) * 100, 2)}
    print(f'{Training_Testing} Accuracy: \n')
    print(f'-> Root Mean Squared Error: {rmse}')
    print(f'-> Normalized Root Mean Squared Error: {normalized_rmse}')
    print(f'-> R-Square score {Training_Testing}: {R2} % \n')



def Distribution_Plot(estimator, Training_Testing, X, y, Title):

    """This function is used to perform some model evaluation using training and testing data \
    by plotting the distribution of the actual and predicted values of the training or testing data."""

    # Y predict of X train or X test
    yhat = estimator.predict(X)
    plt.figure(figsize=(14, 6))
    ax1 = sns.distplot(y, hist = False, color = "b", label = f'Actual Values ({Training_Testing})')
    ax2 = sns.distplot(yhat, hist = False, color = "r", label = f'Predicted Values ({Training_Testing})', ax = ax1)
    plt.title(Title, size = 18)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,10))
    plt.scatter(y, yhat, c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(yhat), max(y))
    p2 = min(min(yhat), min(y))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


def cross_validation_score(estimator, X_train, y_train, score = 'r2', n = 5):

    '''This function is to validate the model'''


    validate = cross_val_score(estimator, X_train, y_train, scoring = score, cv = n)

    print(f'Cross Validation Scores: {validate} \n')
    print(f'Mean of Scores: {round(validate.mean() * 100, 2)} % \n')
    print(f'Standard Deviation of Scores: {validate.std()}')

def hyperparameter_tunning(estimator, X_train, y_train, param_grid, score = 'r2', n = 5):

    '''This function is used to find the best set of hyperparameters for the model to optimize its performance'''


    # Perform grid search
    grid_search = GridSearchCV(estimator = estimator,
                               param_grid = param_grid,
                               scoring = score,
                               cv = n, verbose=3, n_jobs=-1)

    # Fit the data
    grid_search.fit(X_train,y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best parameters and score
    print(f'Best parameters: {best_params} \n')
    print(f'Best score: {best_score}')

    # best estimator
    best_estimator = grid_search.best_estimator_

    return best_estimator

# Linear Regression Model
LR = LinearRegression()

# Fitting the data
LR.fit(X_train_transformed, y_train)

# Training Accuracy
model_evaluation(LR, 'Training', X_train_transformed, y_train)

# Figure 1: Plot of predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(LR, 'Training', X_train_transformed, y_train, Title)

# Polynomial Regression Model
LR_pipe = Pipeline([('poly_feat', PolynomialFeatures()),
                    ('lin_reg', LinearRegression())])

# Define the parameter grid to search
param_grid = {'poly_feat__degree': [2, 3, 4]}

best_estimator = hyperparameter_tunning(LR_pipe, X_train_transformed, y_train, param_grid, score = 'r2', n = 5)

# Linear Regression Model after tuning
poly_reg = best_estimator

# Training Accuracy Afer tuning
model_evaluation(poly_reg, 'Training', X_train_transformed, y_train)

# Figure 2: Plot of predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(poly_reg, 'Training', X_train_transformed, y_train, Title)

cross_validation_score(poly_reg, X_train_transformed, y_train)

# Testing Accuracy
model_evaluation(poly_reg, 'Testing', X_test_transformed, y_test)

# Figure 3: Plot of predicted value using the test data compared to the actual values of the test data.
Title='Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
Distribution_Plot(poly_reg, 'Testing', X_test_transformed, y_test, Title)

# Decision Tree regressor Model
tree = DecisionTreeRegressor()

# Fitting the training data
tree.fit(X_train_transformed, y_train)

# Training Accuracy
model_evaluation(tree, 'Training', X_train_transformed, y_train)

# Figure 1: Plot of predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(tree, 'Training', X_train_transformed, y_train, Title)

# Define the parameter grid to search
param_grid = {'max_depth': np.arange(2, 15),
              'min_samples_split': [10, 20, 30, 40, 50, 100, 200, 300]}

best_estimator = hyperparameter_tunning(tree, X_train_transformed, y_train, param_grid, score = 'r2', n = 5)

Best_Tree = best_estimator

model_evaluation(Best_Tree, 'Training', X_train_transformed, y_train)

Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(Best_Tree, 'Training', X_train_transformed, y_train, Title)

cross_validation_score(Best_Tree, X_train_transformed, y_train, n = 10)

# Testing Accuracy
model_evaluation(Best_Tree, 'Testing', X_test_transformed, y_test)

# Figure 3: Plot of predicted value using the test data compared to the actual values of the test data.
Title='Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
Distribution_Plot(Best_Tree, 'Testing', X_test_transformed, y_test, Title)

RF = RandomForestRegressor() # Random Forest Regressor Model

RF.fit(X_train_transformed, y_train) #Fitting the Training Data

y_pred = RF.predict(X_test_transformed)

model_evaluation(RF, 'Training', X_train_transformed, y_train)

# Plot of predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(RF, 'Training', X_train_transformed, y_train, Title)

# Define the parameter grid to search
param_grid = {'n_estimators': [25, 50, 70, 100],
              'max_features': [None, 1,3,5,8],
              'max_depth': np.arange(2,15),
              'min_samples_split': [2,5,10],
              'min_samples_leaf': [1, 2, 4]
              }

best_estimator = hyperparameter_tunning(RF, X_train_transformed, y_train, param_grid, score = 'r2', n = 5)

Best_RF = best_estimator

model_evaluation(Best_RF, 'Training', X_train_transformed, y_train)

Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(Best_RF, 'Training', X_train_transformed, y_train, Title)

cross_validation_score(Best_RF, X_train_transformed, y_train, n = 10)

# Testing Accuracy
model_evaluation(Best_RF, 'Testing', X_test_transformed, y_test)

# Figure 3: Plot of predicted value using the test data compared to the actual values of the test data.
Title='Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
Distribution_Plot(Best_RF, 'Testing', X_test_transformed, y_test, Title)

xgb_model = XGBRegressor()

xgb_model.fit(X_train_transformed, y_train)

model_evaluation(xgb_model, 'Training', X_train_transformed, y_train)

# Plot of predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(xgb_model, 'Training', X_train_transformed, y_train, Title)

# Define the parameter grid to search
param_grid = {
    'max_depth': np.arange(2,10),
    'n_estimators': np.arange(20,201,20)
}

best_estimator = hyperparameter_tunning(xgb_model, X_train_transformed, y_train, param_grid, score = 'r2', n = 5)

Best_xgb_model = best_estimator

model_evaluation(Best_xgb_model, 'Training', X_train_transformed, y_train)

Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
Distribution_Plot(Best_xgb_model, 'Training', X_train_transformed, y_train, Title)

cross_validation_score(Best_xgb_model, X_train_transformed, y_train, n = 10)

# Testing Accuracy
model_evaluation(Best_xgb_model, 'Testing', X_test_transformed, y_test)

# Figure 3: Plot of predicted value using the test data compared to the actual values of the test data.
Title='Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
Distribution_Plot(Best_xgb_model, 'Testing', X_test_transformed, y_test, Title)

