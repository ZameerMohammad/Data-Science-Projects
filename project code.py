
#---------------------------------------Different types of EDA's-----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Ecommerce_Customers.csv")
df
df.columns
df.dtypes

# Univariate Analysis
# Histogram of Avg Session Length
plt.figure(figsize=(8, 6))
sns.histplot(df['Avg Session Length'], bins=20, kde=True)
plt.title('Distribution of Avg Session Length')
plt.xlabel('Avg Session Length')
plt.ylabel('Frequency')
plt.show()

# Histogram of Time on App
plt.figure(figsize=(8, 6))
sns.histplot(df['Time on App'], bins=20, kde=True)
plt.title('Distribution of Time on App')
plt.xlabel('Time on App')
plt.ylabel('Frequency')
plt.show()

# Histogram of Time on Website
plt.figure(figsize=(8, 6))
sns.histplot(df['Time on Website'], bins=20, kde=True)
plt.title('Distribution of Time on Website')
plt.xlabel('Time on Website')
plt.ylabel('Frequency')
plt.show()

# Histogram of Length of Membership
plt.figure(figsize=(8, 6))
sns.histplot(df['Length of Membership'], bins=20, kde=True)
plt.title('Distribution of Length of Membership')
plt.xlabel('Length of Membership')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
# Scatter plot of Avg Session Length vs. Yearly Amount Spent
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Avg Session Length', y='Yearly Amount Spent')
plt.title('Scatter plot of Avg Session Length vs. Yearly Amount Spent')
plt.xlabel('Avg Session Length')
plt.ylabel('Yearly Amount Spent')
plt.show()

# Scatter plot of Time on App vs. Yearly Amount Spent
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Time on App', y='Yearly Amount Spent')
plt.title('Scatter plot of Time on App vs. Yearly Amount Spent')
plt.xlabel('Time on App')
plt.ylabel('Yearly Amount Spent')
plt.show()

# Scatter plot of Time on Website vs. Yearly Amount Spent
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Time on Website', y='Yearly Amount Spent')
plt.title('Scatter plot of Time on Website vs. Yearly Amount Spent')
plt.xlabel('Time on Website')
plt.ylabel('Yearly Amount Spent')
plt.show()

# Scatter plot of Length of Membership vs. Yearly Amount Spent
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Length of Membership', y='Yearly Amount Spent')
plt.title('Scatter plot of Length of Membership vs. Yearly Amount Spent')
plt.xlabel('Length of Membership')
plt.ylabel('Yearly Amount Spent')
plt.show()

# Select only numerical columns for analysis
numerical_columns = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']
df_numeric = df[numerical_columns]

# Pair plot
sns.pairplot(df_numeric)
plt.show()

# Correlation Analysis
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Correlation Analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Correlation Analysis
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Select only numerical columns for analysis
numerical_columns = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']
df_numeric = df[numerical_columns]

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

#---------------------------------model building--------------------------------------------------------------------------------
import pandas as pd
import numpy as np
df=pd.read_csv("Ecommerce_Customers.csv")
df

#data splitting
Y=df["Yearly Amount Spent"]
X=df[['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

#Data transformation
#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)

#Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

#1)================================LINEAR REGRESSION==========================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
print("\nLinear Regression Model:")
LR=LinearRegression()
LR.fit(X_train, Y_train)

# Predict on training and testing sets
Y_train_pred = LR.predict(X_train)
Y_test_pred = LR.predict(X_test)

# Evaluate model performance
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nTraining Set Performance:")
print("Mean Squared Error:", mse_train.round(2))
print("R-squared Score:", r2_train.round(2))

print("\nTest Set Performance:")
print("Mean Squared Error:", mse_test.round(2))
print("R-squared Score:", r2_test.round(2))

#cross validation for linear regression results
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test  = LR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print("\nCross validation for linear regression results:")
print("\nCross validation training Error:",np.mean(training_error).round(2))
print("Cross validation test Error:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error)).round(2))

'''
Cross validation training Error: 9.91
Cross validation test Error: 9.99
variance: 0.08
'''

#2)===================================KNN=============================================

from sklearn.neighbors import KNeighborsRegressor
KNNR = KNeighborsRegressor(metric='euclidean', n_neighbors=13)

KNNR.fit(X_train,Y_train)
Y_pred_train = KNNR.predict(X_train)
Y_pred_test  = KNNR.predict(X_test)

#metrics
print("\nKNN Regression Model:")
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nTraining Set Performance:")
print("Mean Squared Error:", mse_train.round(2))
print("R-squared Score:", r2_train.round(2))

print("\nTest Set Performance:")
print("Mean Squared Error:", mse_test.round(2))
print("R-squared Score:", r2_test.round(2))

#cross validation for KNN regression results
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    KNNR.fit(X_train,Y_train)
    Y_pred_train = KNNR.predict(X_train)
    Y_pred_test  = KNNR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print("\nCross validation for knn regression results:")
print("Cross validation training Error:",np.mean(training_error).round(2))
print("Cross validation test Error:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error)).round(2))

'''
Cross validation for knn regression results:
Cross validation training Error: 26.52
Cross validation test Error: 28.44
variance: 1.91
'''

#3)=============================================DT===================================================

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(criterion="squared_error")

DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_test  = DT.predict(X_test)

#metrics
print("\nDT Regression Model:")
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nTraining Set Performance:")
print("Mean Squared Error:", mse_train.round(2))
print("R-squared Score:", r2_train.round(2))

print("\nTest Set Performance:")
print("Mean Squared Error:", mse_test.round(2))
print("R-squared Score:", r2_test.round(2))

#cross validation for DT regression results
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    LR.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print("\nCross validation for DT regression results:")
print("Cross validation training Error:",np.mean(training_error).round(2))
print("Cross validation test Error:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error)).round(2))

'''
Cross validation for DT regression results:
Cross validation training Error: 14.84
Cross validation test Error: 14.45
variance: -0.39
'''

#4)============================================SVR==================================================


from sklearn.svm import SVR
SVR = SVR(kernel='poly', degree=3, gamma='scale', coef0=1)

SVR.fit(X_train,Y_train)
Y_pred_train = SVR.predict(X_train)
Y_pred_test  = SVR.predict(X_test)

#metrics
print("\nSVR Regression Model:")
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nTraining Set Performance:")
print("Mean Squared Error:", mse_train.round(2))
print("R-squared Score:", r2_train.round(2))

print("\nTest Set Performance:")
print("Mean Squared Error:", mse_test.round(2))
print("R-squared Score:", r2_test.round(2))

#cross validation for SVR regression results
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    SVR.fit(X_train,Y_train)
    Y_pred_train = SVR.predict(X_train)
    Y_pred_test  = SVR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print("\nCross validation for SVR regression results:")
print("Cross validation training Error:",np.mean(training_error).round(2))
print("Cross validation test Error:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error)).round(2))

'''
Cross validation for SVR regression results:
Cross validation training Error: 11.44
Cross validation test Error: 12.8
variance: 1.35
'''

#5)========================================Random forest==================================================

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100, max_depth=8,
                        max_samples=0.6,
                        max_features=0.7,
                        min_samples_split=2)
RFR.fit(X_train,Y_train)
Y_pred_train = RFR.predict(X_train)
Y_pred_test  = RFR.predict(X_test)

#metrics
print("\nRFR Regression Model:")
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nTraining Set Performance:")
print("Mean Squared Error:", mse_train.round(2))
print("R-squared Score:", r2_train.round(2))

print("\nTest Set Performance:")
print("Mean Squared Error:", mse_test.round(2))
print("R-squared Score:", r2_test.round(2))

#cross validation for RFR regression results
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFR.fit(X_train,Y_train)
    Y_pred_train = RFR.predict(X_train)
    Y_pred_test  = RFR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print("\nCross validation for RFR regression results:")
print("Cross validation training Error:",np.mean(training_error).round(2))
print("Cross validation test Error:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error)).round(2))

'''
Cross validation for RFR regression results:
Cross validation training Error: 13.54
Cross validation test Error: 22.69
variance: 9.15
'''
