import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)
train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns

y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

rowNumber = x_train.shape[0]
weightColumn = x_train
unitMatrix = np.ones((rowNumber,1))
xmatrix = np.hstack((weightColumn.reshape((-1, 1)), unitMatrix))


y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution

y_t=y_train.reshape((-1, 1))
theta_best= np.linalg.inv(xmatrix.T @ xmatrix) @ xmatrix.T @ y_train

print(theta_best)
# TODO: calculate error
m=len(x_train)
def error(setx,sety):
    sum = 0
    m = len(setx)
    for i in range(m):
        prediction = float(theta_best[1]) + float(theta_best[0]) * setx[i]
        temp = (prediction - sety[i]) ** 2
        sum += temp
    return sum / m


MSEtest=error(x_test,y_test)
print("MSE for test ",MSEtest)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[1]) + float(theta_best[0]) * x

plt.plot(x, y, label='Rozwiąznie jawne')
plt.scatter(x_test, y_test, label='Dane testowe', color='red')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Regresja liniowa: MPG w zależności od wagi')
plt.legend()
plt.show()


def standarization(matrixTostandarization,size):
    standarizationMatrix = np.empty(size)
    standardDeviationForX = np.std(matrixTostandarization)
    averageForX = np.mean(matrixTostandarization)
    for i in range(m):
        standarizationMatrix[i] = (matrixTostandarization[i] - averageForX) / standardDeviationForX

    return standarizationMatrix

# TODO: standardization
#
standarizationForX=standarization(x_train,m)
standarizationForY=standarization(y_train,m)

xmatrixStandarization = np.hstack((standarizationForX.reshape((-1, 1)), unitMatrix))

x_test =(x_test-np.mean(x_train))/np.std(x_train)
y_test =(y_test-np.mean(y_train))/np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
learning_rate =0.01
theta_best=np.random.rand(2)

for i in range(1000):
    gradients = 2 / len(x_train) * xmatrixStandarization.T.dot(xmatrixStandarization.dot(theta_best) - standarizationForY )
    theta_best = theta_best - learning_rate * gradients

print(theta_best)

# TODO: calculate error
MSEtestGradient=error(x_test,y_test)
print("MSE for test in gradient method",MSEtestGradient)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[1]) + float(theta_best[0]) * x

plt.plot(x, y)
plt.scatter(x_test, y_test, label='Dane testowe', color='red')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Metoda gradietu prostego : MPG w zależności od wagi')
plt.show()