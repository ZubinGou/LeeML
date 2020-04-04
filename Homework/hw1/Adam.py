import sys
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/train.csv', encoding='big5')

# preprocessing
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# extract features 
# 4320 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。
month_data = {} 
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 471 data per month, 12 * 471 total; data: 9*18, output:1*18
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) # vec dim: 18*9
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] # value
# print(x)
# print(y)

# normalize
mean_x = np.mean(x, axis=0) # 18 * 9
std_x = np.std(x, axis=0) # 18 * 9
for i in range(len(x)):  # data count: 12 * 471
    for j in range(len(x[0])): # every data: 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# print(x)

# split data
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
# print('-'*80)
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print('-'*80)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

# train
# Adagrad: 19950:5.680932254430758
# Adam 44800:5.679518334390746
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 0.03
iter_time = 20000
eps = 1e-8
beta_1 = 0.9 # Exponential decay rates for the moment estimates
beta_2 = 0.999 
m = np.zeros([dim, 1]) # Initialize 1st moment vector
v = np.zeros([dim, 1]) # Initialize 2nd moment vector
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / len(x)) # rmse
    if t%50 == 0:
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
    # sliding average
    m = beta_1 * m + (1 - beta_1) * gradient # Update biased first moment estimate
    v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2) # Update biased second raw moment estimate
    m_hat = m / (1 - np.power(beta_1, t + 1)) # Compute bias-corrected first moment estimate
    # bias correction
    v_hat = v / (1 - np.power(beta_2, t + 1)) # Compute bias-corrected second moment estimate
    w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
np.save('weight.npy', w)
# print(w)

# test
testdata = pd.read_csv('./dataset/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data = test_data.where(test_data != 'NR', 0)
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i+1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
# print(test_x)

# prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# print(ans_y)

# save to csv
import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    # print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)