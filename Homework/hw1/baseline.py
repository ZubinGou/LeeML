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
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 120 # 120->5.680 125->5.68093  100->5.6809
iter_time = 20000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12) # rmse
    if t%50 == 0:
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
    adagrad += gradient ** 2
    w -= learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
# print(w)

# test
testdata = pd.read_csv('./dataset/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
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