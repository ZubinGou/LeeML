import numpy as np

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

## parse csv to numpy
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio=0.25):
    train_size = int(len(X) * (1- dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# normalize
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# split data
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


## some useful functions 
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

# functions about gradient and loss
def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


## train
w = np.zeros((data_dim,))
b = np.zeros((1,))

max_iter = 30
batch_size = 8 # 8=0.88636 20=8858
learning_rate = 0.1 # 0.2=0.8856 0.1=0.88636

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# num of parameter updates
step = 1

# training
for epoch in range(max_iter):
    X_train, Y_train = _shuffle(X_train, Y_train)

    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size: (idx+1) * batch_size]
        Y = Y_train[idx * batch_size: (idx+1) * batch_size]

        w_grad, b_grad = _gradient(X, Y, w, b)

        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step += 1
    
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

## plotting loss and accuracy curve
import matplotlib.pyplot as plt

# loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# predicting testing labels
predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write(f'{i},{label}\n')

# print most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])


# ## probabilistic generative model
# # Parse csv files to numpy array
# with open(X_train_fpath) as f:
#     next(f)
#     X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
# with open(Y_train_fpath) as f:
#     next(f)
#     Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
# with open(X_test_fpath) as f:
#     next(f)
#     X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

# # Normalize training and testing data
# X_train, X_mean, X_std = _normalize(X_train, train = True)
# X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

# # Compute in-class mean
# X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
# X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

# mean_0 = np.mean(X_train_0, axis = 0)
# mean_1 = np.mean(X_train_1, axis = 0)  

# # Compute in-class covariance
# cov_0 = np.zeros((data_dim, data_dim))
# cov_1 = np.zeros((data_dim, data_dim))

# for x in X_train_0:
#     cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
# for x in X_train_1:
#     cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# # Shared covariance is taken as a weighted average of individual in-class covariance.
# cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# # Compute inverse of covariance matrix.
# # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
# u, s, v = np.linalg.svd(cov, full_matrices=False)
# inv = np.matmul(v.T * 1 / s, u.T)

# # Directly compute weights and bias
# w = np.dot(inv, mean_0 - mean_1)
# b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
#     + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

# # Compute accuracy on training set
# Y_train_pred = 1 - _predict(X_train, w, b)
# print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

# # Predict testing labels
# predictions = 1 - _predict(X_test, w, b)
# with open(output_fpath.format('generative'), 'w') as f:
#     f.write('id,label\n')
#     for i, label in  enumerate(predictions):
#         f.write('{},{}\n'.format(i, label))

# # Print out the most significant weights
# ind = np.argsort(np.abs(w))[::-1]
# with open(X_test_fpath) as f:
#     content = f.readline().strip('\n').split(',')
# features = np.array(content)
# for i in ind[0:10]:
#     print(features[i], w[i])
