#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    # heaviside activation function
    def activation(self, z):
        return np.heaviside(z,0)
    
    def fit(self, X, y):
        n_features = X.shape[1]
        counts = []
        #initializing weights and bias
        self.weights = np.zeros((n_features))
        self.bias = 0
    
        #Iterating until the number of epochs
        for epoch in range(self.epochs):
            #Traversing through the entire training set
            count = 0
            for i in range(len(X)):
                z = np.dot(X,self.weights) + self.bias
                y_pred = self.activation(z)
                count += y_pred[i]!=y[i]
                self.weights = self.weights + self.learning_rate*(y[i] - y_pred[i])*X[i]
                self.bias = self.bias + self.learning_rate*(y[i] - y_pred[i])
            counts.append(count)
        return self.weights, self.bias, counts

    def predict(self, X):
        z = np.dot(X,self.weights) + self.bias
        return self.activation(z)


# In[98]:


class Val:
    def find_TP(y, y_hat):
       # counts the number of true positives (y = 1, y_hat = 1)
       return sum((y == 1) & (y_hat == 1))
    def find_FN(y, y_hat):
       # counts the number of false negatives (y = 1, y_hat = 0) Type-II error
       return sum((y == 1) & (y_hat == 0))
    def find_FP(y, y_hat):
       # counts the number of false positives (y = 0, y_hat = 1) Type-I error
       return sum((y == 0) & (y_hat == 1))
    def find_TN(y, y_hat):
       # counts the number of true negatives (y = 0, y_hat = 0)
       return sum((y == 0) & (y_hat == 0))


# In[99]:


class Per_mat:
    def __init__(self,y, y_hat):
        self.TP = Val.find_TP(y,y_hat)
        self.FN = Val.find_FN(y, y_hat)
        self.FP = Val.find_FP(y, y_hat)
        self.TN = Val.find_TN(y, y_hat)
        
    def Accuracy(self,y,y_hat):
        return (self.TP+self.TN)/(self.FP+self.TP+self.FN+self.TN)
    
    def Precision(self,y,y_hat):
        return self.TP/(self.TP+self.FP)
    
    def Recall(self,y,y_hat):
        return self.TP/(self.TP+self.FN)
    
    def F1_ab(self, y,y_hat):
        return 2*(1/((1/self.Precision(y,y_hat))+ 1/self.Recall(y,y_hat)))
        


# In[100]:


# Loading dataset
X = np.load("./Dataset-1/inputs_Dataset-1.npy")
y = np.load("./Dataset-1/outputs_Dataset-1.npy")


# In[101]:


#splitting the training and testing sets
X_train = X[int(0.8*len(X)):]
y_train = y[int(0.8*len(y)):]
X_test = X[:int(0.2*len(X))]
y_test = y[:int(0.2*len(y))]


# In[102]:


perceptron = Perceptron(0.001,2000)
weights, bias, misclassified = perceptron.fit(X_train,y_train)

pred = perceptron.predict(X_test)


# In[103]:


perf = Per_mat(y_test, pred)

acc = perf.Accuracy(y_test, pred)
        
precision = perf.Precision(y_test, pred)
        
recall = perf.Recall(y_test, pred)
        
f1 = perf.F1_ab(y_test, pred)

print('Accuracy of 1st model : {}'.format(acc))

print('Precision of 1st model : {}'.format(precision))

print('Recall of 1st model : {}'.format(recall))
    
print('F1_score of 1st model : {}'.format(f1))


# In[104]:


# Variance calculation
def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance , mean


# In[105]:


#Implementing K-Fold cross validation
from sklearn.model_selection import KFold

for k in range(2,11):
    kf = KFold(n_splits=k, random_state = None)
    acc_score = []
    precision_score = []
    recall_score = []
    F1_score = []
    for train_index, test_index in kf.split(X_train):
    #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train1, X_test1 = X[train_index], X[test_index]
        y_train1, y_test1 = y[train_index], y[test_index]

        perceptron.fit(X_train1,y_train1)
        pred_val = perceptron.predict(X_test1)
        perf = Per_mat(y_test1, pred_val)
        
        acc = perf.Accuracy(y_test1, pred_val)
        acc_score.append(acc)
        
        precision = perf.Precision(y_test1, pred_val)
        precision_score.append(precision)
        
        recall = perf.Recall(y_test1, pred_val)
        recall_score.append(recall)
        
        f1 = perf.F1_ab(y_test1, pred_val)
        F1_score.append(f1)

    var_acc_score, avg_acc_score = variance(acc_score)
    print('Avg accuracy of {}th : {}'.format(k,avg_acc_score))
    print('Var accuracy of {}th : {}'.format(k,var_acc_score))
    
    var_pre_score, avg_pre_score = variance(precision_score)
    print('Avg Precision of {}th : {}'.format(k,avg_pre_score))
    print('Var Precision of {}th : {}'.format(k,var_pre_score))
    
    var_rec_score, avg_rec_score = variance(recall_score)
    print('Avg Recall of {}th : {}'.format(k,avg_rec_score))
    print('Var Recall of {}th : {}'.format(k,var_rec_score))
    
    var_f1_score, avg_f1_score = variance(F1_score)
    print('Avg F1_score of {}th : {}'.format(k,avg_f1_score))
    print('Var F1_score of {}th : {}'.format(k,var_f1_score))
    
    print(' ')


# In[107]:


import matplotlib.pyplot as plt

plt.plot(misclassified)
plt.show()


# In[108]:


# For second dataset


# In[109]:


X2 = np.load("./Dataset-2/inputs_Dataset-2.npy")
y2 = np.load("./Dataset-2/outputs_Dataset-2.npy")


# In[110]:


#splitting the training and testing sets
X2_train = X2[int(0.8*len(X2)):]
y2_train = y2[int(0.8*len(y2)):]
X2_test = X2[:int(0.2*len(X2))]
y2_test = y2[:int(0.2*len(y2))]


# In[111]:


perceptron2 = Perceptron(0.001,2000)
weights2, bias2, misclassified2 = perceptron2.fit(X2_train,y2_train)

pred2 = perceptron2.predict(X2_test)


# In[112]:


perf = Per_mat(y2_test, pred2)

acc2 = perf.Accuracy(y2_test, pred2)
        
precision2 = perf.Precision(y2_test, pred2)
        
recall2 = perf.Recall(y2_test, pred2)
        
f12 = perf.F1_ab(y2_test, pred2)

print('Accuracy of 2nd model : {}'.format(acc2))

print('Precision of 2nd model : {}'.format(precision2))

print('Recall of 2nd model : {}'.format(recall2))
    
print('F1_score of 2nd model : {}'.format(f12))


# In[113]:


plt.plot(misclassified2)
plt.show()


# In[114]:


# For Third dataset


# In[115]:


X3 = np.load("./Dataset-3/inputs_Dataset-3.npy")
y3 = np.load("./Dataset-3/outputs_Dataset-3.npy")


# In[116]:


X3_train = X3[int(0.8*len(X3)):]
y3_train = y3[int(0.8*len(y3)):]
X3_test = X3[:int(0.2*len(X3))]
y3_test = y3[:int(0.2*len(y3))]


# In[117]:


perceptron3 = Perceptron(0.001,2000)
weights3, bias3, misclassified3 = perceptron3.fit(X3_train,y3_train)

pred3 = perceptron3.predict(X3_test)


# In[118]:


perf = Per_mat(y3_test, pred3)

acc3 = perf.Accuracy(y3_test, pred3)
        
precision3 = perf.Precision(y3_test, pred3)
        
recall3 = perf.Recall(y3_test, pred3)
        
f13 = perf.F1_ab(y3_test, pred3)

print('Accuracy of 3rd model : {}'.format(acc3))

print('Precision of 3rd model : {}'.format(precision3))

print('Recall of 3rd model : {}'.format(recall3))
    
print('F1_score of 3rd model : {}'.format(f13))


# In[119]:


plt.plot(misclassified3)
plt.show()


# In[ ]:




