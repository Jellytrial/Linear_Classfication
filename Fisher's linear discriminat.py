@@ -0,0 +1,108 @@
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:40:40 2018

@author: Jelly
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = load_iris()
#Repeat random select training sets and test sets 100 times
count = 0
while count < 100:
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.5, random_state = 1)
    count += 1
print('x train:\n',x_train, '\n', 'y train:\n', y_train)

#calculate mean of samples
def class_mean(data, label, clusters):
    mean_vectors = []
    for cl in range(0, clusters):
        mean_vectors.append(np.mean(data[label == cl], axis = 0))
    print('\nMean Vector class :\n', mean_vectors)
    return mean_vectors

#calculate scatter matrix within class
def within_class_SW(data, label, clusters):
    m = data.shape[1]
    S_W = np.zeros((m, m))
    mean_vectors = class_mean(data, label, clusters)
    for cl, mv in zip(range(0, clusters), mean_vectors):
        class_sc_mat = np.zeros((m, m))
#Matrix multiplication for each sample
        for row in data[label == cl]:
            row, mv = row.reshape(4,1), mv.reshape(4,1)
            class_sc_mat += (row - mv). dot((row-mv).T)
        S_W += class_sc_mat
    print('\nWithin class scatter matrix:\n', S_W)
    return S_W

#calculate scatter matrix between classes
def between_class_SB(data, label, clusters):
    m = data.shape[1]
    all_mean = np.mean(data, axis = 0)
    S_B = np.zeros((m, m))
    mean_vectors = class_mean(data, label, clusters)
    for cl, mean_vec in enumerate(mean_vectors):
        n = data[label == cl+1, :].shape[0]
        mean_vec = mean_vec.reshape(4,1)
        all_mean = all_mean.reshape(4,1)
        S_B += n *(mean_vec - all_mean).dot((mean_vec - all_mean).T)
    print('\nBetween-class Scatter Matrix:\n', S_B)
    return S_B

def lda():
    data, label=x_train, y_train;
    clusters = 3
    S_W = within_class_SW(data,label,clusters)
    S_B = between_class_SB(data,label,clusters)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    print (S_W) 
    print (S_B) 
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(4,1)
        print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
    print ('\nMatrix W:\n', W.real)
    print (data.dot(W))
    return W

def plot_lda():
    data,labels = x_train, y_train
    W = lda()
    Y = data.dot(W)
    print (Y) 
    ax = plt.subplot(111)
    for label, marker, color in zip(range(0,3),('^','s','o'),('blue','red','green')):
        plt.scatter(x=Y[:,0][labels == label],
            y=Y[:,1][labels == label],
            marker = marker,
            color = color,
            alpha = 0.5,
            )
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.legend(['Setosa', 'Versicolor', 'Virginica'], loc = 0)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.show()
  
#calculate accuracy
def sklearnLDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
    lda = LDA(n_components=1)
    ld = lda.fit(x_train, y_train).predict(x_test)
    print('\nPredicted labels:\n',ld)
    print('Actual labels:\n', y_test)
    print('\nAccuracy:', str(lda.score(x_test, y_test)))
    return ld

if __name__ =="__main__":
    lda()
    sklearnLDA()
    plot_lda() 
