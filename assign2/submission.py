#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {"so":1,"touching":1,"quite":0,"impressive":0,"not":-1,"boring":-1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    ret = dict()
    for w in x.split():
        ret[w] = ret.get(w,0) + 1
    return ret
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    predictor=lambda x:1 if dotProduct(weights,featureExtractor(x))>=0 else -1
    for _ in range(numIters):
        for x,y in trainExamples:
            features=featureExtractor(x)
            sigmoid_=sigmoid(dotProduct(weights,features))
            coef=sigmoid_ if y==-1 else sigmoid_-1
            increment(weights,-eta*coef,features)
        print(f"train score : {evaluatePredictor(trainExamples,predictor)} test score : {evaluatePredictor(testExamples,predictor)}")
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    phi = {}
    splitted_x = x.split()
    for i in range(len(splitted_x)-n+1):
        k = " ".join(splitted_x[i:i+n])
        phi[k] = phi.get(k,0) + 1
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x':-0.5,'mu_y':1.5},{'mu_x':3,'mu_y':1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x':-1,'mu_y':0},{'mu_x':2,'mu_y':2}
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    def dist(a,b):
        ret = 0.0
        for p1,p2 in zip(a.keys(),b.keys()):
            if p1==p2:
                ret = ret + math.pow(b[p1]-a[p1],2)
            else:
                ret = ret + math.pow(a[p1],2) + math.pow(b[p2],2)
        return math.sqrt(ret)
    #centroids = examples[:K]
    centroids = []
    for i in range(K):
        centroids.append({k:v for k,v in examples[i].items()})
    assignments = [-1] * len(examples)
    for _ in range(maxIters):
        loss = 0.0
        analysis = [[] for _ in range(K)]
        for e_idx,e in enumerate(examples):
            min_dist = float("inf")
            for c_idx,c in enumerate(centroids):
                ret = dist(c,e)
                if min_dist>ret:
                    min_dist=ret
                    assignments[e_idx] = c_idx
            analysis[assignments[e_idx]].append(e)
            loss = loss + min_dist
        print(f"reconstruction loss : {loss}")
        new_centroids = [{} for _ in range(K)]
        for i in range(K):
            l = len(analysis[i])
            for point in analysis[i]:
                increment(new_centroids[i],1/l,point)
        if new_centroids == centroids:
            break
        else:
            centroids = new_centroids
    return centroids,assignments,loss
    # END_YOUR_ANSWER