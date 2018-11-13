#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.

     => l2
    """

    ### YOUR CODE HERE
    # [n, m]
    # print(x.shape)
    # (n, )
    # denom 分母项
    denom = np.apply_along_axis(lambda x: np.sqrt(np.dot(x, x)), 1, x)
    # print(denom)
    # print(np.sqrt(np.dot(x, x)))
    # np.expand_dims(denom, axis=1) == denom[:,None]
    x = x / np.expand_dims(denom, axis=1)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # [dim]
    v_hat = predicted
    # [n, dim]
    u_o = outputVectors
    # [n] => softmax => [n]
    p = softmax(np.dot(u_o, v_hat))

    # J(theta)
    cost = -np.log(p[target])

    # grad for u_o
    # (2)sum(y)
    # np.dot(np.expand_dims(p, 0), u_o) == np.dot(u_o.T, p)
    z = p.copy()
    z[target] -= 1.0
    gradPred = -u_o[target] + np.dot(np.expand_dims(p, 0), u_o)
    gradPred = gradPred.squeeze(axis=0)
    # gradPred = np.reshape(gradPred, [gradPred.shape[1]])
    # version 2
    # [n, dim] [n]
    # gradPred = np.dot(u_o.T, z)
    # grad for u_k
    # (y_hat-1)*v_hat => [n, dim]
    grad = np.outer(z, v_hat)
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]

    # indices[0]=target, indices[1--K] = not target
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    u_o = outputVectors[indices[0]]
    v_c = predicted
    # [dim]
    z = sigmoid(np.dot(u_o.T, v_c))
    cost = - np.log(z)
    gradPred = np.zeros(np.shape(predicted))
    gradPred += (z-1.0) * u_o
    grad = np.zeros(np.shape(outputVectors))
    # 按位乘 [1,2,3]*[1,2,3] = [1,4,9]
    # 对u_o求偏导
    grad[target] += (z-1.0)*v_c

    # for negative samples
    for k in range(K):
        u_k = outputVectors[indices[k+1]]
        z = sigmoid(np.dot(u_k.T, v_c))
        cost -= np.log(1.0-z)
        gradPred += z*u_k
        grad[indices[k+1]] += z*v_c
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    v_id = tokens[currentWord]
    v_hat = inputVectors[v_id]

    for word in contextWords:
        u_id = tokens[word]
        # 求导
        cost_, gradPred, grad = word2vecCostAndGradient(v_hat, u_id, outputVectors, dataset)
        cost += cost_
        # 存储有更新的词向量的偏导数
        gradIn[v_id] += gradPred
        gradOut += grad

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    # bug python2.7 => 10/2=5, python3.6 => 10/2=5.0
    # 所以使用floor除法替代  3//2 = 1
    inputVectors = wordVectors[:N//2, :]
    outputVectors = wordVectors[N//2:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        # 获取中心词和上下文
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    # 一个参数返回对象类型, 三个参数，返回新的类型对象
    # 返回一个dummy类
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    # 定义类的方法
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    # 每个单词随机生成一个词向量(行)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    # 生成词表=>每个单词和索引
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    # f, x = > f(x) return (f'(x),grad)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
