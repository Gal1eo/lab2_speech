import numpy as np
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors D = 13
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
            means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    A = hmm1['transmat']#4*4
    PI = hmm1['startprob']#1*4
    B = hmm2['transmat']
    P = hmm2['startprob']
    m = A.shape[0] - 1
    m2 = B.shape[0] - 1
    K = m + m2
    A_con = np.zeros((K+1, K+1))
    Pi_con = np.zeros((1, K+1))
    A_con[:m, :m] = A[:m, :m]
    A_con[m:, m:] = B
    A_con[:m, m:] = np.dot(A[:m,m].reshape(-1, 1), P.reshape(1, -1))
    PP = PI.reshape(1, -1)
    Pi_con[0, :m] = PP[0, :m]
    Pi_con[0, m:] = PP[0, m] * P

    twoHMMs = {}
    twoHMMs['startprob'] = Pi_con
    twoHMMs['transmat'] = A_con
    twoHMMs['means'] = np.concatenate((hmm1['means'], hmm2['means']), axis=0)
    twoHMMs['covars'] = np.concatenate((hmm1['covars'] ,hmm2['covars']), axis=0)#K*D

    return twoHMMs

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    gmm_loglik = np.mean(log_emlik)


    return gmm_loglik


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]
    forward_prob = np.zeros((N,M))
    for i in range(N):
        if i == 0:
            forward_prob[0, :] = log_startprob[0, :M] + log_emlik[0, :M]
        else:
            for j in range(M):
                forward_prob[i, j] = logsumexp(forward_prob[i-1] + log_transmat[:M, j]) + log_emlik[i, j]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]
    backward_prob = np.zeros((N, M))
    for n in reversed(range(N-1)):
        for i in range(M):
             backward_prob[n, i] = logsumexp(backward_prob[n + 1, :M] + log_transmat[i, :M] + log_emlik[n + 1, :M])

    return backward_prob


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]
    viterbiarray = np.zeros((N, M))
    for n in range(N):
        if n == 0:
            viterbiarray[n] = log_startprob[0, :M] + log_emlik[0, :M]
        else:
            for i in range(M):
                viterbiarray[n, i] = np.max(viterbiarray[n - 1, :M] + log_transmat[:M, i]) + log_emlik[n, i]

    viterbi_path = np.argmax(viterbiarray, axis=1)
    viterbi_loglik = np.max(viterbiarray[-1, :])

    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N = log_alpha.shape[0]
    M = log_alpha.shape[1]
    log_gamma = np.zeros((N, M))
    for n in range(N):
        log_gamma[n, :] = log_alpha[n, :] + log_beta[n, :] - logsumexp(log_alpha[N-1, :])

    return log_gamma


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N = X.shape[0]#71*13
    D = X.shape[1]
    M = log_gamma.shape[1]
    means = np.zeros((M, D))
    covars = np.zeros((M, D))
    #means = np.mean(X, axis=0)
    #covars = np.cov(X.T)
    for k in range(M):
        for n in range(N):
            means[k, :] += np.exp(log_gamma[n, k]) * X[n,:]
        means[k, :] /= np.sum(np.exp(log_gamma[:, k]))
    for k in range(M):
        for n in range(N):
            covars[k, :] += np.exp(log_gamma[n, k]) * (X[n, :] - means[k, :])**2
        covars[k, :] /= np.sum(np.exp(log_gamma[:, k]))
    covars = np.maximum(covars, varianceFloor)

    return means, covars


