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

    return means, covars


if __name__ == "__main__":
    data = np.load('lab2_data.npz')['data']
    phoneHMMs = np.load('lab2_models_onespkr.npz')['phoneHMMs'].item()
    phoneHMMs_all = np.load('lab2_models_all.npz')['phoneHMMs'].item()
    """4"""
    '''
    hmm1 = phoneHMMs['ah']
    hmm2 = phoneHMMs['ao']
    twohmm= concatTwoHMMs(hmm1, hmm2)
    '''
    """5 HMM Likelihood and Recognition"""
    """
    5.1
    """
    example = np.load('lab2_example.npz')['example'].item()
    isolated = {}
    for digit in prondict.keys():
        isolated[digit] = ['sil'] + prondict[digit] + ['sil']
    wordHMMs = {}
    wordHMMs_all = {}
    #example
    for digit in prondict.keys():
        wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    lpr = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    diff = example['obsloglik'] - lpr  # 0
    # for 11 digits
    for digit in prondict.keys():
        wordHMMs_all[digit] = concatHMMs(phoneHMMs_all, isolated[digit])
    #same digit 'o'
    lpr_o = log_multivariate_normal_density_diag(data[22]['lmfcc'], wordHMMs_all['o']['means'], wordHMMs_all['o']['covars'])
    '''
    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(lpr.T)
    plt.title('example "o" ')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.pcolormesh(lpr_o.T)
    plt.title('test "o" from data22')
    plt.colorbar()
    plt.show()
    '''
    """
    5.2
    """
    lalpha = forward(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    diff1 = example['logalpha'] - lalpha #0
    # log-likelihood
    loglike = logsumexp(lalpha[-1])
    diff0 = example['loglik'] - loglike #0

    # score all the 44 utterances in the data array with each of the 11 HMM
    # models in wordHMMs.
    scores_1 = np.zeros((44, 11))
    scores_2 = np.zeros((44, 11))
    labels_ori = []
    labels_pre = []
    labels_pre2 = []
    keys = list(prondict.keys())
    acc_1 = 0
    acc_2 = 0
    '''
    for i in range(44):
        for j, key in enumerate(keys):
            lpr = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs_all[key]['means'],
                                                         wordHMMs_all[key]['covars'])
            alpha = forward(lpr, np.log(wordHMMs_all[key]['startprob']), np.log(wordHMMs_all[key]['transmat']))
            scores_2[i, j] = logsumexp(alpha[-1])
            lpr_1 = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs[key]['means'],
                                                       wordHMMs[key]['covars'])
            alpha_1 = forward(lpr_1, np.log(wordHMMs[key]['startprob']), np.log(wordHMMs[key]['transmat']))
            scores_1[i, j] = logsumexp(alpha_1[-1])
        ori = data[i]['digit']
        pre_1 = keys[int(np.argmax(scores_1[i, :]))]
        pre_2 = keys[int(np.argmax(scores_2[i, :]))]
        #labels_ori.append(ori)
        labels_pre.append(pre_1)
        labels_pre2.append(pre_2)
        if ori == pre_1:
            acc_1 += 1
        if ori == pre_2:
            acc_2 += 1
    print("Accuracy(trained on all speakers): {0}; Accuracy(trained on one speaker):{1} ".format(acc_2, acc_1))
    print(labels_pre, labels_pre2)
    '''

    """
    5.3 Viterbi
    """
    viterbi_loglik, viterbi_path = viterbi(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    #plt.pcolormesh(lalpha.T)
    #plt.plot(viterbi_path,'r')
    #plt.show()
    diff3 = example['vloglik'] - viterbi_loglik.T  # 0

    # Score all 44 utterances in the data with each of the 11 HMM models in wordHMMs
    '''
    for i in range(44):
        for j, key in enumerate(keys):
            lpr = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs_all[key]['means'],
                                                       wordHMMs_all[key]['covars'])
            viterbi_2, viterbi_path_2 = viterbi(lpr, np.log(wordHMMs_all[key]['startprob']), np.log(wordHMMs_all[key]['transmat']))
            scores_2[i, j] = viterbi_2
            lpr_1 = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs[key]['means'],
                                                         wordHMMs[key]['covars'])
            viterbi_1, viterbi_path_1 = viterbi(lpr_1, np.log(wordHMMs[key]['startprob']), np.log(wordHMMs[key]['transmat']))
            scores_1[i, j] = viterbi_1
        ori = data[i]['digit']
        pre_1 = keys[int(np.argmax(scores_1[i, :]))]
        pre_2 = keys[int(np.argmax(scores_2[i, :]))]
        #labels_ori.append(ori)
        labels_pre.append(pre_1)
        labels_pre2.append(pre_2)
        if ori == pre_1:
            acc_1 += 1
        if ori == pre_2:
            acc_2 += 1
    print("Accuracy(trained on all speakers): {0}; Accuracy(trained on one speaker):{1} ".format(acc_2, acc_1))
    print(labels_pre, labels_pre2)
    '''

    """
    5.4
    """
    lbeta = backward(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    diff2 = example['logbeta'] - lbeta
    # log-likelihood
    loglike = logsumexp(lbeta[0])
    diff4 = example['loglik'] - loglike  # 0

    """6 HMM Retraining(emission probability distributions)"""
    """6.1"""
    lgamma = statePosteriors(lalpha, lbeta)
    #print(np.sum(np.exp(lgamma), axis=1)) #=1
    N = lgamma.shape[0]
    K = 9
    lgamma_gmm = np.zeros((N, K))
    total = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'],
                                         wordHMMs['o']['covars'])
    for k in range(K):
        lgamma_gmm[:, k] = 1/K * total[:, k] / np.sum(total[:, k])
    plt.subplot(2,1, 1)
    plt.pcolormesh(lgamma_gmm.T)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.pcolormesh(lgamma.T)
    plt.colorbar()
    plt.show()
    print(np.sum(np.exp(lgamma), axis=0))
    print(np.sum(np.sum(np.exp(lgamma))))#=length of obs sequence/time steps

    """6.2"""
    #wordHMMs['4'] =
    means, covars = updateMeanAndVar(example['lmfcc'], lgamma)
    #print('means', means)
    #print('covars', covars)
