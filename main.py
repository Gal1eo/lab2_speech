import numpy as np
import matplotlib.pyplot as plt
from prondict import prondict
from lab2_tools import *
from lab2_proto import *
import sklearn.mixture as mixture

def maintask(task):
    data = np.load('lab2_data.npz')['data']
    phoneHMMs = np.load('lab2_models_onespkr.npz')['phoneHMMs'].item()
    phoneHMMs_all = np.load('lab2_models_all.npz')['phoneHMMs'].item()
    if task == '4':
        hmm1 = phoneHMMs['ah']
        hmm2 = phoneHMMs['ao']
        twohmm = concatTwoHMMs(hmm1, hmm2)

    """5 HMM Likelihood and Recognition"""
    example = np.load('lab2_example.npz')['example'].item()
    isolated = {}
    for digit in prondict.keys():
        isolated[digit] = ['sil'] + prondict[digit] + ['sil']
    wordHMMs = {}
    wordHMMs_all = {}
    for digit in prondict.keys():
        wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    # for 11 digits
    for digit in prondict.keys():
        wordHMMs_all[digit] = concatHMMs(phoneHMMs_all, isolated[digit])
    # example
    lpr = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    diff = example['obsloglik'] - lpr  # 0
    # same digit 'o'
    lpr_o = log_multivariate_normal_density_diag(data[22]['lmfcc'], wordHMMs_all['o']['means'],
                                                 wordHMMs_all['o']['covars'])
    if task == '5.1':
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

    """
    5.2
    """
    lalpha = forward(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    diff1 = example['logalpha'] - lalpha  # 0
    # log-likelihood
    loglike = logsumexp(lalpha[-1])
    diff0 = example['loglik'] - loglike  # 0

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
    if task == '5.2':
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

    """
    5.3 Viterbi
    """
    viterbi_loglik, viterbi_path = viterbi(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))

    if task == '5.3':
        plt.pcolormesh(lalpha.T)
        plt.plot(viterbi_path,'r')
        plt.title('alpha array overlaid with best path obtained by Viterbi decoding ')
        plt.colorbar()
        plt.show()
        diff3 = example['vloglik'] - viterbi_loglik.T  # 0

        # Score all 44 utterances in the data with each of the 11 HMM models in wordHMMs
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

    """
    5.4
    """
    lbeta = backward(lpr, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    diff2 = example['logbeta'] - lbeta
    # log-likelihood
    loglike = logsumexp(lbeta[0])
    diff4 = example['loglik'] - loglike  # 0
    if task == '5.4':
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.pcolormesh(lbeta)
        plt.title('log-beta')
        plt.subplot(1, 3, 2)
        plt.pcolormesh(example['logbeta'])
        plt.title('example')
        plt.subplot(1, 3, 3)
        plt.pcolormesh(example['logalpha'])
        plt.title('log-alpha')
        plt.show()

    """6 HMM Retraining(emission probability distributions)"""
    """
    6.1
    """
    lgamma = statePosteriors(lalpha, lbeta)
    N = lgamma.shape[0]
    K = 9
    lgamma_gmm = np.zeros((N, K))
    total = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'],
                                                 wordHMMs['o']['covars'])
    if task == '6.1':
        print('HMM posteriors')
        print('each time step sum along state axis', np.sum(np.exp(lgamma), axis=1)) #=1
        print('each state sum along time axis', np.sum(np.exp(lgamma)/71, axis=0))
        print('sum over both states and time steps',
              np.sum(np.sum(np.exp(lgamma))))  # =length of obs sequence/time steps
        print('length of observation sequence', lalpha.shape[0])
        print('GMM posteriors')
       # for k in range(K):
            #lgamma_gmm[:, k] = 1 / K * total[:, k] / np.sum(total[:, k])
        gmm = mixture.GaussianMixture(n_components=9)
        gmm.fit(example['lmfcc'])
        gmm_post = gmm.predict_proba(example['lmfcc'])
        plt.subplot(2, 1, 1)
        plt.pcolormesh(gmm_post.T)
        plt.title('GMM posteriors')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.pcolormesh(lgamma.T)
        plt.title('HMM posteriors')
        plt.colorbar()
        plt.show()


    """6.2"""
    if task == '6.2':
        plt.figure()
        L = {}
        for d in prondict:
            # initialization
            log_pi = np.log(wordHMMs_all[d]['startprob'])
            log_tr = np.log(wordHMMs_all[d]['transmat'])
            means = wordHMMs_all[d]['means']
            covars = wordHMMs_all[d]['covars']
            l = []
            # repitation:
            for i in range(20):
                lpr = log_multivariate_normal_density_diag(data[10]['lmfcc'],means, covars)
                # Expectation
                lalpha = forward(lpr,log_pi,log_tr)
                lbeta = backward(lpr,log_pi,log_tr)
                log_gamma = statePosteriors(lalpha,lbeta)
                # Maximization
                means, covars = updateMeanAndVar(data[10]['lmfcc'], log_gamma)
                # Estimate likelihood
                log_like = logsumexp(lalpha[-1])
                if i > 2 and log_like - l[-1] < 0.1:
                    l.append(log_like)
                    L[d] = l
                    break
                else:
                    l.append(log_like)
                    L[d] = l

            plt.plot(l, label = d)

        plt.legend()
        plt.title('log-likelihood (data[10] with different wordHMMs)')
        plt.show()


if __name__ == "__main__":
    maintask('6.1')
