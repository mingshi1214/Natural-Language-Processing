from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp
import sys

dataDir = "/u/cs401/A3/data/"
random.seed(3)
np.random.seed(3)

class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))
        self.precompute = None

    def reset_preCompute(self):

        const = self._d/2 * np.log(2 * np.pi)

        precomputed = const + np.sum(np.power(self.mu, 2)/(2* self.Sigma)) + np.sum(np.log(self.Sigma))/2
        self.precompute = precomputed

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """

        const = self.mu.shape[1]/2 * np.log(2 * np.pi)

        self.precompute = const + np.sum(np.power(self.mu[m], 2)/(2* self.Sigma[m])) + np.sum(np.log(self.Sigma[m]))/2
        
        return self.precompute


    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    mu = myTheta.mu[m]
    sigma = myTheta.Sigma[m]

    if len(x.shape)==1:
        axis = 0
    else:
        axis = 1

    temp = 0.5*np.square(x) - (mu * x)
    return -np.sum(temp/sigma, axis = axis) - myTheta.precomputedForM(m)



def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    omegas = myTheta.omega
    log_omeg = np.log(omegas)

    all_logs = log_Bs + log_omeg

    log_pmx = np.zeros(log_Bs.shape)

    lse = logsumexp(all_logs, axis = 0, keepdims = True)

    # probably can just do all_logs - lse for log_pmx 
    for m in range(0, log_Bs.shape[0]):
        omega_m = myTheta.omega[m]
        log_p_num = np.log(omega_m) + log_Bs[m,:]
        log_pmx[m,:] = log_p_num - lse

    return log_pmx


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    omegas = myTheta.omega
    log_omeg = np.log(omegas)

    all_logs = log_Bs + log_omeg

    lse = logsumexp(all_logs, axis = 0, keepdims = True)

    log_lik = lse.sum()

    return log_lik

def updateTheta(myTheta, X, log_probs):
    #update omega, mu, and sigma in myTheta

    # print("shape of log_probs, ", log_probs.shape)
    #update omega
    probs = np.exp(log_probs)
    M = log_probs.shape[0]
    T = len(X)

    mu_reset = probs.dot(X) / (probs.sum(axis = 1, keepdims = True))
    myTheta.reset_mu(mu_reset)

    # instead of log_probs.shape[1] can just write T
    omeg_reset = (np.sum(probs, axis = 1)/log_probs.shape[1]).reshape((M, 1))
    myTheta.reset_omega(omeg_reset)

    sig_reset = (probs.dot(np.power(X, 2)) / (probs.sum(axis = 1, keepdims = True)))
    sig_reset = sig_reset - np.power(myTheta.mu, 2)
    myTheta.reset_Sigma(sig_reset)

    return myTheta


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    # experiments
    # maxIter = 20
    # M = 5

    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    # print("TODO : Initialization")
    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data)
    # myTheta.reset_Sigma(some_appropriate_sigma)

    omeg_const = np.ones(myTheta.omega.shape)/M
    myTheta.reset_omega(omeg_const)

    myTheta.reset_mu(X[np.array(random.sample(range(X.shape[0]), M))])

    sig = np.ones((M, X.shape[1]))
    myTheta.reset_Sigma(sig)

    # print("TODO: Rest of training")
    iteration = 0
    lik_prev = - float("inf")
    dif = float("inf")

    while (iteration < maxIter) and (dif >= epsilon):
        myTheta.precompute = None
        # compute logbs and logprobs
        log_bs = np.array([log_b_m_x(m, X, myTheta) for m in range(0, M, 1)])
        log_probs = np.array(log_p_m_x(log_bs, myTheta))

        # compute likelihood
        lik_curr = logLik(log_bs, myTheta)

        # update parameters
        myTheta = updateTheta(myTheta, X, log_probs)

        dif = lik_curr - lik_prev
        lik_prev = lik_curr
        iteration = iteration + 1

        print(f"Iteration {iteration} likelihood of {round(lik_curr, 4)} and dif of {round(dif, 4)}")
    myTheta.precompute = None
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    # print("TODO")

    # store log likelihoods in dict
    # dict key will be likelihood and value will be the model to find best model later
    logLiks = {}

    # list to store the logLikelihoods
    logLiks_list = []

    M = len(models[0].omega)

    i = 0
    for model in models:
        log_bs = np.array([log_b_m_x(m, mfcc, model) for m in range(0, M, 1)])
        
        # find the log likelihood of this model
        logLik_curr = logLik(log_bs, model)
        #print("testing model: ", model.name, " loglik: ", logLik_curr)

        # add loglik to the list and the dictionary
        if not (isinstance(logLik_curr, float) and np.isnan(logLik_curr)):
            logLiks_list.append(logLik_curr)
            logLiks[logLik_curr] = (model, i)

        i += 1
            
    # sort the list of logLiks
    sorted_logLiks = sorted(logLiks_list, key = lambda x: x, reverse = True)

    # find best model
    _, best_ind = logLiks[sorted_logLiks[0]]
    bestModel = best_ind

    # write to file
    if k > 0:
        k = min(k, len(models))
        print(models[correctID].name)
        for i in range(0, k, 1):
            print(logLiks[sorted_logLiks[i]][0].name, " ", sorted_logLiks[i], "\n")

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    random.seed(3)
    trainThetas = []
    testMFCCs = []
    # print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    
    stdout = sys.stdout  # steal stdout so that we can redirect to file.
    print(f"accuracy: {accuracy}")
    sys.stdout = open('gmmLiks.txt', 'w')
    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0*numCorrect/len(testMFCCs)
    sys.stdout = stdout