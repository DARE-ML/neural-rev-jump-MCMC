 # Rohitash Chandra, 2018 c.rohitash@gmail.com
# rohitash-chandra.github.io

# !/usr/bin/python

# built using: https://github.com/rohitash-chandra/VanillaFNN-Python

# built on: https://github.com/rohitash-chandra/ensemble_bnn

# Dynamic time series prediction: https://arxiv.org/abs/1703.01887 (Co-evolutionary multi-task learning for dynamic time series prediction
#Rohitash Chandra, Yew-Soon Ong, Chi-Keong Goh)


# Multi-task learning via Bayesian Neural Networks for Dynamic Time Series Prediction


import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

import scipy
from scipy import stats


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, MaxTime, MinPer, prob_type):

        self.Top = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime  # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate = 0  # will be updated later with BP call

        self.momenRate = 0

        self.minPerf = MinPer
        # initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()
        self.W1 = np.zeros((self.Top[0], self.Top[1])  )
        self.B1 = np.zeros(self.Top[1])    # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.zeros((self.Top[1], self.Top[2]) )
        self.B2 = np.zeros(self.Top[2])    # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.hidout = np.zeros((self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((self.Top[2]))  # output last layer
        self.prob = prob_type

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, out):
        prob = np.exp(out)/np.sum(np.exp(out))
        return prob

    def printNet(self):
        print(self.Top)
        print(self.W1)

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer#
        z2 = self.hidout.dot(self.W2) #- self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):

        if self.prob == 'classification':  # ensure that one-hot encoding is used in data - for classification problems
            onehot = np.zeros((desired.size, self.Top[2]))
            onehot[np.arange(desired.size),int(desired)] = 1
            desired = onehot.flatten()
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = np.zeros(self.Top[2])
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

    # update weights and bias
        layer = 1  # hidden to output
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in range(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden

        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in range(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]


    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2


    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

        self.total_weightsbias = w_layer1size + w_layer2size + self.Top[1] + self.Top[2]

    def decode_ESPencoding(self, w):
        layer = 0
        gene = 0



        for neu in range(0, self.Top[1]):
            for row in range(0, self.Top[layer]): # for input to each hidden neuron weights
                self.W1[row][neu] = w[gene]
                gene = gene+1

            self.B1[neu] = w[gene]
            gene = gene + 1



            for row in range(0, self.Top[2]): #for each hidden neuron to output weight
                self.W2[neu][row] = w[gene]
                gene = gene+1



        #self.B2[0] = w[gene] # for bias in second layer (assumes one output - change if more than one)


    def net_size(self, netw):
        return ((netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] ) #+ netw[2]



    def decode_MTNencodingX(self, w, mtopo, subtasks):


        position = 0

        Top1 = mtopo[0]


        for neu in range(0, Top1[1]): #in each neuron of hidden layer
            for row in range(0, Top1[0]):
                self.W1[row, neu] = w[position] 
                #print neu, row, position, '    -----  a '
                position = position + 1
            self.B1[neu] = w[position]
            #print neu,   position, '    -----  b '
            position = position + 1

        for neu in range(0, Top1[2]):
            for row in range(0, Top1[1]):
                self.W2[row, neu] = w[position]
                #print neu, row, position, '    -----  c '
                position = position + 1


        if subtasks >=1:


            #for step  in range(1, subtasks+1   ):
            for step  in range(1, subtasks   ):

                TopPrev = mtopo[step-1] #mtopo = [[Inp, Hidd, Out] of model m-1, [Inp, Hid, Out] of model m]
                TopG = mtopo[step]
                Hid = TopPrev[1]
                Inp = TopPrev[0]


                layer = 0

                for neu in range(Hid , TopG[layer + 1]      ) : #neu from hidden neu of model m-1 to hidden neu of model m
                    for row in range(0, TopG[layer]   ): #neu from input layer of model m
                        #print neu, row, position, '    -----  A '

                        self.W1[row, neu] = w[position] #all the W1 from both model m and m-1 -> we store them into a big list of params w
                        position = position + 1

                    self.B1[neu] = w[position]
                    #print neu,   position, '    -----  B '
                    position = position + 1

                diff = (TopG[layer + 1] - TopPrev[layer + 1]) # just the diff in number of hidden neurons between subtasks

                for neu in range(0, TopG[layer + 1]- diff):  # % #neurons in hidden layer of model m-1
                    for row in range(Inp , TopG[layer]): #neu in input layer of model m-1 to input layer of model m
                        #print neu, row, position, '    -----  C '
                        self.W1[row, neu] = w[position]
                        position = position + 1

                layer = 1

                for neu in range(0, TopG[layer + 1]):  # %
                    for row in range(Hid , TopG[layer]):
                        #print neu, row, position, '    -----  D '
                        self.W2[row, neu] = w[position]
                        position = position + 1
                    self.B2[neu] = w[position]
                    #print neu,   position, '    -----  B '
                    position = position + 1
                #print w
                #print self.W1
                #print self.B1
                #print self.W2

    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def evaluate_proposal(self,  data, w , mtopo, subtasks):  # BP with SGD (Stocastic BP)

        #self.decode(w)  # method to decode w into W1, W2, B1, B2.
        self.decode_MTNencodingX(w, mtopo, subtasks)

        size = data.shape[0]
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        fx = np.zeros(size)

        prob = np.zeros((size,self.Top[2]))

        for pat in range(0, size):
            Input[:] =  data[pat, 0:self.Top[0]]
            self.ForwardPass(Input)

            if self.prob == 'classification':
                fx[pat]  = np.argmax(self.out)  
                prob[pat] = self.softmax(self.out)
            else:
                fx[pat] = self.out
        return fx, prob




# -------

# --------------------------------------------------------------------------------------------------------


class BayesNN:  # Multi-Task leaning using Stocastic GD

    def __init__(self, mtaskNet, traindata, testdata, samples, minPerf,   num_subtasks, prob_type):
        # trainData and testData could also be different datasets. this example considers one dataset

        self.traindata = traindata
        self.testdata = testdata
        self.samples = samples
        self.minCriteria = minPerf
        self.subtasks = num_subtasks  # number of network modules
        # need to define network toplogies for the different tasks.

        self.mtaskNet = mtaskNet #[psi_m]

        self.trainTolerance = 0.20  # [eg 0.15 output would be seen as 0] [ 0.81 would be seen as 1]
        self.testTolerance = 0.49
        self.prob = prob_type

    def rmse(self, predictions, targets):

        return np.sqrt(((predictions - targets) ** 2).mean())

    def accuracy(self,pred,actual ):
        count = 0
        for i in range(pred.shape[0]):
            if pred[i] == actual[i]:
                count+=1 

        return 100*(count/pred.shape[0])

    def net_size(self, netw):
        return ((netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] ) #+ netw[2]

    def loglikelihood(self, neuralnet, y, data, w, tausq, subtasks, topo): 

        if self.prob == 'regression':
            [log_lhood, prediction, perf] = self.gaussian_loglikelihood(neuralnet, y, data, w, tausq, subtasks)
        elif self.prob == 'classification':
            [log_lhood, prediction, perf] = self.multinomial_loglikelihood(neuralnet, data, w, subtasks, topo)

        return [log_lhood, prediction, perf]

    def prior(self, sigma_squared, nu_1, nu_2, w, tausq, topo): 

        if self.prob == 'regression':
            logprior = self.prior_regression(sigma_squared, nu_1, nu_2, w, tausq, topo)
        elif self.prob == 'classification':
            logprior = self.prior_classification(sigma_squared, nu_1, nu_2, w, topo)

        return logprior

    def gaussian_loglikelihood(self, neuralnet, y, data, w, tausq, subtasks): 

        fx, prob = neuralnet.evaluate_proposal(data, w, self.mtaskNet, subtasks) #ignore prob
        rmse = self.rmse(fx, y) 

        n = y.shape[0]  # will change for multiple outputs (y.shape[0]*y.shape[1])
        log_lhood = -n/2 * np.log(2 * math.pi * tausq) - (1/(2*tausq)) * np.sum(np.square(y - fx))
        return [log_lhood, fx, rmse]

    def prior_regression(self, sigma_squared, nu_1, nu_2, w, tausq, topo): # for weights and biases and tausq
        h = topo[1]  # number hidden neurons
        d = topo[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss


    def multinomial_loglikelihood(self, neuralnet, data, w, subtasks, topo):
        y = data[:, topo[0]]
        fx, prob = neuralnet.evaluate_proposal(data, w, self.mtaskNet, subtasks)
        acc= self.accuracy(fx,y)
        z = np.zeros((data.shape[0],topo[2]))
        lhood = 0
        for i in range(data.shape[0]):
            for j in range(topo[2]):
                if j == y[i]:
                    z[i,j] = 1
                lhood += z[i,j]*np.log(prob[i,j])

        return [lhood, fx, acc]

    def prior_classification(self, sigma_squared, nu_1, nu_2, w, topo): # for weights and biases only
        h = topo[1]  # number hidden neurons
        d = topo[0]  # number input neurons
        part1 = -1 * ((d * h + h + topo[2]+h*topo[2]) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2
        return log_loss


    def mcmc_sampler(self):

    # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples


        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)


        Netlist = [None] * 10  # create list of Network objects ( just max size of 10 for now )

        samplelist = [None] * samples  # create list of Network objects ( just max size of 10 for now )

        rmsetrain = np.zeros(self.subtasks)
        rmsetest  = np.zeros(self.subtasks)
        trainfx = np.random.randn(self.subtasks, trainsize)
        testfx = np.random.randn(self.subtasks, testsize)

        netsize = np.zeros(self.subtasks, dtype=np.int)


        depthSearch = 5  # declare



#size of each subtask

        for n in range(0, self.subtasks):
            Netlist[n] = Network(self.mtaskNet[n], self.traindata, self.testdata, depthSearch, self.minCriteria, self.prob)
            netw = Netlist[n].Top
            netsize[n] =  self.net_size(netw)  # num of weights and bias
            print(netsize[n])


        y_test = self.testdata[:, netw[0]]  #grab the actual predictions from dataset
        y_train = self.traindata[:, netw[0]]

        w_pos = np.zeros((samples, self.subtasks, netsize[self.subtasks-1]))  # 3D np array

        posfx_train = np.zeros((samples, self.subtasks, trainsize))
        posfx_test = np.zeros((samples, self.subtasks, testsize))

        posrmse_train = np.zeros((samples, self.subtasks))
        posrmse_test = np.zeros((samples, self.subtasks))

        pos_tau = np.zeros(samples)

        print(posrmse_test)
        print(posfx_test)
        print(pos_tau, ' pos_tau')

        w = np.random.randn( self.subtasks, netsize[self.subtasks-1]) #theta_m = [theta_m-1, psi_m]

        w_pro =  np.random.randn( self.subtasks, netsize[self.subtasks-1])

        step_w = 0.05;  # defines how much variation you need in changes to w
        step_eta = 0.01

        print('evaluate Initial w')

        if self.prob == 'regression':
            pred_train, prob = Netlist[0].evaluate_proposal(self.traindata, w[0,:netsize[0]], self.mtaskNet, 0)  # we only take prior calculation for first ensemble, since we have one tau value for all the ensembles.
            eta = np.log(np.var(pred_train - y_train))
            tau_pro = np.exp(eta)

        else: # not used in case of classification
            pred_train, prob = Netlist[0].evaluate_proposal(self.traindata, w[0,:netsize[0]], self.mtaskNet, 0)  # we only take prior calculation for first ensemble, since we have one tau value for all the ensembles.

            eta = 0
            tau_pro = 0

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0



        likelihood = np.zeros(self.subtasks)
        likelihood_pro = np.zeros(self.subtasks)

        prior_likelihood = np.zeros(self.subtasks)

        prior_pro = np.zeros(self.subtasks)

        for n in range(0, self.subtasks):
            prior_likelihood[n] = self.prior(sigma_squared, nu_1, nu_2, w[n,:netsize[0]], tau_pro,  Netlist[n].Top)  # takes care of the gradients

        mh_prob = np.zeros(self.subtasks)

        for s in range(0, self.subtasks-1):
            #calc likelihood and pos of subtask m-1
           [likelihood[s],  posfx_train[0, s,:],  posrmse_train[0, s]] = self.loglikelihood(Netlist[s], y_train, self.traindata, w[s,:netsize[s]], tau_pro, s, Netlist[s].Top)
           # use that of subtask m-1 for subtask m
           w[s + 1, :netsize[s]] = w[s, :netsize[s]]

        s = self.subtasks - 1
        [likelihood[s], posfx_train[0, s, :], posrmse_train[0, s]] = self.loglikelihood(Netlist[s], y_train, self.traindata, w[s,:netsize[s]], tau_pro, s, Netlist[s].Top)


        naccept = 0

    ## PROPOSE FOR PARAMS of MODEL M theta_m

        for i in range(1, samples-1):    # ---------------------------------------

            for s in range(0, self.subtasks):
                w_pro[s, :netsize[s]] = w[s, :netsize[s]] + np.random.normal(0, step_w, netsize[s])

            if self.prob == 'regression': 
                eta_pro = eta + np.random.normal(0, step_eta, 1)
                tau_pro = math.exp(eta_pro)
            else:# not used in case of classification
                eta_pro = 0
                tau_pro = 0

            for s in range(0, self.subtasks-1):
                #using proposed params that include params of all subtasks theta_m
                [likelihood_pro[s],  trainfx[s, :], rmsetrain[s]] = self.loglikelihood(Netlist[s], y_train, self.traindata, w_pro[s, :netsize[s]], tau_pro,s, Netlist[s].Top)
                [likelihood_ignore,  testfx[s, :], rmsetest[s]] = self.loglikelihood(Netlist[s], y_test, self.testdata, w_pro[s, :netsize[s]], tau_pro,s, Netlist[s].Top)

                w_pro[s + 1, :netsize[s]] = w_pro[s, :netsize[s]]

            s = self.subtasks  -1

            [likelihood_pro[s], trainfx[s, :], rmsetrain[s]] = self.loglikelihood(Netlist[s], y_train, self.traindata,  w_pro[s, :netsize[s]],  tau_pro, s, Netlist[s].Top)
            [likelihood_ignore,  testfx[s, :], rmsetest[s]] = self.loglikelihood(Netlist[s], y_test, self.testdata,  w_pro[s, :netsize[s]],  tau_pro, s, Netlist[s].Top)



            for n in range(0, self.subtasks):
                prior_pro[n] = self.prior(sigma_squared, nu_1, nu_2, w_pro[s, :netsize[s]], tau_pro, Netlist[n].Top)

            diff = likelihood_pro  - likelihood
            diff_prior = prior_pro - prior_likelihood


            for s in range(0, self.subtasks):
                try:
                    mh_prob[s] = min(1, math.exp(diff[s] + diff_prior[s]))
                except OverflowError as e:
                    mh_prob[s] = 1

                u = random.uniform(0, 1)


                if u < mh_prob[s]:
                    naccept += 1
                    print(i, ' is accepted sample')
                    likelihood[s] = likelihood_pro[s]
                    w[s,:netsize[s]] = w_pro[s,:netsize[s]]  # _w_proposal
                    eta = eta_pro

                    prior_likelihood[s] = prior_pro[s]

                    #print rmsetrain[s]

                    print(likelihood_pro[s], prior_pro[s], rmsetrain[s], rmsetest[s],  '   for', s)  # takes care of the gradients

                    print(likelihood_pro, prior_pro, rmsetrain, rmsetest, '   for all subtasks')  # takes care of the gradients

                    w_pos[i+1, s, :netsize[s]] = w_pro[s, :netsize[s]] #next samples


                    posfx_train[i+1, s, :] = trainfx[s, :]
                    posfx_test[i+1, s, :] = testfx[s, :]

                    posrmse_train[i+1,s] = rmsetrain[s]
                    posrmse_test[i+1,s] = rmsetest[s]

                    pos_tau[i+1] = tau_pro

                else:

                    w_pos[i + 1, s, :netsize[s]] = w_pos[i, s, :netsize[s]]

                    posfx_train[i+1, s, :] = posfx_train[i, s, :]
                    posfx_test[i+1, s, :] = posfx_test[i, s, :]

                    posrmse_train[i+1,s] = posrmse_train[i,s]
                    posrmse_test[i+1,s] = posrmse_test[i,s]

                    pos_tau[i + 1] =  pos_tau[i]

        print(naccept, ' num accepted')
        accept_ratio = naccept / (samples * self.subtasks *1.0) * 100


        return (w_pos, posfx_train, posfx_test, posrmse_train, posrmse_test, pos_tau, x_train, x_test,  y_test, y_train, accept_ratio)



# ------------------------------------------------------------------------------------------------------


def main():


    for problem in range(17, 18):
        #CREATE NEW FOLDER IN DIR
        outres = open('mt_result_classification.txt', "a+")
        num_samples = 10000 # 80 000 used in exp. note total num_samples is num_samples * num_subtasks

        if problem == 1:
            traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
            name	= "Lazer"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 2:
            traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
            name	= "Sunspot"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 3:
            traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #
            name	= "Mackey"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 4:
            traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
            testdata    = np.loadtxt("Data_OneStepAhead/Lorenz/test.txt")  #
            name    = "Lorenz"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 5:
            traindata = np.loadtxt( "Data_OneStepAhead/Rossler/train.txt")
            testdata    = np.loadtxt( "Data_OneStepAhead/Rossler/test.txt")    #
            name    = "Rossler"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 6:
            traindata = np.loadtxt("Data_OneStepAhead/Henon/train.txt")
            testdata    = np.loadtxt("Data_OneStepAhead/Henon/test.txt")    #
            name    = "Henon"
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        if problem == 7:
            traindata = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt") 
            testdata    = np.loadtxt("Data_OneStepAhead/ACFinance/test.txt")    #
            name    = "ACFinance"  
            hidden = 5
            input = 7  # max input
            output = 1
            prob_type = 'regression'
        # CLASSIFICATION
        separate_flag = False
        if problem == 8: #Wine Quality White
            data  = np.genfromtxt('DATA/winequality-red.csv',delimiter=';')
            data = data[1:,:] #remove Labels
            classes = data[:,11].reshape(data.shape[0],1)
            features = data[:,0:11]
            separate_flag = True
            name = "winequality-red"
            prob_type = 'classification'
            input = 11
            hidden = 50
            output = 10
        if problem == 9: #IRIS
            data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
            classes = data[:,4].reshape(data.shape[0],1)-1
            features = data[:,0:4]
 
            separate_flag = True
            name = "iris"
            prob_type = 'classification'
            input = 4 #input
            hidden = 12
            output = 3
        if problem == 10: #Ionosphere
            traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
            testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
            name = "Ionosphere"
            prob_type = 'classification'
            input = 34
            hidden = 50
            output = 2
        if problem == 11: #Cancer
            traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
            testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
            name = "Cancer"
            prob_type = 'classification'
            input = 9
            hidden = 12
            output = 2    
        if problem == 12: #Bank additional
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Bank/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Bank/ftest.csv',delimiter=',')
            name = "bank-additional"
            prob_type = 'classification'
            input = 51
            hidden = 100
            output = 2
        if problem == 13: #PenDigit
            traindata = np.genfromtxt('DATA/PenDigit/train.csv',delimiter=',') #relative path (needs to cd the current folder that contains DATA)
            testdata = np.genfromtxt('DATA/PenDigit/test.csv',delimiter=',') # / - absolute path, no /: relative path -> use relative path to run on diff computers
            name = "PenDigit"
            prob_type = 'classification'
            for k in range(input):
                mean_train = np.mean(traindata[:,k])
                dev_train = np.std(traindata[:,k]) 
                traindata[:,k] = (traindata[:,k]-mean_train)/dev_train
                mean_test = np.mean(testdata[:,k])
                dev_test = np.std(testdata[:,k]) 
                testdata[:,k] = (testdata[:,k]-mean_test)/dev_test
            input = 16
            hidden = 30
            output = 10

        if problem == 14: #Chess
            data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
            classes = data[:,6].reshape(data.shape[0],1)
            features = data[:,0:6]
            separate_flag = True
            name = "chess"
            prob_type = 'classification'
            input = 6
            hidden = 25
            output = 18


        if problem == 15: #Abalone
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Abalone/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Abalone/ftest.csv',delimiter=',')
            name = "Abalone"
            prob_type = 'classification'
            input = 10
            hidden = 20
            output = 2

        if problem == 16: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Spam/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Spam/ftest.csv',delimiter=',')
            name = "Spam"
            prob_type = 'classification'
            input = 19
            hidden = 50
            output = 2

        if problem == 17: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/ecoli/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/ecoli/ftest.csv',delimiter=',')
            name = "Ecoli"
            prob_type = 'classification'
            input = 7
            hidden = 24
            output = 2

        if problem == 18: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Poker/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Poker/ftest.csv',delimiter=',')
            name = "Poker"
            prob_type = 'classification'
            input = 10
            hidden = 34
            output = 2


        if problem == 19: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Yeast/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Yeast/ftest.csv',delimiter=',')
            name = "Yeast"
            prob_type = 'classification'
            input = 8
            hidden = 29
            output = 2


        if problem == 20: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/SHUTTLE/ftrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/SHUTTLE/ftest.csv',delimiter=',')
            name = "Shuttle"
            prob_type = 'classification'
            input = 9
            hidden = 37
            output = 2

        if problem == 21: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Page Blocks/ttrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Page Blocks/ttest.csv',delimiter=',')
            name = "Page Blocks"
            prob_type = 'classification'
            input = 10
            hidden = 34
            output = 2

        if problem == 22: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/Data_OneStepAhead/energy/train.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/Data_OneStepAhead/energy/test.csv',delimiter=',')
            name    = "Energy"
            hidden = 12
            input = 8  #
            output = 1 
            prob_type = 'regression'

        if problem == 23: 
            traindata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Abalone/rtrain.csv',delimiter=',')
            testdata = np.genfromtxt('/Users/megannguyen/Desktop/mt-bnn-dts-master/DATA/Abalone/rtest.csv',delimiter=',')
            name    = "Abalone"
            input = 10
            hidden = 20
            output = 1
            prob_type = 'regression'

        #Separating data to train and test
        if separate_flag is True:
            #Normalizing Data
            for k in range(input):
                mean = np.mean(features[:,k])
                dev = np.std(features[:,k])
                features[:,k] = (features[:,k]-mean)/dev
            train_ratio = 0.7 #Choosable
            indices = np.random.permutation(features.shape[0])
            traindata = np.hstack([features[indices[:np.int(train_ratio*features.shape[0])],:],classes[indices[:np.int(train_ratio*features.shape[0])],:]])
            testdata = np.hstack([features[indices[np.int(train_ratio*features.shape[0])]:,:],classes[indices[np.int(train_ratio*features.shape[0])]:,:]])

        min_perf = 0.0000001  # stop when RMSE reches this point

        subtasks = 3 #

        mtaskNet = []
        for i in np.arange(0, subtasks, 1):
            mtaskNet.append([input, hidden + i * 4, output])
        mtaskNet = np.array(mtaskNet)

        print(mtaskNet)  # print network topology of all the modules that make the respective tasks. Note in this example, the tasks aredifferent network topologies given by hiddent number of hidden layers.


        bayesnn = BayesNN(mtaskNet, traindata, testdata, num_samples, min_perf,   subtasks, prob_type)

        [w_pos, posfx_train, posfx_test, posrmse_train, posrmse_test, pos_tau, x_train, x_test, y_test, y_train, accept_ratio] = bayesnn.mcmc_sampler()


        print('sucessfully sampled')

        burnin = 0.1 * num_samples  # use post burn in samples

        rmsetr = np.zeros(subtasks)
        rmsetr_std = np.zeros(subtasks)
        rmsetes = np.zeros(subtasks)
        rmsetes_std = np.zeros(subtasks)


        print(accept_ratio)


        for s in range(0, subtasks):
            rmsetr[s] = scipy.mean(posrmse_train[int(burnin):,s])
            rmsetr_std[s] = np.std(posrmse_train[int(burnin):,s])
            rmsetes[s]= scipy.mean(posrmse_test[int(burnin):,s])
            rmsetes_std[s] = np.std(posrmse_test[int(burnin):,s])


        #print rmse for each subtask
        print(rmsetr, rmsetr_std, rmsetes, rmsetes_std, 'each subtask')



        np.savetxt(outres, (problem, accept_ratio), fmt='%1.1f')
        np.savetxt(outres, (mtaskNet), fmt='%1.1f')
        np.savetxt(outres, (rmsetr, rmsetr_std, rmsetes, rmsetes_std), fmt='%1.5f')



        #next outputs  ------------------------------------------------------------




        fx_mu = posfx_test[int(burnin):,0,:].mean(axis=0)
        fx_high = np.percentile(posfx_test[int(burnin):,0,:], 95, axis=0)
        fx_low = np.percentile(posfx_test[int(burnin):,0,:], 5, axis=0)

        fx_mu_tr = posfx_train[int(burnin):,0,:].mean(axis=0)
        fx_high_tr = np.percentile(posfx_train[int(burnin):,0,:], 95, axis=0)
        fx_low_tr = np.percentile(posfx_train[int(burnin):,0,:], 5, axis=0)


        fx_mu1 = posfx_test[int(burnin):,1,:].mean(axis=0) #burned in samples, subtask #, testdata
        fx_high1 = np.percentile(posfx_test[int(burnin):,1,:], 95, axis=0)
        fx_low1 = np.percentile(posfx_test[int(burnin):,1,:], 5, axis=0)

        fx_mu_tr1 = posfx_train[int(burnin):,1,:].mean(axis=0)
        fx_high_tr1 = np.percentile(posfx_train[int(burnin):,1,:], 95, axis=0)
        fx_low_tr1 = np.percentile(posfx_train[int(burnin):,1,:], 5, axis=0)

        fx_mu2 = posfx_test[int(burnin):, 2, :].mean(axis=0)
        fx_high2= np.percentile(posfx_test[int(burnin):, 2, :], 95, axis=0)
        fx_low2 = np.percentile(posfx_test[int(burnin):, 2, :], 5, axis=0)

        fx_mu_tr2 = posfx_train[int(burnin):, 2, :].mean(axis=0)
        fx_high_tr2 = np.percentile(posfx_train[int(burnin):, 2, :], 95, axis=0)
        fx_low_tr2 = np.percentile(posfx_train[int(burnin):, 2, :], 5, axis=0)

        #subtask 1

        plt.plot(x_test, y_test, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Test data prediction performance and uncertainity")
        plt.savefig('restest.png')
        plt.savefig('restest.svg', format='svg', dpi=600)
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, y_train, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Train data prediction performance and uncertainity ")
        plt.savefig( 'restrain.png')
        plt.savefig('restrain.svg', format='svg', dpi=600)

        plt.clf()

        #subtask 2


        plt.plot(x_test, y_test, label='actual')
        plt.plot(x_test, fx_mu1, label='pred. (mean)')
        plt.plot(x_test, fx_low1, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high1, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low1, fx_high1, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Test data prediction performance and uncertainity")
        plt.savefig('restest1.png')
        plt.savefig('restest1.svg', format='svg', dpi=600)
        plt.clf()
        # ------------------------------1-----------
        plt.plot(x_train, y_train, label='actual')
        plt.plot(x_train, fx_mu_tr1, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr1, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr1, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr1, fx_high_tr1, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Train data prediction performance and uncertainity ")
        plt.savefig('restrain1.png')
        plt.savefig('restrain1.svg', format='svg', dpi=600)

        plt.clf()

        # subtask 3


        plt.plot(x_test, y_test, label='actual')
        plt.plot(x_test, fx_mu2, label='pred. (mean)')
        plt.plot(x_test, fx_low2, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high2, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low2, fx_high2, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Test data prediction performance and uncertainity")
        plt.savefig('restest2.png')
        plt.savefig('restest2.svg', format='svg', dpi=600)
        plt.clf()
        # ------------------------------1-----------
        plt.plot(x_train, y_train, label='actual')
        plt.plot(x_train, fx_mu_tr2, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr2, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr2, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr2, fx_high_tr2, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Train data prediction performance and uncertainity ")
        plt.savefig('restrain2.png')
        plt.savefig( 'restrain2.svg', format='svg', dpi=600)

        plt.clf()


if __name__ == "__main__": main()
