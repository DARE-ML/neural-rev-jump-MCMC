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
    def __init__(self, Topo, Train, Test, MaxTime, MinPer):

        self.Top = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime  # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate = 0  # will be updated later with BP call

        self.minPerf = MinPer
        # initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()
        self.W1 = np.zeros((self.Top[0], self.Top[1])  )
        self.B1 = np.zeros(self.Top[1])    # bias first layer
        self.W2 = np.zeros((self.Top[1], self.Top[2]) )
        self.B2 = np.zeros(self.Top[2])    # bias second layer
        self.hidout = np.zeros((self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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


    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size =  self.Top[1] *  self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, ( self.Top[0],  self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, ( self.Top[1],  self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size +  self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]
    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w
    def net_size(self):
        return ((self.Top[0] * self.Top[1]) + (self.Top[1] * self.Top[2]) + self.Top[1] + self.Top[2])
    def evaluate_proposal(self,   w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = self.TrainData.shape[0]
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        fx = np.zeros(size)



        for pat in range(0, size):
            Input[:] =  self.TrainData[pat, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[pat] = self.out
        return fx

    def test_proposal(self,  w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = self.TestData.shape[0]
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        sse = 0

        for pat in range(0, size):
            Input[:] = self.TestData[pat, 0:self.Top[0]]
            Desired[:] = self.TestData[pat, self.Top[0]:]
            self.ForwardPass(Input)
            fx[pat] = self.out
            sse = sse + self.sampleEr(Desired)

        rmse = np.sqrt(sse / size)


        return [fx,rmse]
#---- RJMCMC

class MCMC:
    def __init__(self, samples, traindata, testdata, minPerf, mtaskNet):
        self.samples = samples  
        self.mtaskNet = mtaskNet  # mtaskNet = np.array([baseNet, secondnet]) -> to access each nn, mtaskNet[dims]
        self.traindata = traindata  #
        self.testdata = testdata 
        self.minPerf = minPerf
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq, dims):
        topology = self.mtaskNet[dims]
        y = data[:, topology[0]]
        fx = neuralnet.evaluate_proposal(w)
        rmse = self.rmse(fx, y)
        n = y.shape[0]

        loss =( -(n/2) * np.log(2 * math.pi * tausq)) -( (1/(2*tausq)) * np.sum(np.square(y - fx))) #tausq is variance of error terms
        return [loss, fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq, model_prior, dims):
        topology = self.mtaskNet[dims]
        h = topology[1]  # number hidden neurons
        d = topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = -1 / (2 * sigma_squared) * (sum(np.square(w)))
        part3 = 1 * np.log(model_prior)
        log_loss = part1 + part2 + part3 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def v_jump(self, tausq, v, size_diff): #choose tausq well for better proposal
        log_v =( -(size_diff/2) * np.log(2 * math.pi * tausq)) -( (1/(2*tausq)) * np.sum(np.square(v))) #tausq is variance of error terms
        return log_v

    def sampler(self, dims1, dims2):
        #----- Model 1
        netw_1 = self.mtaskNet[dims1]		
        Net1 = Network(netw_1, self.traindata, self.testdata, 5, self.minPerf)
        y_test_1 = self.testdata[:, netw_1[0]]
        y_train_1 = self.traindata[:, netw_1[0]]

        #----- Model 2
        netw_2 = self.mtaskNet[dims2]
        Net2 = Network(netw_2, self.traindata, self.testdata, 5, self.minPerf)
        y_test_2 = self.testdata[:, netw_2[0]]
        y_train_2 = self.traindata[:, netw_2[0]]
        #----- Initialize MCMC
        samples = self.samples
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        fxtrain_samples = []
        fxtest_samples = []
        rmse_train = []
        rmse_test = []
        pos_w = []
        #--- HYPERPARAMETERS
        sigma_squared = 5
        nu_1 = 3
        nu_2 = 1
        model_prior = 1/2 #for both models 1 and 2

        naccept = 0
        for i in range(samples -1):
            w_size1 = (netw_1[0] * netw_1[1]) + (netw_1[1] * netw_1[2]) + netw_1[1] + netw_1[2]
            netw_2[1] = netw_1[1] + random.choice([-1, 1])
            w_size2 = (netw_2[0] * netw_2[1]) + (netw_2[1] * netw_2[2]) + netw_2[1] + netw_2[2]

            if w_size2 > w_size1:
                #--- Initial w
                w = np.random.rand(w_size1)
                #--- Propose error term
                pred_train = Net1.evaluate_proposal(w)
                eta = np.log(np.var(pred_train - y_train_1))
                tau_pro = np.exp(eta)
                #--- Calc current prior and likelihood
                [likelihood_current, pred_train, rmse_train_current] = self.likelihood_func(Net1, self.traindata, w, tau_pro, dims1) #w here used for bnn
                [pred_test_current, rmse_test_current] = Net1.test_proposal(w)	
                prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, model_prior, dims1)
                #--- Propose jump vector v
                v = np.random.normal(0, 1, w_size2-w_size1)
                #--- Propose w
                w_proposal = np.lib.pad(w, (0, w_size2-w_size1), 'constant', constant_values=(0,v))
                #--- Calc proposed prior and likelihood
                [likelihood_proposal, pred_train_proposal, rmse_train_proposal] = self.likelihood_func(Net2, self.traindata, w_proposal, tau_pro, dims2)
                [pred_test_proposal, rmse_test_proposal] = Net2.test_proposal(w_proposal)				
                prior_proposal = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, model_prior, dims2)
                log_v = self.v_jump(tau_pro, v, w_size2-w_size1)
                #--- Calc log posterior 
                log_posterior = (prior_proposal + likelihood_proposal) - (prior_current + likelihood_current + log_v) 
                try:
                    mh_prob = min(1, math.exp(log_posterior))

                except OverflowError as e:
                    mh_prob = 1

                u = random.uniform(0, 1)

                if u < mh_prob:
                    # ACCEPT
                    naccept += 1
                    likelihood_current = likelihood_proposal
                    prior_current = prior_proposal            
                    w = w_proposal #assign proposed w to w which now has higher dim
                    pos_w.append(w)

                    fxtrain_samples.append(pred_train_proposal)
                    fxtest_samples.append(pred_test_proposal)
                    rmse_train.append(rmse_train_proposal)
                    rmse_test.append(rmse_test_proposal)
                else:
                    fxtrain_samples.append(pred_train)
                    fxtest_samples.append(pred_test_current)
                    rmse_train.append(rmse_train_current)
                    rmse_test.append(rmse_test_current)

            else: #w_size2 < w_size1
                #--- Initial w
                w = np.random.rand(w_size1)				
                #--- Propose error term
                pred_train = Net1.evaluate_proposal(w)
                eta = np.log(np.var(pred_train - y_train_1))
                tau_pro = np.exp(eta)
                #--- Calc current prior and likelihood

                [likelihood_current, pred_train, rmse_train_current] = self.likelihood_func(Net1, self.traindata, w, tau_pro, dims1) #w here used for bnn
                [pred_test_current, rmse_test_current] = Net1.test_proposal(w)	
                prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, model_prior, dims1)
                #--- Propose jump vector v
                v = np.random.normal(0, 1, w_size1-w_size2)
                #--- Propose w
                w_proposal_down = w[0:w_size2] #propose w_down that has lower dim
                #--- Calc proposed prior and likelihood
                [likelihood_proposal, pred_train_proposal, rmse_train_proposal] = self.likelihood_func(Net2, self.traindata, w_proposal_down, tau_pro, dims2)
                [pred_test_proposal, rmse_test_proposal] = Net2.test_proposal(w_proposal_down)	
                prior_proposal = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal_down, tau_pro, model_prior, dims2)
                log_v = self.v_jump(tau_pro, v, w_size1-w_size2)
                #--- Calc log posterior 

                log_posterior_down = (prior_proposal + likelihood_proposal + log_v) - (prior_current + likelihood_current) 
                try:
                    mh_prob_down = min(1, math.exp(log_posterior_down))
                except OverflowError as e:
                    mh_prob_down = 1
                u_down = random.uniform(0, 1)
                if u_down < mh_prob_down:
                    naccept += 1
                    likelihood_current = likelihood_proposal
                    prior_current = prior_proposal
                    w = w_proposal_down
                    pos_w.append(w)					
                    
                    fxtrain_samples.append(pred_train_proposal)
                    fxtest_samples.append(pred_test_proposal)
                    rmse_train.append(rmse_train_proposal)
                    rmse_test.append(rmse_test_proposal)
                else:
                    fxtrain_samples.append(pred_train)
                    fxtest_samples.append(pred_test_current)
                    rmse_train.append(rmse_test_current)
                    rmse_test.append(rmse_test_current)

            accept_ratio = naccept / (samples)
            print(i, 'jump from model 1 with size', w_size1,'to model 2 with size', w_size2, np.shape(pos_w), '{:.1%}'.format(accept_ratio), 'is accepted')
        return (pos_w, fxtrain_samples, fxtest_samples, rmse_train, rmse_test, accept_ratio)

def main():
    for problem in range(2, 3): 

        if problem == 1:
            traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
            name	= "Lazer"
        if problem == 2:
            traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train7.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test7.txt")  #
            name	= "Sunspot"
        if problem == 3:
            traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #
            name	= "Mackey"
        if problem == 4:
            traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
            testdata    = np.loadtxt("Data_OneStepAhead/Lorenz/test.txt")  #
            name    = "Lorenz"
        if problem == 5:
            traindata = np.loadtxt( "Data_OneStepAhead/Rossler/train.txt")
            testdata    = np.loadtxt( "Data_OneStepAhead/Rossler/test.txt")    #
            name    = "Rossler"
        if problem == 6:
            traindata = np.loadtxt("Data_OneStepAhead/Henon/train.txt")
            testdata    = np.loadtxt("Data_OneStepAhead/Henon/test.txt")    #
            name    = "Henon"
        if problem == 7:
            traindata = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt") 
            testdata    = np.loadtxt("Data_OneStepAhead/ACFinance/test.txt")    #
            name    = "ACFinance"  

        #---- Define BNN topology
        input = 7 
        hidden = 5
        output = 1
        baseNet = [input, hidden, output]
        #secondnet = [input, hidden+1, output]
        mtaskNet = np.array([baseNet, baseNet])
        #--- Hyperparams for MCMC
        numSamples = 5000
        minPerf = 0.0001 #where rmse stops

        #--- Run MCMC
        dims1 = 0
        dims2 = 1

        mcmc = MCMC(numSamples, traindata, testdata, minPerf, mtaskNet)

        [pos_w, fx_train, fx_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler(dims1, dims2)
        print('finished sampling')

        #burnin = 0.5 * numSamples  # use post burn in samples

        #pos_w = pos_w[int(burnin):, ]
        #pos_tau = pos_tau[int(burnin):, ]

        fx_mu = np.mean(fx_test, axis = 0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = np.mean(fx_train,axis = 0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr = np.percentile(fx_train, 5, axis=0)

        rmse_tr = np.mean(rmse_train)
        rmsetr_std = np.std(rmse_train)
        rmse_tes = np.mean(rmse_test)
        rmsetest_std = np.std(rmse_test)
        print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)


        outres_db = open('rjmcmc_result.txt', "a+")

        np.savetxt(outres_db, (len(mtaskNet), rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')
        #----- Plot
        x_test = np.linspace(0, 1, num=testdata.shape[0])
        x_train = np.linspace(0, 1, num=traindata.shape[0])
        ytestdata = testdata[:, input]
        ytraindata = traindata[:, input]

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Prediction  Uncertainty ")
        plt.savefig('rjmcmcrestest.png') 
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Prediction  Uncertainty")
        plt.savefig('rjmcmcrestrain.png') 
        plt.clf()

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        ax.boxplot(pos_w)
        ax.set_xlabel('[W1] [B1] [W2] [B2]')
        ax.set_ylabel('Posterior')
        ax.legend(loc='upper right')
        plt.title("Boxplot of Posterior W (weights and biases)")
        plt.savefig('w_pos_rjmcmc.png')
        
        plt.clf()


if __name__ == "__main__": main()  