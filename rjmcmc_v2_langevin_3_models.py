import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

import scipy
from scipy.special import gamma
from scipy import stats
from scipy.stats import binom

# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, MinPer, LearnRate):

        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate = LearnRate  # will be updated later with BP call

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
    def langevin_gradient(self, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = self.TrainData.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, depth):
            for i in range(0, size):
                pat = i
                Input = self.TrainData[pat, 0:self.Top[0]]
                Desired = self.TrainData[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated

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
    def __init__(self, use_langevin_gradients, langevin_prob, samples, traindata, testdata, minPerf, LearnRate, mtaskNet):
        self.samples = samples  
        self.mtaskNet = mtaskNet  # mtaskNet = np.array([baseNet, secondnet]) -> to access each nn, mtaskNet[dims]
        self.traindata = traindata  #
        self.testdata = testdata

        self.langevin_prob = langevin_prob
        self.use_langevin_gradients  =  use_langevin_gradients
        self.minPerf = minPerf
        self.lrate = LearnRate
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
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(2 * math.pi * sigma_squared)
        part2 = -1 / (2 * sigma_squared) * (sum(np.square(w)))
        part3 = np.log(model_prior)
        log_loss = part1 + part2 + part3 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq) + nu_1 * np.log(nu_2) - np.log(gamma(nu_1))
        return log_loss

    def v_jump_up(self, tausq_v, v, size_diff): #choose tausq well for better proposal
        log_v =( -(size_diff/2) * np.log(2 * math.pi * tausq_v)) -( (1/(2*tausq_v)) * np.sum(np.square(v)))
        return log_v
    def v_jump_down(self, scale, tausq_v, v, size_diff): #choose tausq well for better proposal
        log_v =( -(size_diff/2) * np.log(2 * math.pi * tausq_v)) -( (1/(2*tausq_v)) * np.sum(np.square(np.log(v / scale))))
        return log_v

    def sampler(self, dims1, dims2, dims3, w_limit, eta_limit):

        #---- Empty set of networks
        Netw = []
        Net = []
        w_Size = []
        w_Net = []
        Y_train = []

        for dims in [dims1, dims2, dims3]:
            #---- Define model
            netw = self.mtaskNet[dims]
            Netw.append(netw)
            net = Network(netw, self.traindata, self.testdata, self.minPerf, self.lrate) 
            Net.append(net)
            #---- Initialize w
            w_size = net.net_size()
            w_Size.append(w_size)
            w_net = np.random.normal(0, 1, w_size)
            w_Net.append(w_net)
            y_train = self.traindata[:, netw[0]]
            Y_train.append(y_train)
        
        #----- Initialize MCMC
        samples = self.samples

        self.sgd_depth = 1

        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        fxtrain_samples = []
        fxtest_samples = []
        rmse_train = []
        rmse_test = []
        pos_w = []
        #--- HYPERPARAMETERS
        sigma_squared = 25
        nu_1 = 3
        nu_2 = 1
        model_prior = 1/2 #for both models 1 and 2
        current_dims = 0
        up = True

        #--- HYPERPARAMETERS FOR H
        scale = 0.02
        tausq_v = 1

        naccept = 0
        langevin_count = 0
        within_model_samples = 0

        for i in range(samples -1):
            if up: #jump up dimension: netw1 < netw2
                current_net = Net[current_dims]
                current_w_net = w_Net[current_dims]
                current_y_train = Y_train[current_dims]
                current_w_size = w_Size[current_dims]
                #--- Propose error term
                pred_train = current_net.evaluate_proposal(current_w_net)
                eta = np.log(np.var(pred_train - current_y_train))                
                eta_pro = eta + np.random.normal(0, eta_limit, 1)
                tau_pro = np.exp(eta_pro)
                #--- Propose new dimension
                proposed_dims = (current_dims + 1)
                proposed_net = Net[proposed_dims]
                proposed_w_size = w_Size[proposed_dims]
                #--- Propose jump vector v from a N(0, tausq) CHOOSE WELL!!!
                v = np.random.normal(0, tausq_v, proposed_w_size-current_w_size)
                #--- Propose w
                lx = np.random.uniform(0,1,1)

                if (self.use_langevin_gradients is True) and (lx< self.langevin_prob):  
                    w_gd = current_net.langevin_gradient(current_w_net.copy(), self.sgd_depth)  
                    #w_proposal = np.concatenate((np.random.normal(w_gd, w_limit, w_size1), v), axis = None)
                    w_proposal = np.concatenate((np.random.normal(w_gd, w_limit, current_w_size), scale * np.exp(v)), axis = None)

                    w_prop_gd = proposed_net.langevin_gradient(w_proposal.copy(), self.sgd_depth) 
                    wc_delta = (current_w_net - w_gd) 
                    wp_delta = (w_proposal - w_prop_gd)

                    sigma_sq = w_limit * w_limit #std

                    first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                    second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq

                    diff_prop =  first - second  
                    langevin_count = langevin_count + 1
                else:
                    diff_prop = 0
                    w_proposal = np.concatenate((np.random.normal(current_w_net, w_limit, current_w_size), scale * np.exp(v)), axis = None)
                #--- Calc current prior and likelihood
                [likelihood_current, pred_train, rmse_train_current] = self.likelihood_func(current_net, self.traindata, current_w_net, tau_pro, current_dims) #w here used for bnn
                [pred_test_current, rmse_test_current] = current_net.test_proposal(current_w_net)	
                prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, current_w_net, tau_pro, model_prior, current_dims)
                #--- Calc proposed prior and likelihood
                [likelihood_proposal, pred_train_proposal, rmse_train_proposal] = self.likelihood_func(proposed_net, self.traindata, w_proposal, tau_pro, proposed_dims)
                [pred_test_proposal, rmse_test_proposal] = proposed_net.test_proposal(w_proposal)				
                prior_proposal = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, model_prior, proposed_dims)
                #--- Calc log_v such that tausq is good
                log_v = self.v_jump_up(tausq_v, v, proposed_w_size-current_w_size)
                #--- Calc log posterior 
                acceptance = (prior_proposal + likelihood_proposal) - (prior_current + likelihood_current + log_v) + (proposed_w_size-current_w_size) * np.log(scale) + np.sum(v) + diff_prop 
                try:
                    mh_prob = min(1, math.exp(acceptance))

                except OverflowError as e:
                    mh_prob = 1
                #mh_prob = 1
                u = random.uniform(0, 1)

                if u < mh_prob:
                    # ACCEPT
                    naccept += 1
                    likelihood_current = likelihood_proposal
                    prior_current = prior_proposal            
                    print(i, 'jump from model with size', current_w_size,'to model with size', proposed_w_size)
                    current_dims = proposed_dims

                    w_Net[current_dims] = w_proposal #assign proposed w to w which now has higher dim
                    pos_w.append(w_proposal)
                    if current_dims < dims3:
                        print(i, 'at model', current_w_size)
                        up = True

                    else:
                        print(i, 'at model', current_w_size)
                        up = False

                    fxtrain_samples.append(pred_train_proposal)
                    fxtest_samples.append(pred_test_proposal)
                    rmse_train.append(rmse_train_proposal)
                    rmse_test.append(rmse_test_proposal)
                else:
                    print('Sample from same model with size', current_w_size, 'Start within-model sampling')

                    if (self.use_langevin_gradients is True) and (lx< self.langevin_prob):  
                        w_gd_within = current_net.langevin_gradient(current_w_net.copy(), self.sgd_depth)
                        w_proposal_within = np.random.normal(w_gd_within, w_limit, current_w_size)  
                        w_prop_gd_within = current_net.langevin_gradient(w_proposal_within.copy(), self.sgd_depth) 

                        wc_delta_within = (current_w_net - w_prop_gd_within) 
                        wp_delta_within = (w_proposal_within - w_gd_within )

                        diff_prop_within = (-0.5 * np.sum(wc_delta_within  *  wc_delta_within  ) / sigma_sq) - (-0.5 * np.sum(wp_delta_within * wp_delta_within ) / sigma_sq)  

                    else:
                        diff_prop_within = 0
                        w_proposal_within = np.random.normal(current_w_net, w_limit, current_w_size)

                    #--- Calc proposed prior and likelihood
                    [likelihood_proposal_within, pred_train_proposal_within, rmse_train_proposal_within] = self.likelihood_func(current_net, self.traindata, w_proposal_within, tau_pro, current_dims)
                    [pred_test_proposal_within, rmse_test_proposal_within] = current_net.test_proposal(w_proposal_within)				
                    prior_proposal_within = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal_within, tau_pro, model_prior, current_dims)
                    #--- Calc log posterior 
                    acceptance_within = (prior_proposal_within + likelihood_proposal_within) - (prior_current + likelihood_current) + diff_prop_within

                    try:
                        mh_prob_within = min(1, math.exp(acceptance_within))

                    except OverflowError as e:
                        mh_prob_within = 1


                    t = random.uniform(0, 1)

                    if t < mh_prob_within:
                        # Update position 
                        within_model_samples += 1
                        likelihood_current = likelihood_proposal_within
                        prior_current = prior_proposal_within
                        current_w_net = w_proposal_within
                        pos_w.append(current_w_net)
                        print('Sampling from within model', current_w_net, current_w_size, 'accepted')
    
                        fxtrain_samples.append(pred_train_proposal_within)
                        fxtest_samples.append(pred_test_proposal_within)
                        rmse_train.append(rmse_train_proposal_within)
                        rmse_test.append(rmse_test_proposal_within)


                    else:
                        within_model_samples += 1
                        pos_w.append(current_w_net)
                        fxtrain_samples.append(pred_train)
                        fxtest_samples.append(pred_test_current)
                        rmse_train.append(rmse_train_current)
                        rmse_test.append(rmse_test_current)


            else: #w_size2 > w_size1
                current_net = Net[current_dims]
                current_w_net = w_Net[current_dims]
                current_w_size = w_Size[current_dims]
                current_y_train = Y_train[current_dims]                	
                #--- Propose error term
                pred_train = current_net.evaluate_proposal(current_w_net)
                eta = np.log(np.var(pred_train - current_y_train))
                eta_pro = eta + np.random.normal(0, eta_limit, 1)
                tau_pro = np.exp(eta_pro)
                #--- Propose new dimension
                proposed_dims = (current_dims - 1)
                proposed_net = Net[proposed_dims]
                proposed_w_size = w_Size[proposed_dims]
                #--- Propose jump vector v from a N(0, tausq) CHOOSE WELL!!!
                v = np.random.normal(0, tausq_v, current_w_size-proposed_w_size)
                #--- Propose w
                lx = np.random.uniform(0,1,1)

                if (self.use_langevin_gradients is True) and (lx< self.langevin_prob):  
                    w_gd = current_net.langevin_gradient(current_w_net.copy(), self.sgd_depth)
                    w_proposal = (np.random.normal(w_gd, w_limit, current_w_size))[0:proposed_w_size]
                    #--- Calc v (Can comment)
                    #v = (np.random.normal(w_gd, w_limit, w_size2))[-(w_size2-w_size1):]

                    w_prop_gd = proposed_net.langevin_gradient(w_proposal.copy(), self.sgd_depth) 
                    wc_delta = (current_w_net - w_gd) 
                    wp_delta = (w_proposal - w_prop_gd)

                    sigma_sq = w_limit * w_limit #std

                    first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                    second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq

                    diff_prop =  first - second  
                    langevin_count = langevin_count + 1
                else:
                    diff_prop = 0
                    w_proposal = np.random.normal(current_w_net, w_limit, current_w_size)[0:proposed_w_size] #propose w_down that has lower dim
                    #--- Calc v (Can comment)
                    #v = (np.random.normal(w_net2, w_limit, w_size2))[-(w_size2-w_size1):]

                #--- Calc current prior and likelihood

                [likelihood_current, pred_train, rmse_train_current] = self.likelihood_func(current_net, self.traindata, current_w_net, tau_pro, current_dims) #w here used for bnn
                [pred_test_current, rmse_test_current] = current_net.test_proposal(current_w_net)	
                prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, current_w_net, tau_pro, model_prior, current_dims)

                #--- Calc proposed prior and likelihood
                [likelihood_proposal, pred_train_proposal, rmse_train_proposal] = self.likelihood_func(proposed_net, self.traindata, w_proposal, tau_pro, proposed_dims)
                [pred_test_proposal, rmse_test_proposal] = proposed_net.test_proposal(w_proposal)	
                prior_proposal = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, model_prior, proposed_dims)
                #--- Calc log_v such that tausq is good
                log_v = self.v_jump_down(scale, tausq_v, v, current_w_size-proposed_w_size)
                #--- Calc log posterior 

                acceptance_down = (prior_current + likelihood_current + log_v) - (prior_proposal + likelihood_proposal) - np.log(np.sum(v)) - diff_prop 
                try:
                    mh_prob_down = min(1, math.exp(acceptance_down))
                except OverflowError as e:
                    mh_prob_down = 1
                #mh_prob_down = 1
                u_down = random.uniform(0, 1)
                if u_down < mh_prob_down:
                    naccept += 1
                    likelihood_current = likelihood_proposal
                    prior_current = prior_proposal
                    current_dims = proposed_dims

                    w_Net[current_dims] = w_proposal
                    pos_w.append(w_Net[current_dims])
                    print(i, 'jump from model with size', current_w_size,'to model with size', proposed_w_size)
                    if current_dims > dims1:
                        print(i, 'at model', w_Size[current_dims])
                        up = False

                    else:
                        print(i, 'at model', w_Size[current_dims])
                        up = True
                    
                    fxtrain_samples.append(pred_train_proposal)
                    fxtest_samples.append(pred_test_proposal)
                    rmse_train.append(rmse_train_proposal)
                    rmse_test.append(rmse_test_proposal)
                else:
                    print('Sample from same model with size', current_w_size, 'Start within-model sampling')

                    if (self.use_langevin_gradients is True) and (lx< self.langevin_prob):  
                        w_gd_within = current_net.langevin_gradient(current_w_net.copy(), self.sgd_depth)
                        w_proposal_within = np.random.normal(w_gd_within, w_limit, current_w_size)  
                        w_prop_gd_within = current_net.langevin_gradient(w_proposal_within.copy(), self.sgd_depth) 

                        wc_delta_within = (current_w_net - w_prop_gd_within) 
                        wp_delta_within = (w_proposal_within - w_gd_within )

                        diff_prop_within  = (-0.5 * np.sum(wc_delta_within  *  wc_delta_within  ) / sigma_sq) - (-0.5 * np.sum(wp_delta_within * wp_delta_within ) / sigma_sq)

                    else:
                        diff_prop_within = 0
                        w_proposal_within = np.random.normal(current_w_net, w_limit, current_w_size)

                    #--- Calc proposed prior and likelihood
                    [likelihood_proposal_within, pred_train_proposal_within, rmse_train_proposal_within] = self.likelihood_func(current_net, self.traindata, w_proposal_within, tau_pro, current_dims)
                    [pred_test_proposal_within, rmse_test_proposal_within] = current_net.test_proposal(w_proposal_within)				
                    prior_proposal_within = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal_within, tau_pro, model_prior, current_dims)
                    #--- Calc log posterior 
                    acceptance_down_within = (prior_proposal_within + likelihood_proposal_within) - (prior_current + likelihood_current) + diff_prop_within 

                    try:
                        mh_prob_down_within = min(1, math.exp(acceptance_down_within))

                    except OverflowError as e:
                        mh_prob_down_within = 1


                    t = random.uniform(0, 1)

                    if t < mh_prob_down_within:
                        # Update position 
                        within_model_samples += 1
                        likelihood_current = likelihood_proposal_within
                        prior_current = prior_proposal_within
                        current_w_net = w_proposal_within
                        pos_w.append(current_w_net)
                        print('Sampling from within model', current_w_size, current_w_net, 'accepted')
    
                        fxtrain_samples.append(pred_train_proposal_within)
                        fxtest_samples.append(pred_test_proposal_within)
                        rmse_train.append(rmse_train_proposal_within)
                        rmse_test.append(rmse_test_proposal_within)


                    else:
                        within_model_samples += 1
                        pos_w.append(current_w_net)
                        fxtrain_samples.append(pred_train)
                        fxtest_samples.append(pred_test_current)
                        rmse_train.append(rmse_train_current)
                        rmse_test.append(rmse_test_current)

            within_model_ratio = within_model_samples / samples
            accept_ratio = naccept / (samples)
            print(np.shape(pos_w), '{:.1%}'.format(accept_ratio), 'is accepted with langevin counts', langevin_count, 'with within-model samples', within_model_samples)
        return (pos_w, fxtrain_samples, fxtest_samples, rmse_train, rmse_test, accept_ratio, within_model_ratio)

def main():
    for problem in range(1, 2): 

        if problem == 1:
            traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
            name	= "Lazer"
        if problem == 2:
            traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
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
        input = 4 
        hidden = 5
        output = 1
        baseNet = [input, hidden, output]
        secondnet = [input, hidden+1, output]
        thirdnet = [input, hidden + 2, output]
        mtaskNet = np.array([baseNet, secondnet, thirdnet])
        #--- Hyperparams for MCMC
        numSamples = 10000
        minPerf = 0.0001 #where rmse stops
        LearnRate = 0.1

        #--- Run MCMC
        dims1 = 0
        dims2 = 1
        dims3 = 2
        w_limit = 0.05
        eta_limit = 0.01
        use_langevin_gradients  = True
        langevin_prob = 0.5

        mcmc = MCMC(use_langevin_gradients, langevin_prob, numSamples, traindata, testdata, minPerf, LearnRate, mtaskNet)

        [pos_w, fx_train, fx_test, rmse_train, rmse_test, accept_ratio, within_model_ratio] = mcmc.sampler(dims1, dims2, dims3, w_limit, eta_limit)
        print('finished sampling')

        burnin = 0.5 * numSamples  # use post burn in samples

        pos_w = pos_w[int(burnin):]
        rmse_train = rmse_train[int(burnin): ]
        rmse_test = rmse_test[int(burnin): ]

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


        outres_db = open('rjmcmc_result_3_models.txt', "a+")

        np.savetxt(outres_db, (use_langevin_gradients, len(mtaskNet),  rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio, within_model_ratio), fmt='%1.5f')
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

        plt.title("Prediction  Uncertainty to jump among 3 models ")
        plt.savefig('rjmcmcrestest_3 models.png') 
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Prediction  Uncertainty to jump among 3 models")
        plt.savefig('rjmcmcrestrain_3 models.png') 
        plt.clf()

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        ax.boxplot(pos_w)
        ax.set_xlabel('[W1] [B1] [W2] [B2]')
        ax.set_ylabel('Posterior')
        ax.legend(loc='upper right')
        plt.title("Boxplot of Posterior W (weights and biases) to jump among 3 models")
        plt.savefig('w_pos_rjmcmc_3 models.png')
        
        plt.clf()


if __name__ == "__main__": main()  