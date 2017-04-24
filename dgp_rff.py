## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from __future__ import print_function

import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataset import DataSet
import utils
import likelihoods
import time
import os
#import cifar10_input

current_milli_time = lambda: int(round(time.time() * 1000))

class DgpRff(object):
    def __init__(self, likelihood_fun, num_examples, d_raw_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega):
        """
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_raw_in: Dimensionality of the input, i.e for MNIST d_raw_in = 28 * 28 * 1, d_raw_in = 32 * 32 * 3
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param kernel_arccosine_degree: degree parameter of the arccosine kernel
        :param is_ard: Whether the kernel is ARD or isotropic
        :param feed_forward: Whether the original inputs should be fed forward as input to each layer
        :param Omega_fixed: Whether the Omega weights should be fixed throughout the optimization
        :param theta_fixed: Whether covariance parameters should be fixed throughout the optimization
        :param learn_Omega: How to treat Omega - fixed (from the prior), optimized, or learned variationally
        """
        self.likelihood = likelihood_fun
        self.kernel_type = kernel_type
        self.is_ard = is_ard
        self.feed_forward = feed_forward
        self.q_Omega_fixed = q_Omega_fixed
        self.theta_fixed = theta_fixed
        self.q_Omega_fixed_flag = q_Omega_fixed > 0
        self.theta_fixed_flag = theta_fixed > 0
        self.learn_Omega = learn_Omega
        self.arccosine_degree = kernel_arccosine_degree
        
        self.Dfeatures = 7 * 7 * 64;
        
        ## These are all scalars
        self.num_examples = num_examples
        self.nl = n_layers ## Number of hidden layers
        self.n_Omega = n_layers  ## Number of weigh matrices is "Number of hidden layers"
        self.n_W = n_layers

        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype = np.int64)
        self.df = df * np.ones(n_layers, dtype=np.int64)

        ## Dimensionality of Omega matrices
        if self.feed_forward:
            #self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)] + d_in])
            self.d_in = np.concatenate([[self.Dfeatures], self.df[:(n_layers - 1)] + self.Dfeatures])
        else:
            #self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)]])
            self.d_in = np.concatenate([[self.Dfeatures], self.df[:(n_layers - 1)]])
        self.d_out = self.n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])

        ## When Omega is learned variationally, define the right KL function and the way Omega are constructed
        if self.learn_Omega == "var":
            self.get_kl = self.get_kl_Omega_to_learn
            self.sample_from_Omega = self.sample_from_Omega_to_learn

        ## When Omega is optimized, fix some standard normals throughout the execution that will be used to construct Omega
        if self.learn_Omega == "optim":
            self.get_kl = self.get_kl_Omega_to_learn
            self.sample_from_Omega = self.sample_from_Omega_optim

            self.z_for_Omega_fixed = []
            for i in range(self.n_Omega):
                tmp = utils.get_normal_samples(1, self.d_in[i], self.d_out[i])
                self.z_for_Omega_fixed.append(tf.Variable(tmp[0,:,:], trainable = False))

        ## When Omega is fixed, fix some standard normals throughout the execution that will be used to construct Omega
        if self.learn_Omega == "no":
            self.get_kl = self.get_kl_Omega_fixed
            self.sample_from_Omega = self.sample_from_Omega_fixed
 
            self.z_for_Omega_fixed = []
            for i in range(self.n_Omega):
                tmp = utils.get_normal_samples(1, self.d_in[i], self.d_out[i])
                self.z_for_Omega_fixed.append(tf.Variable(tmp[0,:,:], trainable = False))

        ## Parameters defining prior over Omega
        self.log_theta_sigma2 = tf.Variable(tf.zeros([n_layers]), name="log_theta_sigma2")

        if self.is_ard:
            self.llscale0 = []
            for i in range(self.nl):
                self.llscale0.append(tf.constant(0.5 * np.log(self.d_in[i]), tf.float32))
        else:
            self.llscale0 = tf.constant(0.5 * np.log(self.d_in), tf.float32)

        if self.is_ard:
            self.log_theta_lengthscale = []
            for i in range(self.nl):
                self.log_theta_lengthscale.append(tf.Variable(tf.mul(tf.ones([self.d_in[i]]), self.llscale0[i]), name="log_theta_lengthscale"))
        else:
            self.log_theta_lengthscale = tf.Variable(self.llscale0, name="log_theta_lengthscale")
        self.prior_mean_Omega, self.log_prior_var_Omega = self.get_prior_Omega(self.log_theta_lengthscale)

        ## Set the prior over weights
        self.prior_mean_W, self.log_prior_var_W = self.get_prior_W()

        ## Initialize posterior parameters
        if self.learn_Omega == "var":
            self.mean_Omega, self.log_var_Omega = self.init_posterior_Omega()
        if self.learn_Omega == "optim":
            self.mean_Omega, self.log_var_Omega = self.init_posterior_Omega()

        self.mean_W, self.log_var_W = self.init_posterior_W()
        
        ## Initialize filters and bias of CNN structures
        self.filters1, self.bias1, self.filters2, self.bias2 = self.init_param_CNN()

        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.mc =  tf.placeholder(tf.int32) 

        ## Batch data placeholders
        Din = d_raw_in
        Dout = d_out
        self.X = tf.placeholder(tf.float32, [None, Din])
        self.Y = tf.placeholder(tf.float32, [None, Dout])

        ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()

        ## Initialize the session
        self.session = tf.Session()

    ## Definition of a prior for Omega - which depends on the lengthscale of the covariance function
    def get_prior_Omega(self, log_lengthscale):
        if self.is_ard:
            prior_mean_Omega = []
            log_prior_var_Omega = []
            for i in range(self.nl):
                prior_mean_Omega.append(tf.zeros([self.d_in[i],1]))
            for i in range(self.nl):
                log_prior_var_Omega.append(-2 * log_lengthscale[i])
        else:
            prior_mean_Omega = tf.zeros(self.nl)
            log_prior_var_Omega = -2 * log_lengthscale
        return prior_mean_Omega, log_prior_var_Omega

    ## Definition of a prior over W - these are standard normals
    def get_prior_W(self):
        prior_mean_W = tf.zeros(self.n_W)
        log_prior_var_W = tf.zeros(self.n_W)
        return prior_mean_W, log_prior_var_W

    ## Function to initialize the posterior over omega
    def init_posterior_Omega(self):
        mu, sigma2 = self.get_prior_Omega(self.llscale0)

        mean_Omega = [tf.Variable(mu[i] * tf.ones([self.d_in[i], self.d_out[i]]), name="q_Omega") for i in range(self.n_Omega)]
        log_var_Omega = [tf.Variable(sigma2[i] * tf.ones([self.d_in[i], self.d_out[i]]), name="q_Omega") for i in range(self.n_Omega)]

        return mean_Omega, log_var_Omega

    ## Function to initialize the posterior over W
    def init_posterior_W(self):
        mean_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]
        log_var_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]

        return mean_W, log_var_W
    
    ## Function to initialize the filters and bias of CNN structure
    def init_param_CNN(self):
        filters1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), tf.float32, name="filters1")
        bias1 = tf.Variable(tf.truncated_normal([32], stddev=0.1), tf.float32, name="bias1")
        filters2= tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), tf.float32, name="filters2")
        bias2= tf.Variable(tf.truncated_normal([64], stddev=0.1), tf.float32, name="bias2")
        return filters1, bias1, filters2, bias2

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (Omega and W) when q(Omega) is to be learned 
    def get_kl_Omega_to_learn(self):
        kl = 0
        for i in range(self.n_Omega):
            kl = kl + utils.DKL_gaussian(self.mean_Omega[i], self.log_var_Omega[i], self.prior_mean_Omega[i], self.log_prior_var_Omega[i])
        for i in range(self.n_W):
            kl = kl + utils.DKL_gaussian(self.mean_W[i], self.log_var_W[i], self.prior_mean_W[i], self.log_prior_var_W[i])
        return kl

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (W only) when q(Omega) is not to be learned 
    def get_kl_Omega_fixed(self):
        kl = 0
        for i in range(self.n_W):
            kl = kl + utils.DKL_gaussian(self.mean_W[i], self.log_var_W[i], self.prior_mean_W[i], self.log_prior_var_W[i])
        return kl

    ## Returns samples from approximate posterior over Omega
    def sample_from_Omega_to_learn(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = utils.get_normal_samples(self.mc, self.d_in[i], self.d_out[i])
            Omega_from_q.append(tf.add(tf.mul(z, tf.exp(self.log_var_Omega[i] / 2)), self.mean_Omega[i]))

        return Omega_from_q

    ## Returns Omega values calculated from fixed random variables and mean and variance of q() - the latter are optimized and enter the calculation of the KL so also lengthscale parameters get optimized
    def sample_from_Omega_optim(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = tf.mul(self.z_for_Omega_fixed[i], tf.ones([self.mc, self.d_in[i], self.d_out[i]]))
            Omega_from_q.append(tf.add(tf.mul(z, tf.exp(self.log_var_Omega[i] / 2)), self.mean_Omega[i]))

        return Omega_from_q

    ## Returns samples from prior over Omega - in this case, randomness is fixed throughout learning (and Monte Carlo samples)
    def sample_from_Omega_fixed(self):
        Omega_from_q = []
        for i in range(self.n_Omega):
            z = tf.mul(self.z_for_Omega_fixed[i], tf.ones([self.mc, self.d_in[i], self.d_out[i]]))

            if self.is_ard == True:
                reshaped_log_prior_var_Omega = tf.tile(tf.reshape(self.log_prior_var_Omega[i] / 2, [self.d_in[i],1]), [1,self.d_out[i]])
                Omega_from_q.append(tf.mul(z, tf.exp(reshaped_log_prior_var_Omega)))
            if self.is_ard == False:
                Omega_from_q.append(tf.add(tf.mul(z, tf.exp(self.log_prior_var_Omega[i] / 2)), self.prior_mean_Omega[i]))

        return Omega_from_q

    ## Returns samples from approximate posterior over W
    def sample_from_W(self):
        W_from_q = []
        for i in range(self.n_W):
            z = utils.get_normal_samples(self.mc, self.dhat_in[i], self.dhat_out[i])
            W_from_q.append(tf.add(tf.mul(z, tf.exp(self.log_var_W[i] / 2)), self.mean_W[i]))
        return W_from_q

    ## Returns the expected log-likelihood term in the variational lower bound 
    def get_ell(self):
        Din = self.d_in[0]
        MC = self.mc
        N_L = self.nl
        X = self.X
        Y = self.Y
        batch_size = tf.shape(X)[0] # This is the actual batch size when X is passed to the graph of computations
        
        ## CNN structure
        x_images = tf.reshape(X, [-1, 28, 28, 1]) #[batch_size, 28, 28, 1]
        conv_bias1 = tf.add(tf.nn.conv2d(x_images, self.filters1, strides=[1,1,1,1], padding = "SAME"), self.bias1) #[batch_size, 28, 28, 32]
        relu1 = tf.nn.relu(conv_bias1) #[batch_size, 28, 28, 32]
        subsampling1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #[batch_size, 14, 14, 32]
        conv_bias2 = tf.add(tf.nn.conv2d(subsampling1, self.filters2, strides=[1,1,1,1], padding="SAME"), self.bias2) #[batch_size, 14, 14, 64]
        relu2 = tf.nn.relu(conv_bias2) #[batch_size, 14, 14, 64]
        subsampling2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #[batch_size, 7, 7, 64]
        featuresCNN = tf.reshape(subsampling2, [-1, 7 * 7 * 64]) #[batch_size, 7 * 7 * 64]
        
        ## The representation of the information is based on 3-dimensional tensors (one for each layer)
        ## Each slice [i,:,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.layer = []
        self.layer.append(tf.mul(tf.ones([self.mc, batch_size, self.Dfeatures]), featuresCNN))

        ## Forward propagate information from the input to the output through hidden layers
        Omega_from_q  = self.sample_from_Omega()
        W_from_q = self.sample_from_W()
        # TODO: basis features should be in a different class
        for i in range(N_L):
            layer_times_Omega = tf.batch_matmul(self.layer[i], Omega_from_q[i])  # X * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF": 
                # Cluster machine
                Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.cos(layer_times_Omega), tf.sin(layer_times_Omega)])
                # Local machine
                # Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat([tf.cos(layer_times_Omega), tf.sin(layer_times_Omega)], 2)
            if self.kernel_type == "arccosine": 
                if self.arccosine_degree == 0:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.sign(tf.maximum(layer_times_Omega, 0.0))])
                if self.arccosine_degree == 1:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.maximum(layer_times_Omega, 0.0)])
                if self.arccosine_degree == 2:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.square(tf.maximum(layer_times_Omega, 0.0))])

            F = tf.batch_matmul(Phi, W_from_q[i])
            if self.feed_forward and not (i == (N_L-1)): ## In the feed-forward case, no concatenation in the last layer so that F has the same dimensions of Y
                F = tf.concat(2, [F, self.layer[0]])

            self.layer.append(F)

        ## Output layer
        layer_out = self.layer[N_L]

        ## Given the output layer, we compute the conditional likelihood across all samples
        ll = self.likelihood.log_cond_prob(Y, layer_out)

        ## Mini-batch estimation of the expected log-likelihood term
        ell = tf.reduce_sum(tf.reduce_mean(ll, 0)) * self.num_examples / tf.cast(batch_size, "float32")

        return ell, layer_out

    ## Maximize variational lower bound --> minimize Nelbo
    def get_nelbo(self):
        kl = self.get_kl()
        ell, layer_out = self.get_ell()
        nelbo  = kl - ell
        return nelbo, kl, ell, layer_out

    ## Return predictions on some data
    def predict(self, testX, testY, mc_test):
        out = self.likelihood.predict(self.layer_out)
        
        nll = - tf.reduce_sum(-np.log(mc_test) + utils.logsumexp(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        uncertainties = tf.nn.softmax(self.layer_out, dim=-1)
        uncertainties = tf.reduce_mean(uncertainties, 0)
        uncertainties, pred, neg_ll = self.session.run([uncertainties, out, nll], feed_dict={self.X:testX, self.Y: testY, self.mc:mc_test})
        mean_pred = np.mean(pred, 0)
        return mean_pred, neg_ll, uncertainties

    ## Return the list of TF variables that should be "free" to be optimized
    def get_vars_fixing_some(self, all_variables):
        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == False):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == False):
            variational_parameters = all_variables

        return variational_parameters

    ## Function that learns the deep GP model with random Fourier feature approximation
    def learn(self, trainX, trainY, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, testX=None, testY=None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        total_train_time = 0

        if optimizer is None:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        ## Initialize all variables
        init = tf.global_variables_initializer()
        ##init = tf.initialize_all_variables()

        ## Fix any variables that are supposed to be fixed
        train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

        ## Initialize TF session
        self.session.run(init)

        ## Set the folder where the logs are going to be written 
        # summary_writer = tf.train.SummaryWriter('logs/', self.session.graph)
        summary_writer = tf.summary.FileWriter('logs/', self.session.graph)

        if not(less_prints):
            nelbo, kl, ell, _ =  self.session.run(self.get_nelbo(), feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
            print("Initial kl=" + repr(kl) + "  nell=" + repr(-ell) + "  nelbo=" + repr(nelbo), end=" ")
            print("  log-sigma2 =", self.session.run(self.log_theta_sigma2))
        
        train_size = len(trainX)
        test_size = len(testX)
        t = 10
        snapshot = t * display_step
        prefix = "mnist_" +  str(train_size) + "train_" + str(test_size) + "test_" + str(self.nl) + "nl_" + str(self.n_rff[0]) + "rff_" + str(self.df[0]) + "df_" \
                  + str(mc_train) + "mctrain_" + str(mc_test) + "mctest_" + str(batch_size) + "bs_" + str(self.theta_fixed) + "tf_" + str(self.feed_forward) + "ff_" \
                  + str(self.learn_Omega) + "Omega" + "_snap_"
        filename = prefix + str(0)
        file = open(filename, 'w')
        cur_iter = open(str(train_size) + "_" + str(0), 'w')
        start_id = 0
        end_id = start_id + batch_size
        ## Present data to DGP n_iterations times
        ## TODO: modify the code so that the user passes the number of epochs (number of times the whole training set is presented to the DGP)
        for iteration in range(n_iterations):
            
            if (iteration > 0) and (iteration % t == 0):
                os.rename(str(train_size) + "_" + str(iteration - t), str(train_size) + "_" + str(iteration))
                
            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

            ## Present one batch of data to the DGP
            start_train_time = current_milli_time()
            batch_X = trainX[start_id:end_id]
            batch_Y = trainY[start_id:end_id]
            start_id = start_id + batch_size
            end_id = end_id + batch_size
            if (end_id > train_size):
                start_id = 0
                end_id = batch_size

            monte_carlo_sample_train = mc_train
            if (current_milli_time() - start_train_time) < (1000 * 60 * duration / 2.0): 
                monte_carlo_sample_train = 1

            self.session.run(train_step, feed_dict={self.X: batch_X, self.Y: batch_Y, self.mc: monte_carlo_sample_train})
            total_train_time += current_milli_time() - start_train_time

            ## After reaching enough iterations with Omega fixed, unfix it
            if self.q_Omega_fixed_flag == True:
                if iteration >= self.q_Omega_fixed:
                    self.q_Omega_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            if self.theta_fixed_flag == True:
                if iteration >= self.theta_fixed:
                    self.theta_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            ## Display logs every "FLAGS.display_step" iterations
            if iteration % display_step == 0:
                start_predict_time = current_milli_time()

                if less_prints:
                    print("i=" + repr(iteration), end = " ")

                else:
                    nelbo, kl, ell, _ = self.session.run(self.get_nelbo(), feed_dict={self.X: trainX, self.Y: trainY, self.mc: mc_train})
                    print("i=" + repr(iteration)  + "  kl=" + repr(kl) + "  nell=" + repr(-ell)  + "  nelbo=" + repr(nelbo), end=" ")

                    print(" log-sigma2=", self.session.run(self.log_theta_sigma2), end=" ")
                    # print(" log-lengthscale=", self.session.run(self.log_theta_lengthscale), end=" ")
                    # print(" Omega=", self.session.run(self.mean_Omega[0][0,:]), end=" ")
                    # print(" W=", self.session.run(self.mean_W[0][0,:]), end=" ")

                if loss_function is not None:
                    pred, nll_test, uncertainties = self.predict(testX, testY, mc_test)
                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)
                    err_rate = loss_function.eval(testY, pred)
                    true_false = np.reshape((np.argmax(pred, 1) == np.argmax(testY, 1)), [len(pred), 1])
                    uncertainties = np.concatenate((true_false, uncertainties), axis=-1)
                    print(loss_function.get_name() + "=" + "%.9f" % err_rate, end = " ")
                    print(" nll_test=" + "%.9f" % (nll_test / len(testY)), end = " ")
                    file.write("%d\t%s\t%s\n" % (iteration, err_rate, nll_test / len(testY)))
                    if (iteration >= 0) and (iteration % snapshot == 0):
                        file.close()
                        filename = prefix + str(iteration)
                        file = open(filename, 'w')
                        # Save the uncertainties
                        np.savetxt(prefix + str(iteration) + "_uncer", uncertainties, fmt='%0.5f', delimiter='\t', newline='\n')
                        
                print(" time=" + repr(elapsed_time), end = " ")
                print("")