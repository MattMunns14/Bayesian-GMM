import numpy as np
import random as rand
from scipy.stats import norm, dirichlet, beta, gamma
from statistics import mean
import matplotlib.pyplot as plt
import math


class MCMC_GMM():
    def __init__(self, n_iter, data, k):
        self.rho = []
        self.mu = []
        self.phi = []
        self.k = k
        for i in range(k):
            self.mu.append(rand.uniform(0, 1))
            self.phi.append(rand.uniform(0, .1))

        self.mu = [0.0, 1]
        self.rho = [0.5, 0.5]
        self.phi = [0.1, 0.1]

        self.rho_posterior = []
        self.mu_posterior = []
        self.phi_posterior = []
        for i in range(k):
            self.rho_posterior.append([])
            self.mu_posterior.append([])
            self.phi_posterior.append([])
        self.n_iter = n_iter
        self.data = data
        self.burn_in = int(n_iter / 2)
        self.gibbsSampler(self.rho, self.mu, self.phi, self.rho_posterior,
                          self.mu_posterior, self.phi_posterior, self.n_iter, self.data, self.k, self.burn_in)

    def gibbsSampler(self, rho_list, mu_list, phi_list, rho_posterior, mu_posterior, phi_posterior, n_iter, data, k,
                     burn_in):
        self.count = 0
        while self.count < n_iter:
            n_list, label_array = self.updateZ(rho_list, mu_list, phi_list, data, k)
            rho_list, rho_posterior = self.updateRho(rho_list, data, n_list, k, rho_posterior)
            phi_list, phi_posterior = self.updatePhi(data, n_list, mu_list, k, label_array, phi_posterior)
            mu_list, mu_posterior = self.updateMu(data, phi_list, n_list, k, label_array, mu_posterior, )

            self.count += 1
            print("Iter: ", self.count)
        for i in range(k):
            phi_posterior[i] = phi_posterior[i][burn_in:]
            mu_posterior[i] = mu_posterior[i][burn_in:]
            rho_posterior[i] = rho_posterior[i][burn_in:]

    def updateMu(self, data, phi_list, n_list, k, label_array, mu_posterior, ):
        mu_list = []
        mean_list = []
        xbar_list = []
        alpha_list = []
        data_list = []
        initial_alpha = 1
        initial_m = 1

        for i in range(k):
            alpha_list.append(initial_alpha + n_list[i])
            data_list.append([])
        for i in range(len(data)):
            data_list[int(label_array[i])].append(data[i])

        for i in range(k):
            try:
                xbar_list.append(mean(data_list[i]))
            except:
                print('except')
                xbar_list.append(1)

        for i in range(k):
            mean_list.append((initial_alpha * initial_m + n_list[i] * xbar_list[i]) / (initial_alpha + n_list[i]))
        for i in range(k):
            r = norm.rvs(mean_list[i], (1 / (alpha_list[i] * phi_list[i]) ** (1 / 2)))
            mu_list.append(r)
            mu_posterior[i].append(r)

        return mu_list, mu_posterior

    def updatePhi(self, data, n_list, mu_list, k, label_array, phi_posterior):

        initial_alpha = 1
        initial_beta = 1
        phi_list = []
        alpha_list = []
        data_list = []
        beta_list = []
        for i in range(k):
            alpha_list.append(initial_alpha + n_list[i])
            data_list.append([])
        for i in range(len(data)):
            data_list[int(label_array[i])].append(data[i])

        for i in range(k):
            sumOf = 0
            for j in range(len(data_list[i])):
                sumOf += (data_list[i][j] - mu_list[i]) ** 2
            beta_list.append(initial_beta + sumOf)
        for i in range(k):
            r = gamma.rvs((initial_alpha + n_list[i]) / 2, (initial_beta + beta_list[i]) / 2)
            phi_list.append(r)
            phi_posterior[i].append(r)

        return phi_list, phi_posterior

    def updateZ(self, initial_rho, initial_mu, initial_phi, data, k):
        n_list = []
        label_array = np.zeros(len(data))
        for i in range(k):
            n_list.append(0)

        for i in range(len(data)):
            prob_list = []
            prob_sum = 0

            for j in range(k):
                prob_k = initial_rho[j] * (initial_phi[j]) ** (1 / 2) * (
                    math.exp(-(initial_phi[j]) * (1 / 2) * (data[i] - initial_mu[j]) ** 2))
                prob_list.append(prob_k)
                prob_sum += prob_k
            prob_list = [x / prob_sum for x in prob_list]

            s = np.random.random_sample()

            # s = norm.rvs(0.5,.25)
            ### THIS CODE IS SPECIFIC TO K=2###
            if s < prob_list[0]:
                label_array[i] = 0
                n_list[0] += 1

            else:
                label_array[i] = 1
                n_list[1] += 1
            ####################################

        return n_list, label_array

    def updateRho(self, initial_rho, data, n_list, k, rho_posterior):
        param_list = []
        for j in range(k):
            param_list.append(1 + n_list[j])

        rand_dir = dirichlet.rvs(param_list)
        for i in range(k):
            rho_posterior[i].append(rand_dir[0][i])
            initial_rho[i] = rand_dir[0][i]
        return initial_rho, rho_posterior


class GMM_Generator():
    def __init__(self, mu, phi, rho, k):
        self.mu = mu
        self.phi = phi
        self.rho = rho
        self.k = k
        self.num_samples = 1000
        self.distribution = []
        self.generateDistribution(self.mu, self.phi, self.rho, self.k, self.num_samples, self.distribution)

    def generateDistribution(self, mu, phi, rho, k, num_samples, distribution):
        for n in range(num_samples):
            s = np.random.random_sample()

            #### SPECIFIC TO K=2 CASE ####
            if s < rho[0]:
                sample_mu = mu[0]
                sample_phi = phi[0]
            else:
                sample_mu = mu[1]
                sample_phi = phi[1]
            ###############################

            x = norm.rvs(sample_mu, (1 / sample_phi) ** (1 / 2))
            distribution.append(x)


def distribution_generator(mu_posterior, phi_posterior, rho_posterior, k):
    mu_list = []
    phi_list = []
    rho_list = []
    for i in range(k):
        mu_list.append(mean(mu_posterior[i]))
        phi_list.append(mean(phi_posterior[i]))
        rho_list.append(mean(rho_posterior[i]))
    phi_list = [100, 100]
    generated_distribution = GMM_Generator(mu_list, phi_list, rho_list, k).distribution

    return generated_distribution


def graph_posterior(mu_posterior, phi_posterior, rho_posterior, k):
    for i in range(k):
        mu_posterior_k = mu_posterior[i]
        phi_posterior_k = phi_posterior[i]
        rho_posterior_k = rho_posterior[i]
        mu_string = "Posterior of $\mu_{}$".format(i)
        phi_string = "Posterior of $\phi_{}$".format(i)
        rho_string = "Posterior of $\\rho_{}$".format(i)

        fig=plt.figure()
        plt.hist(mu_posterior_k, 50)
        plt.title(mu_string)
        fig.savefig(mu_string+".png")
        plt.close()

        fig=plt.figure()
        plt.hist(phi_posterior_k, 50)
        plt.title(phi_string)
        fig.savefig(phi_string+".png")
        plt.close()

        fig = plt.figure()
        plt.hist(rho_posterior_k, 50)
        plt.title(rho_string)
        fig.savefig(rho_string+".png")
        plt.close()


def print_mean(mu_posterior, phi_posterior, rho_posterior, k):
    for i in range(k):
        print('Mean of mu', i, 'is', mean(mu_posterior[i]))
        print('Mean of phi', i, 'is', mean(phi_posterior[i]))
        print('Mean of rho', i, 'is', mean(rho_posterior[i]))


if __name__ == "__main__":
    data = np.genfromtxt('data2.txt', delimiter=',')
    data_mean = np.mean(data)

    fig = plt.figure()
    plt.hist(data, 50)
    title = "Original Data"
    plt.title(title)
    fig.savefig(title+".png")
    plt.close()

    iterations = 20000
    k = 2
    print("Start MCMC: ")
    mcmc_run = MCMC_GMM(iterations, data, k)
    mu_posterior = mcmc_run.mu_posterior
    phi_posterior = mcmc_run.phi_posterior
    rho_posterior = mcmc_run.rho_posterior

    graph_posterior(mu_posterior, phi_posterior, rho_posterior, k)

    generated_distribution = distribution_generator(mu_posterior, phi_posterior, rho_posterior, k)

    fig = plt.figure()
    plt.hist(generated_distribution, 50)
    plt.title("Generated Distribution")
    plt.savefig("Generated Distribution.png")
    plt.close()

    print_mean(mu_posterior, phi_posterior, rho_posterior, k)
