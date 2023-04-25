import numpy as np
import pandas as pd
from scipy.stats import invgamma, t
from andi_xp_utils.abstract_distribution import AbstractDistribution


class NormalDistribution(AbstractDistribution):
    """
    A class representing a Bayesian A/B test with normally distributed outcomes
    (e.g., AOV or other continuous data).

    This class is a subclass of the AbstractDistribution class and implements a Bayesian analysis
    using the Normal-Inverse-Gamma conjugate prior for normal data.
    """

    def __init__(self, total_samples, proportion_treated, baseline_mu, baseline_sigma, treatment_mu, treatment_sigma):
        super().__init__(total_samples, proportion_treated)
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.treatment_mu = treatment_mu
        self.treatment_sigma = treatment_sigma
        self.proportion_treated = proportion_treated

    def simulate_data(self):
        baseline_data = np.random.normal(loc=self.baseline_mu, scale=self.baseline_sigma, size=self.total_samples)
        treatment_data = np.random.normal(loc=self.treatment_mu, scale=self.treatment_sigma, size=self.total_samples)
        return baseline_data, treatment_data
    
    @classmethod
    def _update_posterior(self, data, prior_mu, prior_nu, prior_alpha, prior_beta):
        n = len(data)
        x_bar = np.mean(data)
        s2 = np.var(data, ddof=1)

        posterior_mu = (prior_nu * prior_mu + n * x_bar) / (prior_nu + n)
        posterior_nu = prior_nu + n
        posterior_alpha = prior_alpha + n / 2
        posterior_beta = prior_beta + 0.5 * (n * s2 + (prior_nu * n * (x_bar - prior_mu)**2) / (prior_nu + n))

        return posterior_mu, posterior_nu, posterior_alpha, posterior_beta
    
    @classmethod
    def analysis(self, baseline_data, treatment_data, prior_mu_baseline=0, prior_nu_baseline=1, prior_alpha_baseline=1, prior_beta_baseline=1,
                        prior_mu_treatment=0, prior_nu_treatment=1, prior_alpha_treatment=1, prior_beta_treatment=1):
        
        posterior_baseline = self._update_posterior(baseline_data, prior_mu_baseline, prior_nu_baseline, prior_alpha_baseline, prior_beta_baseline)
        posterior_treatment = self._update_posterior(treatment_data, prior_mu_treatment, prior_nu_treatment, prior_alpha_treatment, prior_beta_treatment)

        return posterior_baseline, posterior_treatment

    def calculate_credible_interval(self, posterior, alpha=0.9):
        posterior_mu, posterior_nu, posterior_alpha, posterior_beta = posterior

        sqrt_posterior_variance = np.sqrt(posterior_beta / (posterior_alpha * (posterior_nu + 1)))

        lower_bound = posterior_mu - sqrt_posterior_variance * t.ppf((1 + alpha) / 2, 2 * posterior_alpha)
        upper_bound = posterior_mu + sqrt_posterior_variance * t.ppf((1 + alpha) / 2, 2 * posterior_alpha)

        return lower_bound, upper_bound

