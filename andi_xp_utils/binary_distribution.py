import numpy as np
import pandas as pd
from scipy.stats import beta
from andi_xp_utils.abstract_distribution import AbstractDistribution


class BinaryDistribution(AbstractDistribution):
    """
    A class representing a Bayesian A/B test with binary outcomes (e.g., success/failure, conversion/non-conversion).

    This class is a subclass of the AbstractDistribution class and implements a Bayesian analysis
    using the Beta-Binomial conjugate prior for binary data.

    Attributes:
        total_samples (int): The total number of samples in the experiment.
        proportion_treated (float): The proportion of samples in the treatment group.
        baseline_conversion_rate (float): The conversion rate for the baseline group.
        treatment_conversion_rate (float): The conversion rate for the treatment group.
        num_samples_treatment (int): The number of samples in the treatment group.
        num_samples_baseline (int): The number of samples in the baseline group.
    """
    def __init__(self, total_samples, proportion_treated, baseline_conversion_rate, expected_lift):
        """
        Initialize the BinaryDistribution class with the given parameters.

        Args:
            total_samples (int): The total number of samples in the experiment.
            proportion_treated (float): The proportion of samples in the treatment group.
            baseline_conversion_rate (float): The conversion rate for the baseline group.
            expected_lift (float): The expected lift in conversion rate for the treatment group.
        """
        super().__init__(total_samples, proportion_treated)
        self.baseline_conversion_rate = baseline_conversion_rate
        self.treatment_conversion_rate = baseline_conversion_rate + expected_lift
        self.proportion_treated = proportion_treated


    def simulate_data(self):
        """
        Simulate binary data for the baseline and treatment groups using the specified conversion rates.

        Returns:
            tuple: A tuple containing two NumPy arrays, one for the baseline data and one for the treatment data.
        """
        baseline_data = np.random.binomial(n=1, p=self.baseline_conversion_rate, size=self.num_samples_baseline)
        treatment_data = np.random.binomial(n=1, p=self.treatment_conversion_rate, size=self.num_samples_treatment)
        return baseline_data, treatment_data

    @classmethod
    def analysis(self, baseline_data, treatment_data, prior_alpha=1, prior_beta=1):
        """
        Perform Bayesian analysis on binary data using a Beta-Binomial conjugate prior.

        Given the observed data for the baseline and treatment groups, update the prior
        distribution with the observed successes and failures to obtain the posterior
        distribution for each group.

        Args:
            baseline_data (array-like): Binary data for the baseline group (1 for success, 0 for failure).
            treatment_data (array-like): Binary data for the treatment group (1 for success, 0 for failure).
            prior_alpha (float): The alpha parameter of the Beta prior, defaults to 1.
            prior_beta (float): The beta parameter of the Beta prior, defaults to 1.

        Returns:
            tuple: A tuple containing the posterior distributions (as tuples of alpha and beta parameters)
                   for the baseline and treatment groups.
        """
        successes_baseline = np.sum(baseline_data)
        successes_treatment = np.sum(treatment_data)
        
        posterior_alpha_baseline = prior_alpha + successes_baseline
        posterior_beta_baseline = prior_beta + self.num_samples_baseline - successes_baseline
        
        posterior_alpha_treatment = prior_alpha + successes_treatment
        posterior_beta_treatment = prior_beta + self.num_samples_treatment - successes_treatment
        
        return (posterior_alpha_baseline, posterior_beta_baseline), (posterior_alpha_treatment, posterior_beta_treatment)

    def calculate_credible_interval(self, posterior, alpha=0.9):
        """
        Calculate the credible interval for the given Beta posterior distribution and alpha level.

        Args:
            posterior (tuple): The posterior distribution as a tuple (alpha, beta).
            alpha (float): The alpha level for the credible interval, default is 0.9 for a 90% interval.

        Returns:
            tuple: The lower and upper bounds of the credible interval for the posterior.
        """
        posterior_alpha, posterior_beta = posterior
        lower_bound = beta.ppf((1 - alpha) / 2, posterior_alpha, posterior_beta)
        upper_bound = beta.ppf(1 - (1 - alpha) / 2, posterior_alpha, posterior_beta)

        return lower_bound, upper_bound
