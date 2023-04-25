import numpy as np
import pandas as pd
from scipy.stats import invgamma, t
from andi_xp_utils.abstract_distribution import AbstractDistribution




class LogNormalDistribution(AbstractDistribution):
    """
    A class representing a Bayesian A/B test with log-normal distributed outcomes
    (e.g., revenue per session or other continuous, positive data).

    This class is a subclass of the AbstractDistribution class and implements a Bayesian analysis
    using the Normal-Gamma conjugate prior for log-normal data.

    Attributes:
        total_samples (int): The total number of samples in the experiment.
        proportion_treated (float): The proportion of samples in the treatment group.
        mu_baseline (float): The mean of the log-normal distribution for the baseline group.
        sigma_baseline (float): The standard deviation of the log-normal distribution for the baseline group.
        mu_treatment (float): The mean of the log-normal distribution for the treatment group.
        sigma_treatment (float): The standard deviation of the log-normal distribution for the treatment group.
        num_samples_treatment (int): The number of samples in the treatment group.
        num_samples_baseline (int): The number of samples in the baseline group.
    """
    def __init__(self, total_samples, proportion_treated, baseline_mu, baseline_sigma, treatment_mu, treatment_sigma):
        """
        Initialize the LogNormalDistribution class with the given parameters.

        Args:
            total_samples (int): The total number of samples in the experiment.
            proportion_treated (float): The proportion of samples in the treatment group.
            baseline_mu (float): The mean of the log-normal distribution for the baseline group.
            baseline_sigma (float): The standard deviation of the log-normal distribution for the baseline group.
            treatment_mu (float): The mean of the log-normal distribution for the treatment group.
            treatment_sigma (float): The standard deviation of the log-normal distribution for the treatment group.
        """
        super().__init__(total_samples, proportion_treated)
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.treatment_mu = treatment_mu
        self.treatment_sigma = treatment_sigma
        self.proportion_treated = proportion_treated

    def simulate_data(self):
        # np.random.seed(self.seed)

        # Generate normal random data using logarithm of the mean and standard deviation
        baseline_normal_data = np.random.normal(loc=np.log(self.baseline_mu), scale=self.baseline_sigma, size=self.total_samples)
        treatment_normal_data = np.random.normal(loc=np.log(self.treatment_mu), scale=self.treatment_sigma, size=self.total_samples)

        # Take the exponent of the generated data
        baseline_data = np.exp(baseline_normal_data)
        treatment_data = np.exp(treatment_normal_data)
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
        """
        Perform Bayesian analysis on log-normal data using a Normal-Gamma conjugate prior.

        Given the observed data for the baseline and treatment groups, update the prior
        distribution with the observed data to obtain the posterior
        distribution for each group.

        Args:
            baseline_data (array-like): Log-normal data for the baseline group.
            treatment_data (array-like): Log-normal data for the treatment group.
            prior_mu_baseline (float): The prior mean for the normal distribution of the baseline group, defaults to 0.
            prior_nu_baseline (float): The prior number of observations for the normal distribution of the baseline group, defaults to 1.
            prior_alpha_baseline (float): The prior shape parameter for the gamma distribution of the baseline group, defaults to 1.
            prior_beta_baseline (float): The prior scale parameter for the gamma distribution of the baseline group, defaults to 1.
            prior_mu_treatment (float): The prior mean for the normal distribution of the treatment group, defaults to 0.
            prior_nu_treatment (float): The prior number of observations for the normal distribution of the treatment group, defaults to 1.
            prior_alpha_treatment (float): The prior shape parameter for the gamma distribution of the treatment group, defaults to 1.
            prior_beta_treatment (float): The prior scale parameter for the gamma distribution of the treatment group, defaults to 1.

        Returns:
            tuple: A tuple containing the posterior distributions (as tuples of mu, nu, alpha, and beta parameters)
                for the baseline and treatment groups.
        """

        log_baseline_data = np.log(baseline_data)
        log_treatment_data = np.log(treatment_data)

        posterior_baseline = self._update_posterior(log_baseline_data, prior_mu_baseline, prior_nu_baseline, prior_alpha_baseline, prior_beta_baseline)
        posterior_treatment = self._update_posterior(log_treatment_data, prior_mu_treatment, prior_nu_treatment, prior_alpha_treatment, prior_beta_treatment)

        return posterior_baseline, posterior_treatment


    def calculate_credible_interval(self, posterior, alpha=0.9):
        """
        Calculate the credible interval for the given Normal-Gamma posterior distribution and alpha level.

        Args:
            posterior (tuple): The posterior distribution as a tuple (mu, nu, alpha, beta).
            alpha (float): The alpha level for the credible interval, default is 0.9 for a 90% interval.

        Returns:
            tuple: The lower and upper bounds of the credible interval for the posterior.
        """
        posterior_mu, posterior_nu, posterior_alpha, posterior_beta = posterior

        # Calculate the square root of the posterior variance
        sqrt_posterior_variance = np.sqrt(posterior_beta / (posterior_alpha * (posterior_nu + 1)))

        # Calculate the lower and upper bounds for the credible interval
        lower_bound = np.exp(posterior_mu - sqrt_posterior_variance * t.ppf((1 + alpha) / 2, 2 * posterior_alpha))
        upper_bound = np.exp(posterior_mu + sqrt_posterior_variance * t.ppf((1 + alpha) / 2, 2 * posterior_alpha))

        return lower_bound, upper_bound


    def required_sample_size(self, n_runs=1000, directional_accuracy=0.9, credible_interval_coverage=0.9, start_sample_size=1000, size_increments = 100,  return_df=False):
        """
        Calculate the required sample size for a desired directional accuracy and credible interval coverage
        in a log-normal A/B test with a specified baseline value, expected lift, and proportion treated.

        Args:
            baseline_value (float): The baseline log-normal mean.
            expected_lift (float): The expected lift in the treatment group relative to the baseline group.
            proportion_treated (float): The proportion of samples in the treatment group.
            n_runs (int): The number of simulation runs, default is 1000.
            directional_accuracy (float): The desired directional accuracy, default is 0.9 (90%).
            credible_interval_coverage (float): The desired credible interval coverage, default is 0.9 (90%).
            start_sample_size (int): The starting sample size, default is 1000.
            size_increments (int): The number of sample size increments to use, default is 100.
            return_df (bool): Whether to return a DataFrame with sample size and directional accuracy, default is False.

        Returns:
            int or DataFrame: The required sample size to achieve the desired directional accuracy
                            and credible interval coverage, or a DataFrame with sample size and
                            corresponding directional accuracy if return_df is True.
        """
        sample_size = start_sample_size
        directional_accuracies = []

        while True:
            self.total_samples = sample_size
            self.num_samples_treatment = int(np.ceil(self.total_samples * self.proportion_treated))
            self.num_samples_baseline = self.total_samples - self.num_samples_treatment

            directional_accuracy_count = 0

            for _ in range(n_runs):
                baseline_data, treatment_data = self.simulate_data()
                posterior_baseline, posterior_treatment = self.analysis(baseline_data, treatment_data)
                lb_baseline, ub_baseline = self.calculate_credible_interval(posterior_baseline, credible_interval_coverage)
                lb_treatment, ub_treatment = self.calculate_credible_interval(posterior_treatment, credible_interval_coverage)

                if (ub_baseline < lb_treatment):
                    directional_accuracy_count += 1

            directional_accuracy_rate = directional_accuracy_count / n_runs
            directional_accuracies.append(directional_accuracy_rate)

            if directional_accuracy_rate >= directional_accuracy:
                break
            else:
                sample_size += size_increments

        if return_df:
            results_df = pd.DataFrame({
                'Sample Size': np.arange(1, len(directional_accuracies) + 1),
                'Directional Accuracy': directional_accuracies
            })
            return results_df
        else:
            return sample_size
