import numpy as np
import pandas as pd
from scipy.stats import invgamma, t, beta
from andi_xp_utils.abstract_distribution import AbstractDistribution
from andi_xp_utils.log_normal_distribution import LogNormalDistribution

class ZeroInflatedLogNormalDistribution(AbstractDistribution):
    def __init__(self, total_samples, proportion_treated, proportion_zeros, baseline_mu, baseline_sigma, treatment_mu, treatment_sigma):
        """
        Initialize the ZeroInflatedLogNormalDistribution class with the given parameters.

        Args:
            total_samples (int): The total number of samples in the experiment.
            proportion_treated (float): The proportion of samples in the treatment group.
            baseline_mu (float): The mean of the log-normal distribution for the baseline group.
            baseline_sigma (float): The standard deviation of the log-normal distribution for the baseline group.
            treatment_mu (float): The mean of the log-normal distribution for the treatment group.
            treatment_sigma (float): The standard deviation of the log-normal distribution for the treatment group.
        """
        super().__init__(total_samples, proportion_treated)
        self.proportion_zeros = proportion_zeros
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.treatment_mu = treatment_mu
        self.treatment_sigma = treatment_sigma


    def simulate_data(self):
        """
        Simulate zero-inflated log-normal data for the baseline and treatment groups using the specified parameters.

        Returns:
            tuple: A tuple containing two NumPy arrays, one for the baseline data and one for the treatment data.
        """
        baseline_data_nonzero = np.random.lognormal(mean=self.baseline_mu, sigma=self.baseline_sigma, size=int(self.num_samples_baseline * (1 - self.proportion_zeros)))
        treatment_data_nonzero = np.random.lognormal(mean=self.treatment_mu, sigma=self.treatment_sigma, size=int(self.num_samples_treatment * (1 - self.proportion_zeros)))

        baseline_data = np.concatenate((baseline_data_nonzero, np.zeros(int(self.num_samples_baseline * self.proportion_zeros))))
        treatment_data = np.concatenate((treatment_data_nonzero, np.zeros(int(self.num_samples_treatment * self.proportion_zeros))))

        np.random.shuffle(baseline_data)
        np.random.shuffle(treatment_data)

        return baseline_data, treatment_data

    def analysis(self, baseline_data, treatment_data, lognormal_kwargs=None, binary_kwargs=None):
        """
        Perform Bayesian analysis on zero-inflated log-normal data using a Normal-Gamma conjugate prior for log-normal data and a Beta prior for binary data.

        Given the observed data for the baseline and treatment groups, update the prior
        distribution with the observed data to obtain the posterior
        distribution for each group.

        Args:
            baseline_data (array-like): Zero-inflated log-normal data for the baseline group.
            treatment_data (array-like): Zero-inflated log-normal data for the treatment group.
            prior_mu (float): The prior mean for the normal distribution, defaults to 0.
            prior_nu (float): The prior number of observations for the normal distribution, defaults to 1.
            prior_alpha (float): The prior shape parameter for the gamma distribution, defaults to 1.
            prior_beta (float): The prior scale parameter for the gamma distribution, defaults to 1.
            prior_theta_baseline (float): The prior success rate (theta) for the Beta distribution of the baseline group, defaults to 0.5.
            prior_theta_treatment (float): The prior success rate (theta) for the Beta distribution of the treatment group, defaults to 0.5.

        Returns:
            tuple: A tuple containing the posterior distributions (as tuples of mu, nu, alpha, and beta parameters) for log-normal data
                   and Beta posterior distributions (as tuples of alpha and beta parameters) for binary data of the baseline and treatment groups.
        """        
        if lognormal_kwargs is None:
            lognormal_kwargs = {}

        if binary_kwargs is None:
            binary_kwargs = {}

        # Log-normal analysis for non-zero values
        lognormal_baseline_data = baseline_data[baseline_data > 0]
        lognormal_treatment_data = treatment_data[treatment_data > 0]

        lognormal_analysis = LogNormalDistribution(len(lognormal_baseline_data) + len(lognormal_treatment_data),
                                                   len(lognormal_treatment_data) / (len(lognormal_baseline_data) + len(lognormal_treatment_data)),
                                                   self.baseline_mu, self.baseline_sigma, self.treatment_mu, self.treatment_sigma)

        lognormal_posterior_baseline, lognormal_posterior_treatment = lognormal_analysis.analysis(lognormal_baseline_data, lognormal_treatment_data, **lognormal_kwargs)

        # Binary analysis for zero counts
        binary_baseline_data = (baseline_data == 0).astype(int)
        binary_treatment_data = (treatment_data == 0).astype(int)

        binary_posterior_baseline, binary_posterior_treatment = self.binary_analysis(binary_baseline_data, binary_treatment_data, **binary_kwargs)

        return (lognormal_posterior_baseline, binary_posterior_baseline), (lognormal_posterior_treatment, binary_posterior_treatment)

    def binary_analysis(self, baseline_data, treatment_data, alpha_prior=1, beta_prior=1):
        sum_baseline = np.sum(baseline_data)
        sum_treatment = np.sum(treatment_data)

        posterior_baseline = (alpha_prior + sum_baseline, beta_prior + len(baseline_data) - sum_baseline)
        posterior_treatment = (alpha_prior + sum_treatment, beta_prior + len(treatment_data) - sum_treatment)

        return posterior_baseline, posterior_treatment

    def calculate_credible_interval(self, posterior, alpha=0.9, method='lognormal'):
        """
        Calculate the credible interval for the given Normal-Gamma posterior distribution for log-normal data and Beta posterior distribution for binary data with the specified alpha level.

        Args:
            lognormal_posterior (tuple): The Normal-Gamma posterior distribution as a tuple (mu, nu, alpha, beta).
            binary_posterior (tuple): The Beta posterior distribution as a tuple (alpha, beta).
            alpha (float): The alpha level for the credible interval, default is 0.9 for a 90% interval.

        Returns:
            tuple: The lower and upper bounds of the credible interval for the combined zero-inflated log-normal distribution as a tuple (lower, upper).
        """

        if method == 'lognormal':
            return self._calculate_credible_interval_lognormal(posterior, alpha)
        elif method == 'binary':
            return self._calculate_credible_interval_binary(posterior, alpha)
        else:
            raise ValueError("Invalid method specified. Use 'lognormal' or 'binary'.")

    def _calculate_credible_interval_lognormal(self, posterior, alpha):
        posterior_mu, posterior_nu, posterior_alpha, posterior_beta = posterior
        sigma_squared_lower, sigma_squared_upper = invgamma.interval(alpha, posterior_alpha, scale=posterior_beta)

        t_alpha = t.ppf((1 + alpha) / 2, 2 * posterior_alpha)
        sqrt_nu_over_nu_plus1 = np.sqrt(posterior_nu / (posterior_nu + 1))

        lower_bound = posterior_mu - sqrt_nu_over_nu_plus1 * np.sqrt(sigma_squared_upper) * t_alpha
        upper_bound = posterior_mu + sqrt_nu_over_nu_plus1 * np.sqrt(sigma_squared_upper) * t_alpha

        return lower_bound, upper_bound

    def _calculate_credible_interval_binary(self, posterior, alpha):
        a, b = posterior
        lower_bound, upper_bound = beta.interval(alpha, a, b)

        return lower_bound, upper_bound


    def required_sample_size(self, n_runs=1000, directional_accuracy=0.9, credible_interval_coverage=0.9, return_df=False):
        """
        Calculate the required sample size for a desired directional accuracy and credible interval coverage
        in a zero-inflated log-normal A/B test with a specified baseline value, expected lift, and proportion treated.

        Args:
            baseline_mu (float): The baseline log-normal mean.
            expected_lift (float): The expected lift in the treatment group relative to the baseline group.
            proportion_treated (float): The proportion of samples in the treatment group.
            n_runs (int): The number of simulation runs, default is 1000.
            directional_accuracy (float): The desired directional accuracy, default is 0.9 (90%).
            credible_interval_coverage (float): The desired credible interval coverage, default is 0.9 (90%).
            return_df (bool): Whether to return a DataFrame with sample size and directional accuracy, default is False.

        Returns:
            int or DataFrame: The required sample size to achieve the desired directional accuracy
                            and credible interval coverage, or a DataFrame with sample size and
                            corresponding directional accuracy if return_df is True.
        """
        sample_size = 1
        directional_accuracies = []

        while True:
            self.total_samples = sample_size  # Directly set the total_samples attribute
            directional_accuracy_count = 0

            for _ in range(n_runs):
                baseline_data, treatment_data = self.simulate_data()
                posterior_baseline, posterior_treatment = self.analysis(baseline_data, treatment_data)

                lognormal_posterior_baseline, binary_posterior_baseline = posterior_baseline
                lognormal_posterior_treatment, binary_posterior_treatment = posterior_treatment



                lb_baseline_lognormal, ub_baseline_lognormal = self._calculate_credible_interval_lognormal(lognormal_posterior_baseline, credible_interval_coverage)
                lb_treatment_lognormal, ub_treatment_lognormal = self._calculate_credible_interval_lognormal(lognormal_posterior_treatment, credible_interval_coverage)

                lb_baseline_binary, ub_baseline_binary = self._calculate_credible_interval_binary(posterior_baseline, credible_interval_coverage)
                lb_treatment_binary, ub_treatment_binary = self._calculate_credible_interval_binary(posterior_treatment, credible_interval_coverage)

                if (ub_baseline_lognormal < lb_treatment_lognormal and self.baseline_mu < self.treatment_mu) or \
                (ub_treatment_lognormal < lb_baseline_lognormal and self.treatment_mu < self.baseline_mu) or \
                (ub_baseline_binary < lb_treatment_binary and self.baseline_mu < self.treatment_mu) or \
                (ub_treatment_binary < lb_baseline_binary and self.treatment_mu < self.baseline_mu):
                    directional_accuracy_count += 1

            directional_accuracy_rate = directional_accuracy_count / n_runs
            directional_accuracies.append(directional_accuracy_rate)

            if directional_accuracy_rate >= directional_accuracy:
                break
            else:
                sample_size += 1

        if return_df:
            results_df = pd.DataFrame({
                'Sample Size': np.arange(1, len(directional_accuracies) + 1),
                'Directional Accuracy': directional_accuracies
            })
            return results_df
        else:
            return sample_size