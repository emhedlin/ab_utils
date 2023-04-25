import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class AbstractDistribution(ABC):

    def __init__(self, total_samples, proportion_treated):
        self.total_samples = total_samples
        self.num_samples_treatment = int(total_samples * proportion_treated)
        self.num_samples_baseline = total_samples - self.num_samples_treatment

    @abstractmethod
    def simulate_data(self):
        pass

    @abstractmethod
    def analysis(self, baseline_data, treatment_data, *args, **kwargs):
        """
        Perform Bayesian analysis on the given data and return the posteriors for the baseline and treatment.
        """
        pass

    def simulated_runs(self, n_runs, alpha=0.9, *args, **kwargs):
        """
        Run the analysis method multiple times and return the number of times
        the 90% credible interval of the difference between the treatment and baseline
        posteriors does not overlap zero.

        Args:
            n_runs (int): The number of times to run the analysis method.
            alpha (float): The alpha level for the credible interval. Default is 0.9, but higher values will provide
            more conservative results.
            *args, **kwargs: Any additional arguments and keyword arguments that should be
                             passed to the analysis method.

        Returns:
            int: The number of times the 90% credible interval of the difference between
                 the treatment and baseline posteriors does not overlap zero.
        """
        true_directional_results_count = 0

        for _ in range(n_runs):
            baseline_data, treatment_data = self.simulate_data()
            posterior_baseline, posterior_treatment = self.analysis(baseline_data, treatment_data, *args, **kwargs)

            # Calculate the credible intervals for the posteriors - credibles specified by alpha argument.
            lower_bound_baseline, upper_bound_baseline = self.calculate_credible_interval(posterior_baseline,
                                                                                          alpha=alpha)
            lower_bound_treatment, upper_bound_treatment = self.calculate_credible_interval(posterior_treatment,
                                                                                            alpha=alpha)

            # Check if the credible intervals do not overlap zero.
            if lower_bound_treatment - upper_bound_baseline > 0:
                true_directional_results_count += 1

        return true_directional_results_count

    @abstractmethod
    def calculate_credible_interval(self, posterior, alpha=0.9):
        """
        Calculate the credible interval for the given posterior distribution and alpha level.

        Args:
            posterior: The posterior distribution. This can be any object that the specific
                       distribution's implementation of this method can work with.
            alpha (float): The alpha level for the credible interval, default is 0.9 for a 90% interval.

        Returns:
            tuple: The lower and upper bounds of the credible interval for the posterior.
        """
        pass

    def required_sample_size(self, n_runs=1000, directional_accuracy=0.9, credible_interval_coverage=0.9,
                             start_sample_size=1000, size_increments=100, return_df=False):
        """
        Calculate the required sample size for a desired directional accuracy and credible interval coverage
        in a normal A/B test with a specified baseline value, expected lift, and proportion treated.

        Args:
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
                _, ub_baseline = self.calculate_credible_interval(posterior_baseline, credible_interval_coverage)
                lb_treatment, _ = self.calculate_credible_interval(posterior_treatment, credible_interval_coverage)

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
                'sample_size': np.arange(start_sample_size, sample_size + 1, size_increments)[
                               :len(directional_accuracies)],
                'directional_accuracy': directional_accuracies
            })
            return results_df
        else:
            return sample_size
