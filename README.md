# andi_xp_utils

andi_xp_utils is a Python package to support Bayesian A/B testing on various types of data, including binary, log-normal, and normal distributions. The package is defined by an abstract base class that outlines a process for data simulation / evaluation, and specialized subclasses that apply the process to specific probability distributions.


## Installation

Install the library using pip in google colab:

```bash
from getpass import getpass


github_username = "enter your user name"
github_pat = getpass(prompt='GitHub PAT:')


# Configure git
!git config --global user.name {github_username}
!git config --global url."https://{Personal_access_token}@github.com/".insteadOf "https://github.com/"

!pip install git+https://github.com/Groupe-Atallah/bi-analytics-insights.git@ab_testing#subdirectory=ab_testing/andi_xp_utils
```


## Usage

### BinaryDistribution
```python
from andi_xp_utils import BinaryDistribution

# Create a BinaryDistribution instance with specified parameters
total_samples = 10000000
proportion_treated = 0.5
baseline_conversion_rate = 0.01
expected_lift = 0.01

binary_dist = BinaryDistribution(
    total_samples=total_samples,
    proportion_treated=proportion_treated,
    baseline_conversion_rate=baseline_conversion_rate,
    expected_lift=expected_lift
)

# Simulate data for the baseline and treatment groups
baseline_data, treatment_data = binary_dist.simulate_data()

# Perform Bayesian analysis on the simulated data
baseline_posterior, treatment_posterior = binary_dist.analysis(baseline_data, treatment_data)

# Calculate the 90% credible intervals for the baseline and treatment posteriors
baseline_credible_interval = binary_dist.calculate_credible_interval(baseline_posterior)
treatment_credible_interval = binary_dist.calculate_credible_interval(treatment_posterior)

# Print the results
print("Baseline Posterior: ", baseline_posterior)
print("Treatment Posterior: ", treatment_posterior)
print("Baseline 90% Credible Interval: ", baseline_credible_interval)
print("Treatment 90% Credible Interval: ", treatment_credible_interval)

# Run the multiple Bayesian analyses and count the true directional results
num_runs = 100
num_true_directional_results = binary_dist.simulated_runs(num_runs)
print(f"Number of true directional results in {num_runs} runs: {num_true_directional_results}")

```






