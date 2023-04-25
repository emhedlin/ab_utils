from andi_xp_utils.log_normal_distribution import LogNormalDistribution




def test_lognormal_distribution_init():
    lognormal = LogNormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    assert lognormal.total_samples == 100
    assert lognormal.proportion_treated == 0.5
    assert lognormal.baseline_mu == 1
    assert lognormal.baseline_sigma == 0.5
    assert lognormal.treatment_mu == 1.1
    assert lognormal.treatment_sigma == 0.6


def test_simulate_data():
    lognormal = LogNormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = lognormal.simulate_data()
    assert len(baseline_data) == 100
    assert len(treatment_data) == 100


def test_analysis():
    lognormal = LogNormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = lognormal.simulate_data()
    posterior_baseline, posterior_treatment = lognormal.analysis(baseline_data, treatment_data)
    assert len(posterior_baseline) == 4
    assert len(posterior_treatment) == 4
    assert all(isinstance(value, (float, int)) for value in posterior_baseline)
    assert all(isinstance(value, (float, int)) for value in posterior_treatment)



def test_calculate_credible_interval():
    lognormal = LogNormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = lognormal.simulate_data()
    posterior_baseline, posterior_treatment = lognormal.analysis(baseline_data, treatment_data)
    lb_baseline, ub_baseline = lognormal.calculate_credible_interval(posterior_baseline)
    lb_treatment, ub_treatment = lognormal.calculate_credible_interval(posterior_treatment)
    assert isinstance(lb_baseline, float)
    assert isinstance(ub_baseline, float)
    assert isinstance(lb_treatment, float)
    assert isinstance(ub_treatment, float)


def test_required_sample_size():
    lognormal = LogNormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    sample_size = lognormal.required_sample_size(1, 0.1, 0.5)
    assert isinstance(sample_size, int)
    assert 0 < sample_size <= 1000  # Adjust the upper limit based on your expectations


lognormal = LogNormalDistribution(100, 0.5, 10, 0.5, 12, 0.6)
baseline_data, treatment_data = lognormal.simulate_data()

import numpy as np
np.mean(treatment_data)

sample_size = lognormal.required_sample_size(
    start_sample_size = 10000, size_increments = 10000
    )



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lognormal = LogNormalDistribution(20, 0.5, 10, 0.5, 12, 0.6)


baseline_data, treatment_data = lognormal.simulate_data()

posterior_baseline, posterior_treatment = lognormal.analysis(baseline_data, treatment_data)


credible_interval_baseline = lognormal.calculate_credible_interval(posterior_baseline)
credible_interval_treatment = lognormal.calculate_credible_interval(posterior_treatment)

print("Credible interval for baseline group:", credible_interval_baseline)
print("Credible interval for treatment group:", credible_interval_treatment)

required_sample_size = lognormal.required_sample_size()
print("Required sample size for desired directional accuracy:", required_sample_size)
