from andi_xp_utils.binary_distribution import BinaryDistribution


total_samples = 1000
proportion_treated = 0.5
baseline_conversion_rate = 0.01
expected_lift = 0.005

binary_dist = BinaryDistribution(total_samples, proportion_treated, baseline_conversion_rate, expected_lift)

binary_dist.required_sample_size(start_sample_size = 5000, size_increments=1000)

def test_binary_distribution():
    total_samples = 1000
    proportion_treated = 0.5
    baseline_conversion_rate = 0.01
    expected_lift = 0.005

    binary_dist = BinaryDistribution(total_samples, proportion_treated, baseline_conversion_rate, expected_lift)

    # Test simulate_data
    baseline_data, treatment_data = binary_dist.simulate_data()
    assert len(baseline_data) == binary_dist.num_samples_baseline
    assert len(treatment_data) == binary_dist.num_samples_treatment

    # Test analysis
    prior_alpha = 1
    prior_beta = 1
    posterior_baseline, posterior_treatment = binary_dist.analysis(baseline_data, treatment_data, prior_alpha, prior_beta)
    assert len(posterior_baseline) == 2
    assert len(posterior_treatment) == 2

    # Test calculate_credible_interval
    alpha = 0.9
    lb_baseline, ub_baseline = binary_dist.calculate_credible_interval(posterior_baseline, alpha)
    lb_treatment, ub_treatment = binary_dist.calculate_credible_interval(posterior_treatment, alpha)
    assert 0 <= lb_baseline <= ub_baseline <= 1
    assert 0 <= lb_treatment <= ub_treatment <= 1

    # Test required_sample_size
    directional_accuracy = 0.9
    credible_interval_coverage = 0.9
# Test required_sample_size
    n_runs = 10  # Reduce the number of simulation runs
    sample_size = binary_dist.required_sample_size(baseline_conversion_rate, expected_lift, proportion_treated,
                                                n_runs=n_runs,
                                                start_sample_size = 1000,
                                                directional_accuracy=directional_accuracy,
                                                credible_interval_coverage=credible_interval_coverage)

    assert isinstance(sample_size, int)
    assert sample_size > 0
