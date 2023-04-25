from andi_xp_utils.normal_distribution import NormalDistribution

normal = NormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)



def test_normal_distribution_init():
    normal = NormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    assert normal.total_samples == 100
    assert normal.proportion_treated == 0.5
    assert normal.baseline_mu == 1
    assert normal.baseline_sigma == 0.5
    assert normal.treatment_mu == 1.1
    assert normal.treatment_sigma == 0.6


def test_simulate_data():
    normal = NormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = normal.simulate_data()
    assert len(baseline_data) == 100
    assert len(treatment_data) == 100


def test_analysis():
    normal = NormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = normal.simulate_data()
    posterior_baseline, posterior_treatment = normal.analysis(baseline_data, treatment_data)
    assert len(posterior_baseline) == 4
    assert len(posterior_treatment) == 4
    assert all(isinstance(value, (float, int)) for value in posterior_baseline)
    assert all(isinstance(value, (float, int)) for value in posterior_treatment)



def test_calculate_credible_interval():
    normal = NormalDistribution(100, 0.5, 1, 0.5, 1.1, 0.6)
    baseline_data, treatment_data = normal.simulate_data()
    posterior_baseline, posterior_treatment = normal.analysis(baseline_data, treatment_data)
    lb_baseline, ub_baseline = normal.calculate_credible_interval(posterior_baseline)
    lb_treatment, ub_treatment = normal.calculate_credible_interval(posterior_treatment)
    assert isinstance(lb_baseline, float)
    assert isinstance(ub_baseline, float)
    assert isinstance(lb_treatment, float)
    assert isinstance(ub_treatment, float)


def test_required_sample_size():
    normal = NormalDistribution(100, 0.5, 10, 1, 11, 1)
    sample_size = normal.required_sample_size(1, 0.1, 0.5)
    assert isinstance(sample_size, int)
    assert 0 < sample_size <= 1000  # Adjust the upper limit based on your expectations


