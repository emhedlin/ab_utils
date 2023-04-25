
# NormalDistribution Class

The `NormalDistribution` class represents a Bayesian A/B test with normally distributed outcomes, such as average order value (AOV) or other continuous data. It uses the Normal-Inverse-Gamma conjugate prior for normal data and is a subclass of the AbstractDistribution class.

## Initialization

The class takes the following parameters during initialization:

- `total_samples`: The total number of samples in each group (A and B).
- `proportion_treated`: The proportion of the samples in the treatment group.
- `baseline_mu`: The mean of the baseline group.
- `baseline_sigma`: The standard deviation of the baseline group.
- `treatment_mu`: The mean of the treatment group.
- `treatment_sigma`: The standard deviation of the treatment group.

## `simulate_data`

This method generates baseline and treatment data using the normal distribution with the specified mean and standard deviation.

## `_update_posterior`

This class method updates the posterior distribution based on the provided data and prior parameters. It takes the following arguments:


- `data`: The observed data.
- `prior_mu`: Prior mean.
- `prior_nu`: Prior sample size (degrees of freedom).
- `prior_alpha`: Prior shape parameter.
- `prior_beta`: Prior scale parameter.

The method calculates the updated posterior parameters using the following formulas:

$$
\begin{align}
    \texttt{posterior\_mu} &= \frac{\texttt{prior\_nu} \cdot \texttt{prior\_mu} + n \cdot \bar{x}}{\texttt{prior\_nu} + n} \\
    \texttt{posterior\_nu} &= \texttt{prior\_nu} + n \\
    \texttt{posterior\_alpha} &= \texttt{prior\_alpha} + \frac{n}{2} \\
    \texttt{posterior\_beta} &= \texttt{prior\_beta} + \frac{1}{2}\left(ns^2 + \frac{\texttt{prior\_nu} \cdot n \cdot (\bar{x} - \texttt{prior\_mu})^2}{\texttt{prior\_nu} + n}\right)
\end{align}
$$

These update rules are derived from the conjugate nature of the Normal-Inverse-Gamma distribution when used as a prior for normally distributed data with unknown mean and variance. In the context of the model, the shape parameter `alpha` plays a role in determining the "shape" of the Inverse-Gamma distribution for the variance ($\sigma^2$). As you collect more data points, you gain more information about the underlying population's variance. The updated $\texttt{posterior\_alpha}$ reflects this increase in information by adding $\frac{n}{2}$ to the prior shape parameter $\texttt{prior\_alpha}$. The division by 2 comes from the fact that we are using the sum of squared differences in our likelihood function. When dealing with squared differences, each pair of positive and negative deviations will produce the same squared result, effectively halving the contribution of the deviations to the shape parameter.
[Link to more thorough explanation](https://lmc2179.github.io/drafts/bayes_norm.html)

## `analysis`

This method conducts the Bayesian analysis using the provided prior parameters for the baseline and treatment groups. The method calls $\texttt{\_update\_posterior}$ for both groups and returns the updated posterior parameters.

## `calculate_credible_interval`

This method calculates the credible interval for a given posterior distribution and a specified confidence level. The method takes the following arguments:


- `posterior`: A tuple containing the posterior parameters: (`posterior_mu`, `posterior_nu`, `posterior_alpha`, `posterior_beta`).
- `alpha` (optional): The desired confidence level for the credible interval (default: 0.9).

The method computes the credible interval using the following formulas:

$$
\begin{align*}
    \texttt{sqrt\_posterior\_variance} &= \sqrt{\frac{\texttt{posterior\_beta}}{\texttt{posterior\_alpha} \cdot (\texttt{posterior\_nu} + 1)}} \\
    \texttt{lower\_bound} &= \texttt{posterior\_mu} - \texttt{sqrt\_posterior\_variance} \cdot t_{\frac{1 + \alpha}{2}}(2 \cdot \texttt{posterior\_alpha}) \\
    \texttt{upper\_bound} &= \texttt{posterior\_mu} + \texttt{sqrt\_posterior\_variance} \cdot t_{\frac{1 + \alpha}{2}}(2 \cdot \texttt{posterior\_alpha})
\end{align*}
$$

These formulas are based on the fact that the marginal distribution of the mean parameter ($\mu$) follows a scaled and shifted Student's t-distribution when using the Normal-Inverse-Gamma conjugate prior.

First, the method computes the square root of the posterior variance ($\texttt{sqrt\_posterior\_variance}$) for the mean parameter. The square root of the variance is calculated as the square root of the ratio of the posterior scale parameter ($\texttt{posterior\_beta}$) to the product of the posterior shape parameter ($\texttt{posterior\_alpha}$) and the sum of the posterior degrees of freedom ($\texttt{posterior\_nu}$) plus one.

Next, the method calculates the lower and upper bounds of the credible interval by subtracting and adding, respectively, the product of the square root of the posterior variance and the percent-point function (inverse of the cumulative distribution function) of the t-distribution with $2 \cdot \texttt{posterior\_alpha}$ degrees of freedom evaluated at the specified confidence level.

This method effectively calculates the region where the true mean parameter ($\mu$) lies with a probability equal to the specified confidence level (`alpha`).

