from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var, sample_size = 10, 1, 1000
    samples = np.random.normal(loc=mu, scale=var, size=sample_size)
    fit_univariate_gaussian = UnivariateGaussian().fit(samples)
    print((fit_univariate_gaussian.mu_, fit_univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    min_sample_size, max_sample_size, sample_size_step = 10, 1000, 10
    num_samples = int((min_sample_size + max_sample_size) / sample_size_step - 1)
    ms = np.linspace(min_sample_size, max_sample_size, num=num_samples)\
        .astype(int)

    differences = []
    for n in range(min_sample_size, max_sample_size, 10):
        # Add to differences the difference between the actual expectation and the estimated expectation
        # for the first n samples.
        difference = np.abs(mu - UnivariateGaussian().fit(samples[:n]).mu_)
        differences.append(difference)

    go.Figure([go.Scatter(x=ms, y=differences, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Distance between estimated expectation and the true expectation}$",
                               xaxis_title="$n\\text{ - number of samples}$",
                               yaxis_title="r$|\hat{\mu}-\mu|$",
                               height=300)).write_image("mean_deviation_over_sample_size.png")

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_values = fit_univariate_gaussian.pdf(samples)
    sorted_indexes = np.argsort(pdf_values)
    samples = samples[sorted_indexes]
    pdf_values = pdf_values[sorted_indexes]

    go.Figure([go.Scatter(x=samples, y=pdf_values, mode='lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Empirical PDF of fitted model in question 1}$",
                               xaxis_title="$n\\text{ - number of samples}$",
                               yaxis_title="$\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$$",
                               height=300)).write_image("empirical_pdf.png")

    # We expect to see a bell graph where the pdf value is higher the closer it gets to the mean value

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    num_samples = 1000
    samples = np.random.multivariate_normal(mean=mu, cov=cov, size=num_samples)
    multivariate_fit_gaussian = MultivariateGaussian().fit(samples)
    print(multivariate_fit_gaussian.mu_)
    print(multivariate_fit_gaussian.cov_)

    # Question 5 - Likelihood evaluation

    f1_values = np.linspace(-10, 10, 200)
    f3_values = np.linspace(-10, 10, 200)
    log_likelihoods = np.zeros((200, 200))
    i, j = 0, 0
    for f1_value in f1_values:
        for f3_value in f3_values:
            current_mu = np.array([f1_value, 0, f3_value, 0])
            log_likelihoods[i, j] = MultivariateGaussian.log_likelihood(current_mu, cov, samples)
            j += 1
        i += 1
        j = 0

    go.Figure([go.Heatmap(x=f1_values, y=f3_values, z=log_likelihoods)],
              layout=go.Layout(template="simple_white",
                               title=r"$\text{Log-likelihood of expectation depending on f1 and f3}$",
                               xaxis_title=r"$\mu_1$",
                               yaxis_title=r"$\mu_3$",
                               width=300,
                               height=300)).write_image("log_likelihoods.png")


    # Question 6 - Maximum likelihood
    argmax = log_likelihoods.argmax()
    coordinates = np.unravel_index(argmax, log_likelihoods.shape)
    argmax_feature_values = f1_values[list(coordinates)]
    print("Maximum Likelihood: ", np.round(argmax_feature_values, 3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
