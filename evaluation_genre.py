import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

data_genre = {
    'user': list(range(1, 21)),
    'original_mean': [
        0.735, 0.624, 0.682, 0.739, 0.767, 0.676, 0.756, 0.763, 0.672,
        0.842, 0.929, 0.671, 0.760, 0.816, 0.809, 0.887, 0.736, 0.827,
        0.766, 0.727
    ],
    'diversified_mean': [
        0.719, 0.595, 0.675, 0.736, 0.744, 0.648, 0.747, 0.756, 0.656,
        0.833, 0.905, 0.615, 0.750, 0.809, 0.798, 0.874, 0.729, 0.823,
        0.744, 0.711
    ]
}

df = pd.DataFrame(data_genre)

# Shapiro-Wilk test for normality on the differences
shapiro_test_original = stats.shapiro(df['original_mean'])
shapiro_test_diversified = stats.shapiro(df['diversified_mean'])


normal_distribution = True if (shapiro_test_original.pvalue > 0.05 and shapiro_test_diversified.pvalue > 0.05) else False

# Perform a paired t-test if both sets of data are normally distributed
# Otherwise, perform the Wilcoxon signed-rank test
if normal_distribution:
    t_test_result = stats.ttest_rel(df['original_mean'], df['diversified_mean'])
else:
    wilcoxon_test_result = stats.wilcoxon(df['original_mean'], df['diversified_mean'])

# Print the Shapiro-Wilk test results
print(f'Shapiro-Wilk Test for original mean: statistic={shapiro_test_original.statistic}, p-value={shapiro_test_original.pvalue}')
print(f'Shapiro-Wilk Test for diversified mean: statistic={shapiro_test_diversified.statistic}, p-value={shapiro_test_diversified.pvalue}')

# Print the results of the paired t-test or Wilcoxon signed-rank test
if normal_distribution:
    # print(f'Paired t-test: statistic={t_test_result.statistic}, p-value={t_test_result.pvalue}')
    print(f"Result is statistically significant: {t_test_result.pvalue <= 0.05} with 5% significance level")
else:
    print(f'Wilcoxon signed-rank test: statistic={wilcoxon_test_result.statistic}, p-value={wilcoxon_test_result.pvalue}')
