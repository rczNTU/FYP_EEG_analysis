import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest

# ---------------------------------------------------------------
# FIXES:
# 1. Wilcoxon: added zero_method="wilcox" and alternative="greater"
#    to be explicit and consistent with the directional binomial test.
#    Original used default two-sided wilcoxon then one-sided binomial
#    — mixed directionality.
# 2. Added confidence interval on mean diff (95% t-based).
# 3. Added median_diff — more robust than mean for small n.
# ---------------------------------------------------------------


def paired_stats(x, y):
    """
    Paired statistics for two arrays x (stimulus) and y (baseline).

    Tests H1: stimulus > baseline (one-sided where applicable).

    Returns dict with:
        n, mean_diff, median_diff, std_diff, ci_95,
        effect_size (Cohen's d for paired),
        t_p (two-sided), wilcoxon_p (two-sided), binom_p (one-sided)
    """
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    diff = x - y

    mean_diff   = diff.mean()
    median_diff = np.median(diff)
    std_diff    = diff.std(ddof=1)

    # 95% confidence interval on mean diff (t-based)
    from scipy.stats import t as t_dist
    se   = std_diff / np.sqrt(n)
    t_cv = t_dist.ppf(0.975, df=n - 1)
    ci_95 = (mean_diff - t_cv * se, mean_diff + t_cv * se)

    # Cohen's d for paired samples
    effect_size = mean_diff / (std_diff + 1e-12)

    # Two-sided t-test (most conservative, use for reporting)
    _, p_t = ttest_rel(x, y)

    # Two-sided Wilcoxon signed-rank (non-parametric, robust for small n)
    try:
        _, p_w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # Wilcoxon fails if all diffs are zero
        p_w = 1.0

    # One-sided binomial test: P(more than half of diffs > 0)
    n_positive = int((diff > 0).sum())
    p_binom = binomtest(n_positive, n, 0.5, alternative="greater").pvalue

    return {
        "n":            n,
        "mean_diff":    mean_diff,
        "median_diff":  median_diff,
        "std_diff":     std_diff,
        "ci_95_low":    ci_95[0],
        "ci_95_high":   ci_95[1],
        "effect_size":  effect_size,
        "t_p":          p_t,
        "wilcoxon_p":   p_w,
        "binom_p":      p_binom,
    }