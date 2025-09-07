"""
Demo A/B analysis with bootstrap CI and CUPED adjustment.
Usage: python experiments/analyze_ab.py
"""
import numpy as np
from pipelines.cuped import cuped

def bootstrap_ci(data, n_boot=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(data)
    boots = [np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return lo, hi

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 10000
    pre_ctrl  = rng.normal(100, 20, n)
    pre_treat = rng.normal(100, 20, n)
    post_ctrl  = pre_ctrl  * 0.5 + rng.normal(0, 10, n)
    post_treat = pre_treat * 0.5 + 1.0 + rng.normal(0, 10, n)  # +1 lift

    naive = np.mean(post_treat) - np.mean(post_ctrl)
    lo, hi = bootstrap_ci(post_treat - post_ctrl)
    print(f"Naive lift: {naive:.3f}  |  95% CI ~ ({lo:.3f}, {hi:.3f})")

    naive_eff, cuped_eff, theta = cuped(pre_ctrl, pre_treat, post_ctrl, post_treat)
    print(f"CUPED lift: {cuped_eff:.3f} | theta={theta:.3f}")
