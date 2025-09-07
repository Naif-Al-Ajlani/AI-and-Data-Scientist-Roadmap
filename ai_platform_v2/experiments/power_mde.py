"""
Compute power and MDE for a two-sample test on means.
This is aligned with AB testing and MDE concepts in the roadmap.
Usage: python experiments/power_mde.py
"""
import math

def samples_for_mde(std, mde, alpha=0.05, power=0.8):
    # normal approx (two-sided)
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * std / mde)**2
    return math.ceil(n)

def mde_for_samples(std, n_per_group, alpha=0.05, power=0.8):
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    mde = (z_alpha + z_beta) * std * ( (2 / n_per_group) ** 0.5 )
    return mde

if __name__ == "__main__":
    std = 10.0
    mde = 1.0
    n = samples_for_mde(std, mde)
    print(f"Required n/group for std={std}, MDE={mde}: ~{n}")
    print(f"MDE for n=5000/group, std=10 → {mde_for_samples(10, 5000):.3f}")
