"""
Compute naive vs CUPED-adjusted treatment effect with a demo.
Usage: python pipelines/cuped.py
"""
import numpy as np

def cuped(y_before_ctrl, y_before_treat, y_after_ctrl, y_after_treat):
    X = np.concatenate([y_before_ctrl, y_before_treat])
    Y = np.concatenate([y_after_ctrl,  y_after_treat])
    theta = np.cov(X, Y, ddof=1)[0,1] / np.var(X, ddof=1)
    y_after_ctrl_adj  = y_after_ctrl  - theta * (y_before_ctrl  - np.mean(X))
    y_after_treat_adj = y_after_treat - theta * (y_before_treat - np.mean(X))
    naive = np.mean(y_after_treat) - np.mean(y_after_ctrl)
    cuped_eff = np.mean(y_after_treat_adj) - np.mean(y_after_ctrl_adj)
    return naive, cuped_eff, theta

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 8000
    pre_ctrl  = rng.normal(100, 20, n)
    pre_treat = rng.normal(100, 20, n)
    post_ctrl  = pre_ctrl  * 0.5 + rng.normal(0, 10, n)
    post_treat = pre_treat * 0.5 + 1.0 + rng.normal(0, 10, n)  # true +1 lift
    naive, adj, theta = cuped(pre_ctrl, pre_treat, post_ctrl, post_treat)
    var_naive = np.var(post_treat - post_ctrl, ddof=1) / n
    var_adj   = np.var((post_treat - theta*(pre_treat - np.mean(np.r_[pre_ctrl, pre_treat]))) -
                       (post_ctrl  - theta*(pre_ctrl  - np.mean(np.r_[pre_ctrl, pre_treat]))),
                       ddof=1) / n
    print(f"Naive effect: {naive:.3f}  | Var ≈ {var_naive:.4f}")
    print(f"CUPED effect: {adj:.3f}   | Var ≈ {var_adj:.4f}")
    print(f"Variance reduction ≈ {(1 - var_adj/var_naive)*100:.1f}%  | theta={theta:.3f}")
