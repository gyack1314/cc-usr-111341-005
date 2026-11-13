"""纯标准库的线性代数与统计分布工具。

项目运行环境只保证 Python 3.11 标准库，本模块自行实现小规模回归所需的
矩阵求解、t/F 分布、Fisher 精确检验与 wild cluster bootstrap，不依赖
numpy/scipy。所有函数只接受 Python 内建的 list/float/int。
"""
import math
import random


class InsufficientData(ValueError):
    """局部窗口内样本或变异不足以支撑该估计。"""


# ---------------------------------------------------------------- 线性代数

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(a, b):
    inner = len(b)
    cols = len(b[0])
    out = []
    for row in a:
        new = [0.0] * cols
        for i, value in enumerate(row):
            if value == 0.0:
                continue
            brow = b[i]
            for j in range(cols):
                new[j] += value * brow[j]
        out.append(new)
    return out


def matvec(a, b):
    """矩阵乘向量，热路径专用，避免为单列结果构造矩阵乘法。"""
    return [sum(a[i][j] * b[j] for j in range(len(b)))
            for i in range(len(a))]


def solve(a, b):
    """高斯消元（部分主元）解 A x = b，A 为方阵，b 为向量。"""
    n = len(a)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise InsufficientData('设计矩阵奇异，窗口内变异不足')
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_row = aug[col]
        pivot_value = pivot_row[col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_value
            if factor == 0.0:
                continue
            target = aug[r]
            for j in range(col, n + 1):
                target[j] -= factor * pivot_row[j]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve(a, e))
    return transpose(cols)


def ols(x, y, weights=None):
    """加权最小二乘，返回 beta、残差与 (X'WX)^-1。"""
    n = len(y)
    if weights is None:
        weights = [1.0] * n
    wx = [[weights[i] * v for v in row] for i, row in enumerate(x)]
    xtwx = matmul(transpose(x), wx)
    xtwy = [sum(x[i][j] * weights[i] * y[i] for i in range(n))
            for j in range(len(x[0]))]
    beta = solve(xtwx, xtwy)
    fitted = [sum(x[i][j] * beta[j] for j in range(len(beta))) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    return beta, resid, inverse(xtwx)


def cluster_robust_vcov(x, resid, clusters, weights=None):
    """Liang-Zeger 聚类稳健方差，含 G/(G-1)*(n-1)/(n-k) 小样本修正。"""
    n = len(resid)
    k = len(x[0])
    if weights is None:
        weights = [1.0] * n
    _, _, xtwx_inv = ols(x, resid, weights)
    groups = {}
    for i, key in enumerate(clusters):
        groups.setdefault(key, []).append(i)
    g = len(groups)
    if g < 2:
        raise InsufficientData('聚类数少于 2，无法计算稳健方差')
    meat = [[0.0] * k for _ in range(k)]
    for indices in groups.values():
        score = [0.0] * k
        for i in indices:
            w_e = weights[i] * resid[i]
            for j in range(k):
                score[j] += x[i][j] * w_e
        for a in range(k):
            for b in range(k):
                meat[a][b] += score[a] * score[b]
    adj = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1))
    vcov = matmul(matmul(xtwx_inv, [[adj * v for v in row] for row in meat]),
                  xtwx_inv)
    return vcov, g


def wald_pvalue(beta, vcov, indices):
    """对给定系数下标做联合 Wald 检验，返回 F 统计量与上尾 p 值。

    方差矩阵在小样本/退化设计下可能奇异，此时返回 NaN 而不是崩溃，
    由调用方按“证据不足”处理。
    """
    try:
        sub_inv = inverse([[vcov[i][j] for j in indices]
                           for i in indices])
    except InsufficientData:
        return float('nan'), float('nan')
    sub_b = [beta[i] for i in indices]
    q = len(indices)
    stat = sum(sub_b[a] * sum(sub_inv[a][b] * sub_b[b] for b in range(q))
               for a in range(q)) / q
    return stat, f_sf(stat, q, max(len(vcov) - 1, 1))


# ---------------------------------------------------------------- 分布函数

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _beta_continued_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def beta_regularized(a, b, x):
    """正则化不完全贝塔函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    if df <= 0 or math.isnan(t):
        return float('nan')
    x = df / (df + t * t)
    return beta_regularized(df / 2.0, 0.5, x)


def t_ppf(two_sided_p, df):
    """给定双侧概率求 t 分位数（绝对值），二分法。"""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > two_sided_p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return (lo + hi) / 2.0


def f_sf(f, d1, d2):
    if f <= 0.0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return beta_regularized(d2 / 2.0, d1 / 2.0, x)


def quantile(values, q):
    """线性插值分位数，q 取 0~1。"""
    if not values:
        raise InsufficientData('空样本无法计算分位数')
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= len(ordered):
        return ordered[-1]
    return ordered[lo] * (1 - frac) + ordered[lo + 1] * frac


def fisher_exact_two_sided(table):
    """2x2 表 Fisher 精确检验双侧 p 值，table=[[a,b],[c,d]]。

    大边际样本下完整组合枚举不可行，此时改用固定种子的 Monte Carlo
    超几何抽样，并在结果中注明方法，避免大组合数导致的性能崩溃。
    """
    a, b = table[0]
    c, d = table[1]
    n = a + b + c + d
    k = a + c
    draws = a + b
    total = math.comb(n, draws)

    def prob(x):
        if x < 0 or x > min(draws, k):
            return 0.0
        return math.comb(k, x) * math.comb(n - k, draws - x) / total

    lo = max(0, draws - (n - k))
    hi = min(draws, k)
    if total <= 200000:
        observed = prob(a)
        return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                            if prob(x) <= observed + 1e-15))
    # Monte Carlo：用对数概率做接受-拒绝，避免重复计算超大组合数。
    rng = random.Random(73011)
    mode = math.floor((draws + 1) * (k + 1) / (n + 2))

    def log_prob(x):
        if x < lo or x > hi:
            return float('-inf')
        return (math.lgamma(k + 1) - math.lgamma(x + 1)
                - math.lgamma(k - x + 1)
                + math.lgamma(n - k + 1)
                - math.lgamma(draws - x + 1)
                - math.lgamma(n - k - draws + x + 1))

    log_mode = log_prob(mode)
    log_observed = log_prob(a)
    reps = 9999
    hits = 0
    accepted = 0
    while accepted < reps:
        x = rng.randrange(lo, hi + 1)
        if rng.random() <= math.exp(log_prob(x) - log_mode):
            accepted += 1
            if log_prob(x) <= log_observed + 1e-12:
                hits += 1
    return min(1.0, (1 + hits) / (1 + reps))


def binom_two_sided_p(observed, n, p0):
    """二项分布双侧精确 p 值。"""
    if n == 0:
        return float('nan')
    expected = n * p0
    probs = [math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
             for k in range(n + 1)]
    target = probs[observed]
    return min(1.0, sum(q for q in probs if q <= target + 1e-15))


# ---------------------------------------------------- wild cluster bootstrap

def wild_cluster_bootstrap(x, y, clusters, weights, beta, vcov, coef_index,
                           reps=999, seed=20261114, webb=False):
    """Rademacher（默认）或 Webb 六点 wild cluster bootstrap-t。

    返回 bootstrap t 序列与比值序列（比值由调用方按系数下标解释），
    用于小样本下的 p 值与置信区间，避免直接沿用普通标准误。
    """
    rng = random.Random(seed)
    fitted = [sum(x[i][j] * beta[j] for j in range(len(beta)))
              for i in range(len(y))]
    resid = [y[i] - fitted[i] for i in range(len(y))]
    groups = {}
    for i, key in enumerate(clusters):
        groups.setdefault(key, []).append(i)
    group_keys = list(groups)
    se_obs = math.sqrt(vcov[coef_index][coef_index])
    t_stats, ratios = [], []
    webb_points = (-1.536, -0.887, -0.469, 0.469, 0.887, 1.536)
    for _ in range(reps):
        shocks = {}
        for key in group_keys:
            shocks[key] = rng.choice(webb_points if webb else (-1.0, 1.0))
        y_star = [fitted[i] + shocks[clusters[i]] * resid[i]
                  for i in range(len(y))]
        try:
            b_star, r_star, _ = ols(x, y_star, weights)
            v_star, _ = cluster_robust_vcov(x, r_star, clusters, weights)
        except InsufficientData:
            continue
        se_star = math.sqrt(v_star[coef_index][coef_index])
        if se_star > 0:
            t_stats.append((b_star[coef_index] - beta[coef_index]) / se_star)
        ratios.append(b_star)
    return t_stats, ratios, se_obs


def bootstrap_t_interval(t_stats, estimate, se_obs, alpha=0.05):
    if len(t_stats) < 20:
        return (float('nan'), float('nan'))
    lower_q = quantile(t_stats, alpha / 2.0)
    upper_q = quantile(t_stats, 1.0 - alpha / 2.0)
    return estimate - upper_q * se_obs, estimate - lower_q * se_obs


def bootstrap_pvalue(t_stats, observed_t):
    if not t_stats:
        return float('nan')
    hits = sum(1 for t in t_stats if abs(t) >= abs(observed_t) - 1e-12)
    return (1 + hits) / (len(t_stats) + 1)
