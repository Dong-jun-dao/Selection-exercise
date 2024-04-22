import numpy as np

def initialization(nP, dim, ub, lb):
    """
    初始化种群。

    参数:
    nP -- 种群数量
    dim -- 维度
    ub -- 上界，可以是一个数字或一个数组
    lb -- 下界，可以是一个数字或一个数组
    """
    # 检测ub和lb是否是单一值还是数组
    if np.isscalar(ub) and np.isscalar(lb):
        # 如果ub和lb都是单一数值，创建整个矩阵
        X = np.random.rand(nP, dim) * (ub - lb) + lb
    else:
        # 如果每个维度有不同的ub和lb，逐个维度初始化
        X = np.zeros((nP, dim))  # 初始化为0的矩阵
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.random.rand(nP) * (ub_i - lb_i) + lb_i

    return X