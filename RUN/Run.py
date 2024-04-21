import numpy as np
from initialization import initialization
from RungeKutta import RungeKutta

def RUN(nP, MaxIt, lb, ub, dim, fobj):
    Cost = np.zeros((nP, 1))  # 记录所有解的适应度值
    X = initialization(nP, dim, ub, lb)  # 初始化随机解集
    Xnew2 = np.zeros((1, dim))  # 创建一个用于存储新解的临时数组
    Convergence_curve = np.zeros(MaxIt)  # 初始化收敛曲线数组
    for i in range(nP):
        Cost[i] = fobj(X[i, :])  # 计算目标函数的值并存储
    Best_Cost = np.min(Cost)  # 确定最佳解
    Best_X = X[np.argmin(Cost), :]
    Convergence_curve[0] = Best_Cost  # 记录第一次迭代的最佳成本

    # 主循环
    it = 1
    while it < MaxIt:
        it += 1
        f = 20 * np.exp(-(12 * (it / MaxIt)))  # 自适应因子计算
        Xavg = np.mean(X, axis=0)  # 解集的平均值
        SF = 2 * (0.5 - np.random.rand(nP)) * f  # 自适应因子

        for i in range(nP):
            ind_l = np.argmin(Cost)  # 获取成本最小的索引
            lBest = X[ind_l, :]  # 全局最佳解

            A, B, C = RndX(nP, i)  # 获取三个随机索引
            ind1 = np.argmin(Cost[[A, B, C]])

            # Delta X 计算
            gama = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * np.abs(Stp)

            # 为 Runge Kutta 方法确定 Xb 和 Xw
            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]

            SM = RungeKutta(Xb, Xw, DelX)  # 基于 Runge Kutta 方法的搜索机制

            L = np.random.rand(dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]  # 混合解
            Xm = L * Best_X + (1 - L) * lBest  # 混合最优解

            vec = np.array([1, -1])
            flag = np.floor(2 * np.random.rand(dim) + 1).astype(int)
            r = vec[flag - 1]  # 使用 vec 数组，通过 flag 作为索引来选择方向

            g = 2 * np.random.rand()  # 生成一个0到2之间的随机数
            mu = 0.5 + 0.1 * np.random.randn(dim)  # 生成一个dim维的正态分布随机数数组，均值0.5，标准差0.1

            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * SM + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * SM + mu * (X[A] - X[B])

            FU = Xnew > ub
            FL = Xnew < lb
            Xnew = Xnew * ~(FU | FL) + ub * FU + lb * FL  # 对越界解进行修正

            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i] = Xnew
                Cost[i] = CostNew

            # 如果随机数小于0.5，则执行以下操作
            if np.random.rand() < 0.5:
                # 计算EXP，这是一个随迭代次数减少的指数衰减因子
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                # r是一个从-1到2之间的随机整数
                r = np.floor(np.random.uniform(-1, 2))
                # u是一个维度为1*dim的从0到2的均匀分布的随机数
                u = 2 * np.random.rand(dim)
                # w是一个随机数，乘以之前计算的EXP
                w = np.random.uniform(0, 2, dim) * EXP# (Eq.19-1)

                # 随机选择三个不同的索引
                A, B, C = RndX(nP, i)
                # 计算这三个解的平均值
                Xavg = (X[A] + X[B] + X[C]) / 3  # (Eq.19-2)

                # beta是一个随机数向量
                beta = np.random.rand(dim)
                # 计算新的解Xnew1
                Xnew1 = beta * Best_X + (1 - beta) * Xavg  # (Eq.19-3)

                # 遍历所有维度来调整Xnew2
                Xnew2 = np.zeros(dim)
                for j in range(dim):
                    if w[j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[j] * abs(Xnew1[j] - Xavg[j] + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * abs(u[j] * Xnew1[j] - Xavg[j] + np.random.randn())

                # 检查并修正超出边界的解
                FU = Xnew2 > ub
                FL = Xnew2 < lb
                Xnew2 = Xnew2 * ~(FU | FL) + ub * FU + lb * FL
                CostNew = fobj(Xnew2)

                # 如果新解的成本低于当前解的成本，则接受新解
                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else:
                    if np.random.rand() < w[np.random.randint(dim)]:
                        SM = RungeKutta(X[i, :], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM + (2 * np.random.rand(dim) * Best_X - Xnew2))
                        # 再次检查并修正超出边界的解
                        FU = Xnew > ub
                        FL = Xnew < lb
                        Xnew = Xnew * ~(FU | FL) + ub * FU + lb * FL
                        CostNew = fobj(Xnew)
                        # 如果这一次的新解成本更低，接受这一次的新解
                        if CostNew < Cost[i]:
                            X[i, :] = Xnew
                            Cost[i] = CostNew
            # 结束ESQ部分

            # 判断当前解的成本是否小于记录的最佳成本
            if Cost[i] < Best_Cost:  # 对应 MATLAB 中的 if Cost(i) < Best_Cost
                Best_X = X[i, :]  # 更新最佳位置
                Best_Cost = Cost[i]  # 更新最佳成本

        # 保存每次迭代的最佳解
        Convergence_curve[it - 1] = Best_Cost  # 在 Python 中，索引从 0 开始，所以使用 it-1
        print(f'Iteration {it}: Best Cost = {Convergence_curve[it - 1]}')

    return Best_Cost, Best_X, Convergence_curve
def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z

# A function to determine thress random indices of solutions
def RndX(nP, i):
    Qi = np.random.permutation(nP)  # numpy 的 permutation 从 0 开始，因此 +1 调整为从 1 开始
    Qi = Qi[Qi != i]
    A, B, C = Qi[0], Qi[1], Qi[2]
    return A, B, C