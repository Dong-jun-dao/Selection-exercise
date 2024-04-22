import numpy as np

def BenchmarkFunctions(F):
    D = 30  # 所有函数共享的维度数

    # 使用字典来代替MATLAB中的switch-case语句
    functions = {
        'F1': (F1, -100, 100, D),
        'F2': (F2, -100, 100, D),
        'F3': (F3, -100, 100, D),
        'F4': (F4, -100, 100, D),
        'F5': (F5, -100, 100, D),
        'F6': (F6, -100, 100, D),
        'F7': (F7, -100, 100, D),
        'F8': (F8, -100, 100, D),
        'F9': (F9, -100, 100, D),
        'F10': (F10, -32.768, 32.768, D),
        'F11': (F11, -100, 100, D),
        'F12': (F12, -100, 100, D),
        'F13': (F13, -600, 600, D),
        'F14': (F14, -50, 50, D)
    }

    fobj, lb, ub, dim = functions.get(F)
    return lb, ub, dim, fobj

def F1(x):
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

# Power
def F2(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 获取输入数组x的维度
    f = np.zeros(D)  # 初始化一个与x维度相同的数组，用于存储中间计算结果。
    for i in range(D):  # 遍历x中的每一个元素。
        f[i] = np.abs(x[i]) ** (i + 2)  # 对每个元素取绝对值，然后将其乘方，指数为元素的索引+2(因为matlab从1到D，Python是从0到D-1)。
    return np.sum(f)  # 将所有处理过的元素求和，得到函数的最终值。

# Zakharov
def F3(x):
    return np.sum(x**2) + np.sum(0.5 * x)**2 + np.sum(0.5 * x)**4

# Rosenbrock
def F4(x):

    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)
    ff = np.zeros(D - 1)  # 初始化结果数组
    for i in range(D - 1):  # Python 中的索引从 0 开始，因此需要调整
        ff[i] = 100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
    return np.sum(ff)

# Discus
def F5(x):
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)

# High Conditioned Elliptic
def F6(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)
    indices = np.arange(1, D+1)  # 生成1到D的索引数组
    # 计算每个元素的权重
    powers = (10**6)**((indices - 1) / (D - 1))
    return np.sum(powers * x**2)

# 多峰值函数
# np.expanded Schaffer’s F6
def F7(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x) # 获取输入向量 x 的维度或长度
    f = np.zeros(D)  # 初始化结果数组
    for i in range(D):
        if i == D - 1:  # 对最后一个元素特殊处理，与第一个元素相结合形成闭环
            term = x[i] ** 2 + x[0] ** 2
        else:  # 常规情况下，当前元素与下一个元素结合
            term = x[i] ** 2 + x[i + 1] ** 2
        f[i] = 0.5 + (np.sin(np.sqrt(term)) ** 2 - 0.5) / (1 + 0.001 * term) ** 2
    return np.sum(f)

# Levy函数
def F8(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)
    w = 1 + (x - 1) / 4
    f = (w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)
    z = np.sin(np.pi * w[0])**2 + np.sum(f) + (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return z

# np.modified Schwefel‘s
import numpy as np


def F9(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 获取输入向量 x 的维度
    f = np.zeros(D)  # 初始化存储每个维度计算结果的数组
    for i in range(D):
        y = x[i] + 420.9687462275036  # 对每个元素偏移一个常数
        # 应用相同的逻辑条件分支
        if abs(y) < 500:
            f[i] = y * np.sin(np.sqrt(abs(y)))
        elif y > 500:
            f[i] = (500 - y % 500) * np.sin(np.sqrt(abs(500 - y % 500))) - (y - 500) ** 2 / (10000 * D)
        elif y < -500:
            f[i] = (y % 500 - 500) * np.sin(np.sqrt(abs(y % 500 - 500))) - (y + 500) ** 2 / (10000 * D)

    z = 418.9829 * D - np.sum(f)  # 计算最终的函数值
    return z


# Ackley
def F10(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 获取输入向量 x 的维度
    z = -20 * np.exp(-0.2 * np.sqrt((1/D) * np.sum(x**2))) - np.exp((1/D) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
    return z

# weierstrass
def F11(x):
  if x.ndim >= 2:
    D = x.shape[1]
  else:
    D = len(x)  # 获取输入向量 x 的维度
  x = x + 0.5
  a = 0.5
  b = 3
  kmax = 20
  c1 = a ** (np.arange(kmax + 1))
  c2 = 2 * np.pi * b ** (np.arange(kmax + 1))
  f = 0
  c = -w(c1, c2, 0.5)
  for i in range(D):
      f += w(c1, c2, x[:][i])
  z = f + c * D
  return z

def w(c1, c2, x):
  x_arr = np.asarray(x)
  if (x_arr.ndim > 1):
    y = np.zeros((x_arr.ndim, 1))
    for k in range(x_arr.ndim):
      y[k] = sum(c1 * np.cos(c2 * x_arr[k]))
  else:
    y = sum(c1 * np.cos(c2 * x))
  return y


# HappyCat
def F12(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 获取输入向量 x 的维度
    z = (np.abs(np.sum(x**2) - D)**0.25) + ((0.5 * np.sum(x**2) + np.sum(x)) / D) + 0.5
    return z


def F13(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 获取输入向量 x 的维度
    indices = np.arange(1, D + 1)
    z = np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(indices))) + 1
    return z


def F14(x):
    if x.ndim >= 2:
        D = x.shape[1]
    else:
        D = len(x)  # 对于一维数组，其长度就是其“维度”

    term1 = 10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2
    if D > 1:
        term2 = np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2))
    else:
        term2 = 0  # 当 x 只有一个元素时，没有第二项
    term3 = ((x[-1] + 1) / 4) ** 2
    ufun_result = Ufun(x, 10, 100, 4)
    z = (np.pi / D) * (term1 + term2 + term3) + np.sum(ufun_result)
    return z

def Ufun(x, a, k, m):
    # 直接使用 numpy 来处理向量化的条件运算
    return k * (np.power(x - a, m) * (x > a) + np.power(-x - a, m) * (x < -a))

