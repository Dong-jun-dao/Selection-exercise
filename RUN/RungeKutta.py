import numpy as np

def RungeKutta(XB, XW, DelX):
    if XB.ndim >= 2:
        dim = XB.shape[1]
    else:
        dim = len(XB)
    # 获取维度
    C = np.random.randint(1, 3) * (1 - np.random.rand())
    r1 = np.random.rand( dim)
    r2 = np.random.rand( dim)

    K1 = 0.5 * (np.random.rand() * XW - C * XB)
    K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2) - (C * XB + r1 * K1 * DelX / 2))
    K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2) - (C * XB + r1 * K2 * DelX / 2))
    K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

    XRK = K1 + 2 * K2 + 2 * K3 + K4
    SM = 1 / 6 * XRK
    return SM
