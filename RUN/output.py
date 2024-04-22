from BenchmarkFunctions import BenchmarkFunctions
from RUN import RUN
import matplotlib.pyplot as plt
# 设置参数
nP = 50  # 种群数量
Func_name = 'F1'  # 测试函数名称，范围从 F1 到 F14
MaxIt = 500  # 最大迭代次数

# 载入选择的基准函数细节
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)

# 执行优化算法
Best_fitness, BestPositions, Convergence_curve = RUN(nP, MaxIt, lb, ub, dim, fobj)

# 绘制收敛曲线
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.axis('tight')
plt.grid(False)
plt.box(True)
plt.legend(['RUN'])
plt.show()
