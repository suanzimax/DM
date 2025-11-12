import numpy as np
from scipy import linalg
from scipy.optimize import minimize

# ==========================
# 参数设置
# ==========================
np.random.seed(42)

M = 200          # 波前采样点数量（传感器测点）
N_tot = 51       # 总促动器数量
p = 15           # Zernike 模式数量
lambda_ = 633e-9 # 波长（m）
alpha = 1e-6     # 正则化系数
fmin_mN, fmax_mN = -17000.0, 17000.0

# ==========================
# 模拟系统矩阵（真实系统中应由标定获得）
# ==========================
# G: M x N_tot  促动力-波前响应矩阵
G = np.random.randn(M, N_tot) * 1e-9  # 单位 m/N，假设微小变形
# Z: M x p  Zernike 模式矩阵（正交基）
Z = np.random.randn(M, p)
# W: 权重矩阵（假设均匀采样）
W = (1.0 / M) * np.eye(M)

# ==========================
# 构造 A_phi = Z^T W G_phi
# ==========================
def compute_A_phi(G, Z, W, lambda_):
    G_phi = (4.0 * np.pi / lambda_) * G
    A_phi = Z.T @ (W @ G_phi)
    return A_phi

A_phi = compute_A_phi(G, Z, W, lambda_)

# ==========================
# Zernike 目标系数（目标波前相位）
# ==========================
# 模拟目标波前形状（随机）
c_des = np.random.randn(p) * 0.5  # 单位 rad（Zernike 系数目标）

# ==========================
# 选择内圈促动器（0~19）
# ==========================
indices = np.arange(0, 20)

def build_selection_matrix(N_tot, indices):
    S = np.zeros((N_tot, len(indices)))
    for k, idx in enumerate(indices):
        S[idx, k] = 1.0
    return S

S = build_selection_matrix(N_tot, indices)
A_sub = A_phi @ S
K = A_sub.shape[1]
L = np.eye(K)

# ==========================
# 优化目标函数
# ==========================
def obj(u):
    r = A_sub @ u - c_des
    return float(r.T @ r + alpha * (L @ u).T @ (L @ u))

def jac(u):
    r = A_sub @ u - c_des
    g = 2.0 * (A_sub.T @ r) + 2.0 * alpha * (L.T @ (L @ u))
    return g

# 边界约束 (mN 转 N)
lb = (fmin_mN * 1e-3) * np.ones(K)
ub = (fmax_mN * 1e-3) * np.ones(K)
bounds = [(lb[i], ub[i]) for i in range(K)]

# 初值（全零）
u0 = np.zeros(K)

# ==========================
# 优化求解
# ==========================
res = minimize(obj, u0, jac=jac, bounds=bounds, method='L-BFGS-B',
               options={'ftol':1e-12, 'maxiter':1000})

u_opt = res.x  # N 单位
f_total_N = S @ u_opt
f_total_mN = f_total_N * 1e3  # 转为 mN

# ==========================
# 输出结果
# ==========================
print("被控促动器索引（共 {} 个）:".format(len(indices)))
print(indices)
print("\n对应的控制电压/力 (mN):")
print(np.round(f_total_mN[indices], 6))
