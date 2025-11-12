import numpy as np
from scipy import linalg
from scipy.optimize import minimize

# ---------------------------
# 函数定义（与你给出的相同）
# ---------------------------

def compute_A_phi(G, Z, W, lambda_):
    """构造 A_phi = Z^T W G_phi，其中 G_phi = 4π/λ * G"""
    G_phi = (4.0 * np.pi / lambda_) * G            # M x N_tot
    A_phi = Z.T @ (W @ G_phi)                     # p x N_tot
    return A_phi

def build_selection_matrix(N_tot, indices):
    """构造选择矩阵 S (N_tot x K) 将 K 个受控促动器映射到总的 N_tot 促动器"""
    K = len(indices)
    S = np.zeros((N_tot, K), dtype=float)
    for k, idx in enumerate(indices):
        S[idx, k] = 1.0
    return S

def tikhonov_control(A_phi, indices, c_des, alpha=1e-6, L=None):
    N_tot = A_phi.shape[1]
    S = build_selection_matrix(N_tot, indices)   # N_tot x K
    A_sub = A_phi @ S                            # p x K
    K = A_sub.shape[1]
    if L is None:
        L = np.eye(K)
    # 正则化矩阵
    ATA = A_sub.T @ A_sub
    reg = alpha * (L.T @ L)
    Q = ATA + reg                                # K x K
    rhs = A_sub.T @ c_des                        # K
    # 解析解
    u = linalg.solve(Q, rhs, assume_a='pos')     # K
    return u, S, A_sub

def qp_control_with_bounds(A_phi, indices, c_des, alpha=1e-6, L=None,
                           fmin_mN=-17000.0, fmax_mN=17000.0):
    """
    返回 u_opt（单位 N）和总力 f_total (N_tot vector)
    fmin_mN, fmax_mN: 单位 mN（传入后转换为 N）
    """
    N_tot = A_phi.shape[1]
    S = build_selection_matrix(N_tot, indices)
    A_sub = A_phi @ S                           # p x K
    K = A_sub.shape[1]
    if L is None:
        L = np.eye(K)

    # 转换边界到 N
    lb = (fmin_mN * 1e-3) * np.ones(K)
    ub = (fmax_mN * 1e-3) * np.ones(K)
    bounds = [(lb[i], ub[i]) for i in range(K)]

    def obj(u):
        r = A_sub @ u - c_des
        return float(r.T @ r + alpha * (L @ u).T @ (L @ u))

    def jac(u):
        r = A_sub @ u - c_des
        g = 2.0 * (A_sub.T @ r) + 2.0 * alpha * (L.T @ (L @ u))
        return g

    # 初始猜测
    try:
        u0, _, _ = tikhonov_control(A_phi, indices, c_des, alpha, L)
    except Exception:
        u0 = np.zeros(K)

    res = minimize(obj, u0, jac=jac, bounds=bounds, method='L-BFGS-B',
                   options={'ftol':1e-12, 'maxiter':1000})
    if not res.success:
        print("Warning: optimizer did not converge:", res.message)
    u_opt = res.x
    f_total = S @ u_opt                       # N_tot (单位 N)
    return u_opt, f_total, S, A_sub

# ---------------------------
# 示例输入参数（可直接运行）
# ---------------------------

np.random.seed(42)
M = 200         # 传感器测量点数（波前采样）
N_tot = 51      # 总促动器数
p = 15          # Zernike 模式数量
lambda_ = 633e-9  # 波长 633 nm

# 模拟响应矩阵 G (m/N)，假设 51 个促动器对 200 个点
G = np.random.randn(M, N_tot) * 1e-9

# 模拟 Zernike 基（M × p）
Z = np.random.randn(M, p)

# 权重矩阵（假设均匀）
A = 1.0
W = (A / M) * np.eye(M)

# 目标Zernike系数（单位 rad）
c_des = np.random.randn(p) * 1e-6

# 选择控制的20个促动器
indices = np.sort(np.random.choice(N_tot, 20, replace=False))

# 正则化参数
alpha = 1e-6
L = np.eye(len(indices))

# ---------------------------
# 计算控制命令
# ---------------------------
A_phi = compute_A_phi(G, Z, W, lambda_)

# 求解
u_opt, f_total, S, A_sub = qp_control_with_bounds(
    A_phi, indices, c_des,
    alpha=alpha,
    L=L,
    fmin_mN=-17000,
    fmax_mN=17000
)

# ---------------------------
# 输出结果
# ---------------------------
f_total_mN = f_total * 1e3    # 转为 mN

print("被控促动器索引（共 20 个）:")
print(indices)
print("\n对应的控制电压/力 (mN):")
print(f_total_mN[indices])

# 若想绘制电压分布
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.bar(range(N_tot), f_total_mN)
plt.title("51 个促动器输出 (mN)")
plt.xlabel("Actuator Index")
plt.ylabel("Force / Voltage (mN)")
plt.axhline(17000, color='r', linestyle='--', label='Upper bound')
plt.axhline(-17000, color='r', linestyle='--', label='Lower bound')
plt.legend()
plt.show()
