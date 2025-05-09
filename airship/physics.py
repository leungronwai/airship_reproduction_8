# 新建文件：airship/physics.py
"""
物理模型计算模块 - 包含质量、惯性和气动特性的计算
参考:Development of an Aerodynamic Model and Control  Law Design for a High Altitude Airship

"""

# pylint: disable=invalid-name
# cspell:ignore ndarray
import numpy as np




# === 计算附加质量/惯性 (Calculate Added Mass/Inertia) ===
def calculate_added_mass_inertia(a1, a2, b, rho_air_):
    """
    根据双椭球体模型的几何参数计算附加质量和附加惯性矩阵。
    Calculates the added mass and added inertia matrices based on the
    geometric parameters of a double-ellipsoid model.

    参考公式来源：图片中提供的 Eq. 42 - 51
    Reference Equations: Eq. 42 - 51 from the provided image.

    Args:
        a1 (float): 第一个半长轴 (Semi-major axis 1).
        a2 (float): 第二个半长轴 (Semi-major axis 2).
        b (float): 半短轴 (Semi-minor axis).
        rho_air_ (float): 当地空气密度 (Local air density).

    Returns:
        tuple:包含两个 NumPy 数组的元组 (M_prime, I0_prime)
               A tuple containing two NumPy arrays: (M_prime, I0_prime)
               M_prime (np.ndarray): 附加质量矩阵 (Added mass matrix, 3x3).
               I0_prime (np.ndarray): 附加惯性矩阵 (Added inertia matrix, 3x3).
               k1 (float): 附加质量因子 1 (Added mass factor 1).
               k2 (float): 附加质量因子 2 (Added mass factor 2).
               k3 (float): 附加质量因子 3 (Added mass factor 3).

    Raises:
        ValueError: 如果几何参数无效 (If geometric parameters are invalid).
    """

    if b <= 0:
        raise ValueError("半短轴 b 必须大于 0 /Semi-minor axis b must be positive")

    # 计算平均半长轴 (Calculate mean semi-major axis 'a')
    a = (a1 + a2) / 2.0
    if a <= 0:
        raise ValueError("平均半长轴 a 必须大于 0 / Mean semi-major axis a must be positive")

    # 检查是否为长椭球 (Check for prolate spheroid assumption a >= b)
    # 注意：如果 b > a (扁椭球)，偏心率 e 和相关公式定义不同
    # Note: If b > a (oblate), eccentricity and related formulas differ.
    # 这里假设 a >= b，与图片中公式一致
    # Assuming a >= b as consistent with the provided formulas.
    if b > a:
        print(
            f"Warning: The current formula is applicable for prolate spheroids (a >= b), "
            f"but the input is a={a:.3f}, b={b:.3f} (oblate spheroid). "
            f"The result may be inaccurate."
        )
        # print(f"警告：当前公式适用于长椭球 (a >= b)，但输入为 a={a:.3f}, b={b:.3f} (扁椭球)。"
        #   f"结果可能不准确。")
        # 对于扁椭球需要不同的公式或检查源文献
        # Different formulas or source check needed for oblate case.
        # 为避免错误，可以抛出异常或继续计算（结果可能错误） /  Continue calculation (result may be incorrect)
        # raise ValueError("当前公式仅适用于 a >= b 的情况" / Current formula only for a >= b case")

    # 计算体积 (Calculate Volume V - Eq. 43)
    V = (2.0 / 3.0) * np.pi * (a1 + a2) * b**2
    # V = (4.0 / 3.0) * np.pi * a * b**2  # 使用平均值 a 的等效公式 Use equivalent formula with mean value a

    # 计算排开空气的质量 (Calculate mass of displaced air)
    m_air = rho_air_ * V

    # 处理特殊情况：球体 (Handle special case: Sphere)
    tolerance = 1e-9  # 定义一个小的容差 / Define a small tolerance
    if abs(a - b) < tolerance:
        # 对于球体 (For a sphere, a = b, e = 0)
        # k1_ = k2_ = k3_ = 0.5 (标准流体动力学结果 standard hydrodynamic result)
        k1_ = 0.5
        k2_ = 0.5
        k3_ = 0.5
    else:
        # 计算偏心率 (Calculate eccentricity e - Eq. 44)
        # 确保 ensure  a^2 > 0 且 and 1 - (b^2 / a^2) >= 0
        a_sq = a**2
        term_inside_sqrt = 1.0 - (b**2 / a_sq)
        if (1.0 - (b**2 / (a**2))) < 0:
            # 这理论上不应该在 a >= b 时发生，除非有数值误差
            print(f"警告：偏心率计算出现问题 / Eccentricity calculation warning " f"(term = {term_inside_sqrt:.2e})。将 e 设为 0 / set e as 0。")
            _e = 0.0
            k1_ = k2_ = k3_ = 0.5  # 退化为球体情况 / Fallback to sphere case
        else:
            _e = np.sqrt(1.0 - (b**2 / (a**2)))  #!! eq. 44

            # 避免 e 极其接近 1 (避免 f 中的除零) / Avoid e close to 1 (to avoid division by zero in f)
            if abs(1.0 - _e) < tolerance:
                raise ValueError("偏心率 e 接近 1 (b 接近 0)，几何形状无效。/ Eccentricity e approaches 1 (invalid geometry)")

            # 计算中间参数 / Calculate intermediate parameters f, g, alpha_prime, beta_prime
            # Calculate intermediate parameters f, g, alpha_prime, beta_prime

            # f (Eq. 45)
            f = np.log((1.0 + _e) / (1.0 - _e))  #!! eq. 45

            # g (Eq. 46)
            # 避免 e=0 (已在球体情况中处理) / Avoid division by zero for e=0 (handled in sphere case)
            e_sq = _e**2
            e_cubed = _e**3
            if abs(e_cubed) < tolerance:
                # 理论上 e 非零，但数值上可能很小 / Theoretically e is non-zero, but numerically small
                raise ValueError("偏心率 e 的立方接近于零，无法计算 g。 / Eccentricity e cubed is close to zero, cannot calculate g.")
            _g = (1.0 - e_sq) / e_cubed  #!! eq. 46

            # alpha_ (Eq. 47)
            alpha_ = 2.0 * _g * (f / 2.0 - _e)  #!! eq. 47

            # beta_ (Eq. 48)
            if abs(e_sq) < tolerance:
                raise ValueError(
                    "偏心率 e 的平方接近于零，无法计算 beta_prime。" "The square of eccentricity e is close to zero, beta_prime cannot be calculated."
                )
            beta_ = (1.0 / e_sq) - (_g * f / 2.0)  #!! eq. 48

            # 计算惯性因子 k1, k2, k3 / Calculate inertia factors k1, k2, k3
            # k1 (Eq. 49)
            denominator_k1 = 2.0 - alpha_  # denominator 分母   numer 分子，，fraction 分数
            if abs(denominator_k1) < tolerance:
                raise ValueError("计算 k1 时分母接近零。/ Small denominator in k1 calculation")
            k1_ = -alpha_ / (2.0 - alpha_)  #!! eq. 49

            # k2 (Eq. 50)
            denominator_k2 = 2.0 - beta_
            if abs(denominator_k2) < tolerance:
                raise ValueError("计算 k2 时分母接近零。/ Small denominator in k2 calculation")
            k2_ = -beta_ / (2.0 - beta_)  #!! eq. 50

            # k3 (Eq. 51)
            a_sq = a**2
            b_sq = b**2
            term1_num_k3 = (b_sq - a_sq) * (alpha_ - beta_)
            term2_den_k3 = 2.0 * (b_sq - a_sq) + (b_sq + a_sq) * (beta_ - alpha_)

            if abs(term2_den_k3) < tolerance:
                # 检查球体情况是否已处理 (e=0 -> a=b -> b^2-a^2 = 0)
                # / Check if sphere case has been handled (e=0 -> a=b -> b^2-a^2 = 0)
                # 如果 a != b 但分母为零，表示可能有其他问题或特殊共振情况
                # / If a != b but denominator is zero, there may be another issue or resonance case.
                if abs(a - b) > tolerance:
                    raise ValueError("计算 k3 时分母接近零 (非球体情况) Small denominator in k3 calculation。")
                else:  # 如果是球体，分子也为零，极限应为 0.5 / If it's a sphere, numerator is also zero, limit should be 0.5
                    k3_ = 0.5
            else:
                k3_ = -(1.0 / 5.0) * term1_num_k3 / term2_den_k3  #!! eq. 51

    # 构建附加质量矩阵 (Construct Added Mass Matrix M_prime - Eq. 42)
    _M_prime = m_air * np.diag([k1_, k2_, k2_])  #!! eq. 42

    # 构建附加惯性矩阵 (Construct Added Inertia Matrix I0' - Eq. 42)
    _I0_prime = m_air * np.diag([0.0, k3_, k3_])  #!! eq. 42

    return _M_prime, _I0_prime, k1_, k2_, k3_


