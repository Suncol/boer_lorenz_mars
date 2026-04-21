# Mars Exact LEC Integration Plan

## 目的

本文件用于把“在当前 SEBA 代码库中引入火星 exact / topography-aware Lorenz energy cycle 诊断”的方案落盘，作为后续实现、评审和测试的统一对照基线。

当前仓库本质上是 `src/seba` 为核心的谱能量预算工具，因此本方案的基本原则不是把 SEBA 硬改成 Boer (1989) 的四储库预算器，而是：

1. 保留 `src/seba` 作为谱诊断后端。
2. 在当前仓库中新增一套独立的 Mars exact energetics 包。
3. 共享同一套火星预处理和公共数学层。
4. 将 Boer-exact 四储库预算与 SEBA 谱能量诊断并行运行，而不是互相替代。

一句话概括：

```text
同一套火星预处理核心
  -> 支路 A: Boer-exact / topography-aware 四储库预算
  -> 支路 B: SEBA 谱能量诊断
```

---

## 核心结论

### 1. 不建议把 SEBA 直接改造成 Boer 1989 四储库预算器

原因如下：

- SEBA 的理论框架是 Augier & Lindborg 的谱能量预算，不是 Tabataba-Vakili et al. (2015) 使用的 Boer-exact `(A_Z, A_E, K_Z, K_E)` 框架。
- 火星地形起伏强，TV2015 明确依赖 Boer (1989) exact formulation，并显式保留地形项 `(AE_2, AZ_2, CK_2, CZ_2)`。
- SEBA 的 APE、mean/eddy 分解、输出维度和 Boer exact 体系不一致。
- 因此最稳妥的路线是“并行双支路”，而不是“单支路硬兼容”。

### 2. 当前仓库的推荐落地方式

建议保留现有 `src/seba` 目录基本不动，在 `src/` 下新增独立包：

```text
src/
  seba/
  mars_exact_lec/
    __init__.py
    config.py
    constants_mars.py
    io/
      __init__.py
      load_dataset.py
      remap_to_pressure.py
      mask_below_ground.py
    common/
      __init__.py
      grid_weights.py
      zonal_ops.py
      thermo.py
      geopotential.py
      gradients.py
      time_derivatives.py
      integrals.py
      conventions.py
    reference_state/
      __init__.py
      koehler_solver.py
      interpolate_isentropes.py
    boer/
      __init__.py
      reservoirs.py
      conversions.py
      sources_sinks.py
      closure.py
      normalize.py
    spectral/
      __init__.py
      seba_constants_mars.py
      seba_adapter.py
      spectra_validation.py
tests/
  test_mask.py
  test_flat_surface_limit.py
  test_ke_partition.py
  test_reference_state.py
  test_budget_closure.py
```

这样可以最大限度减少对现有 SEBA 主体代码的破坏，同时让 Boer exact 物理逻辑保持独立和可测试。

---

## 需要钉死的 5 个认知边界

### 1. SEBA 的 `representative_mean` 不是 Boer 的 `[X]_R`

- SEBA 中的 `representative_mean` 指 pressure surface 上、地表以上部分的全球加权平均。
- Boer / TV2015 中的 `[X]_R` 是 representative zonal mean，本质上是沿经度的代表性纬圈平均：

```text
[X]_R = [Theta X] / [Theta]
```

- 两者不是同一个算子，不能混用。

### 2. SEBA 的 APE 不是 Boer exact APE

- SEBA 的 APE 基于 Lorenz parameter `gamma` 与温度扰动谱：

```text
APE_seba ~ gamma * spectrum(theta_prime) / 2
```

- Boer exact APE 需要参考态 `pi(theta, t)`、效率因子 `N(pi)`，并包含显式地形 surface terms。
- 不能把 SEBA 的 APE 谱积分后当成 `(A_Z + A_E)`。

### 3. 不能从 SEBA 标准输出反推出 `(K_Z, K_E)`

- SEBA 标准输出是按球谐总次数 `l` 的 degree spectrum。
- 阶数 `m` 在常规输出中已被累加，因此 `m = 0` 的 zonal-mean 信息不再可恢复。
- SEBA 分的是 rotational / divergent KE，而不是 zonal / eddy KE。
- 所以不能从 `hke/rke/dke` 直接构造 `(K_Z, K_E)`。

### 4. SEBA “考虑地形”不等于 Boer 地形项已经完成

- SEBA 的 terrain mask 与 masked-field correction 解决的是谱分析中的地表截断问题。
- 这不能替代 Boer exact 中的 `(AE_2, AZ_2, CZ_2, CK_2)`。
- 对火星来说，这些项不是小修正，而是核心物理组成。

### 5. Stock SEBA 不能直接拿来跑火星

- 当前 `src/seba/constants.py` 是 Earth 常数层。
- 单独改 `rsphere` 不够，至少还要统一替换：

```text
radius, Omega, g, Rd, cp, kappa, p0
```

- 尤其 `chi = Rd / cp` 这类派生量不能遗漏。
- 不能依赖零碎 monkey patch；应做 Mars constants backend 或明确的适配层。

---

## 当前仓库中的总体架构

### 总体原则

- `src/seba` 继续作为谱分析和球谐工具库存在。
- `src/mars_exact_lec` 负责火星 exact energetics 主逻辑。
- 两个分支共享同一套数据读入、压强层插值、below-ground masking 和常数定义。
- 任何 Boer-exact 独有的理论对象都不写进 `src/seba` 的核心逻辑里。

### 数据流

```text
原始 Mars GCM / reanalysis 数据
  -> 统一读入与时间对齐
  -> remap 到 pressure levels
  -> 依据 ps 构造 Theta = H(ps - p)
  -> mask below-ground cells
  -> 公共数学层
     -> Boer 支路: reservoirs / conversions / closure
     -> SEBA 支路: hke / rke / dke / ape / flux diagnostics
```

---

## 哪些部分可以复用 SEBA，哪些必须自写

### 可复用或半复用的部分

#### 1. 球谐变换和网格权重

可以复用 `src/seba/spherical_harmonics.py` 的球谐变换能力，以及与高斯网格相关的面积权重逻辑，用于：

- 可选谱能量诊断。
- 高斯网格 area weighting。
- 某些水平梯度、散度和 Helmholtz 分解后端。

#### 2. 梯度与垂直差分工具

可参考或复用 `src/seba` 中已有的梯度实现思路，尤其是：

- 水平梯度。
- 压强坐标下的一维垂直导数。
- masked regions 下避免误差传播的差分策略。

#### 3. 数据预处理思路

SEBA 对外部 `ps`、isobaric levels、terrain mask 的 I/O 组织方式可以直接借鉴，但不应把它等同于 Boer exact 地形项实现。

#### 4. 作为附加诊断的谱分支

SEBA 最适合作为以下附加诊断：

- `hke/rke/dke/ape` 的谱分布。
- `pi_hke`、`pi_ape` 等尺度传递和累计通量。
- 对总水平 KE 的交叉校验。

### 必须自写的部分

#### 1. 火星常数层

`constants_mars.py` 必须自写，至少包含：

```text
a, g, Omega, Rd, cp, kappa, p00
```

#### 2. Boer 的 representative zonal mean 与 eddy split

`common/zonal_ops.py` 必须自写，因为 SEBA 的 global representative mean 不是 Boer 的 `[X]_R`。

#### 3. Koehler 1986 参考态求解器

`reference_state/koehler_solver.py` 必须自写。这是 Boer exact APE 的核心，不存在现成的 SEBA 对应实现。

#### 4. 四储库、转换项、源汇项

以下模块必须自写：

- `boer/reservoirs.py`
- `boer/conversions.py`
- `boer/sources_sinks.py`
- `boer/closure.py`
- `common/conventions.py`

#### 5. 符号和正负号规范

- `CK` 在 TV2015 中定义为 `K_Z -> K_E`。
- `F_Z` 的实现建议在代码内部用“正耗散率”记法，避免与预算公式混淆。
- 所有正负号约定必须集中写入 `conventions.py`，不能散落在各公式实现中。

---

## 公式与模块映射

下面给出建议的实现映射。公式采用对实现友好的记号，而不是追求排版层面的原样复现。

### 1. `common/grid_weights.py` 与 `common/integrals.py`

先固定面积元和质量元：

```text
d_sigma = a^2 cos(phi) d_lambda d_phi
dm = d_sigma dp / g
```

建议提供两个积分器，避免“对已经做过经向平均的量再次平均”的错误：

```text
I_M_full(X) = sum_{i,j,k} X_ijk * Delta_sigma_ij * Delta_p_k / g

I_M_zm(Y) = sum_{j,k} Y_jk * (2 pi a^2 cos(phi_j) Delta_phi_j) * Delta_p_k / g
```

这里：

- `full` 用于三维全场积分。
- `zm` 用于已经做过 zonal averaging 的场。

### 2. `common/zonal_ops.py`

定义地形指示函数：

```text
Theta(lambda, phi, p, t) = 1  if p <= ps
                           0  if p > ps
```

定义代表性纬圈平均与扰动：

```text
[X]_R = [Theta X] / [Theta]   if [Theta] != 0
[X]_R = X                     if [Theta] == 0

X* = X - [X]_R
```

定义位温：

```text
theta = T * (p00 / p)^kappa
```

这里必须保证：

- 整层位于地下时，积分贡献自动为 0。
- 整层全部在地表以上时，`[X]_R` 退化为普通 zonal mean。

### 3. `reference_state/koehler_solver.py`

Boer exact APE 依赖参考态压强函数 `pi(theta, t)` 与：

```text
N(pi) = 1 - (pi / p)^kappa
pi_Z = pi([theta]_R, t)
N_Z = N(pi_Z)
```

建议把求解器设计成独立类：

```python
class KoehlerReferenceState:
    def solve(self, theta, p, ps, phis):
        ...
```

推荐步骤：

1. 将 `p(lambda, phi, theta_hat, t)` 插值到等熵坐标。
2. 对完全位于自由大气中的等熵层，先取 area-mean pressure。
3. 对与地形相交的等熵层，按质量守恒约束进行迭代修正。
4. 返回 `pi(theta, t)` 及其派生场，例如 `pi_Z`、`pi_s`、`pi_sZ`。

这部分是整个 exact 实现的“心脏”，不应分散到别的模块。

### 4. `boer/reservoirs.py`

动能库：

```text
K_Z = int_M 0.5 [Theta] [V]_R . [V]_R dm
K_E = int_M 0.5 [Theta V* . V*] dm
```

APE 库：

```text
A_Z = A_Z1 + A_Z2
    = int_M cp Theta N_Z [T]_R dm
    + int_S (ps - pi_sZ) Phi_s d_sigma / g

A_E = A_E1 + A_E2
    = int_M cp Theta (N - N_Z) T dm
    + int_S (pi_sZ - pi_s) Phi_s d_sigma / g
```

实现注意事项：

- 这里的 `K` 指水平风动能，不包含垂直动能。
- 与 SEBA 校验时应比较：

```text
K_Z + K_E  <->  total horizontal KE from SEBA
```

而不是 `hke + vke`。

### 5. `boer/conversions.py`

建议先实现最稳健的核心项，再逐步补地形项。

基础转换项：

```text
C_Z = C_Z1 + C_Z2
    = - int_M [Theta] [omega]_R [alpha]_R dm
      - int_S [dps/dt * Phi_s] d_sigma / g

C_E = - int_M [Theta omega* alpha*] dm
```

APE 内部转换：

```text
C_A = - int_M cp (theta / T)
      (
        [Theta T* V*] . grad
        + [Theta T* omega*] d/dp
      )
      ((T / theta) N_Z) dm
```

KE 内部转换：

```text
C_K = C_K1 + C_K2
```

其中 `C_K2` 含显式地形与时间导数项，因此对时间分辨率要求高。

实现要点：

- `C_K2` 和 `C_Z2` 都要求 snapshot 级或足够高时间分辨率数据。
- 某些被微分量已经是 zonal-mean 时，不需要再做经向导数。
- 若模式未输出 geopotential，需要通过静力关系重建。

### 6. `boer/sources_sinks.py`

热源项：

```text
G_Z = int_M Theta N_Z [Q]_R dm
G_E = int_M Theta (N - N_Z) Q dm
```

涡动耗散：

```text
F_E = - int_M [Theta V* . F*] dm
```

建议内部使用正耗散率记法：

```text
F_Z_pos = - int_M [Theta] [V]_R . [F]_R dm
```

并在预算方程中统一写为：

```text
dK_Z/dt = C_Z - C_K - F_Z_pos
```

如果数据中没有显式 `Q` 与 `F`，则：

- 先只做 `(A, K, C)`。
- 再在 global/time-mean 层面用残差恢复 `(G, F)`。
- 不要假装能从粗时间平均数据恢复逐点源汇。

---

## SEBA 适配层设计

### 1. `spectral/seba_adapter.py` 的职责

仅负责三件事：

1. 把已经按火星规则预处理和 mask 的数据喂给 SEBA。
2. 注入火星常数层或使用火星专用 backend。
3. 输出谱诊断结果供交叉校验和附加科学分析使用。

### 2. 输入数据要求

在进入 SEBA 之前必须先依据 surface pressure 构造：

```text
Theta = H(ps - p)
```

并将 `u, v, omega, T` 的 below-ground 单元做成 masked array。

### 3. 火星常数适配策略

推荐做法：

- 新增 `spectral/seba_constants_mars.py`。
- 在适配层中集中管理火星常数。
- 尽量避免直接 monkey patch `src/seba/constants.py` 的模块级全局变量。

最少需要一致替换：

```text
radius, Omega, g, Rd, cp, chi, p0
```

### 4. SEBA 在整体系统中的定位

SEBA 只作为：

- 谱能量分布诊断器。
- 尺度传递分析器。
- 总水平 KE 的交叉验证工具。

SEBA 不作为：

- Boer exact APE 真值来源。
- `(K_Z, K_E)` 的反演器。
- 地形项 `(AE_2, AZ_2, CK_2, CZ_2)` 的替代实现。

---

## 推荐实现顺序

### 阶段 1: 先做不依赖参考态的稳定基础层

优先完成：

- `constants_mars.py`
- `io/mask_below_ground.py`
- `common/grid_weights.py`
- `common/integrals.py`
- `common/zonal_ops.py`
- `boer/reservoirs.py` 中的 `K_Z`, `K_E`
- `boer/conversions.py` 中的 `C_E`, `C_Z1`

阶段目标：

- 确认 `K_Z + K_E` 与网格场直接积分得到的总水平 KE 一致。
- 与 SEBA 的水平 KE 谱积分做交叉验证。

### 阶段 2: 实现参考态求解器

完成：

- `reference_state/koehler_solver.py`
- `reference_state/interpolate_isentropes.py`
- `boer/reservoirs.py` 中的 `A_Z`, `A_E`
- `boer/conversions.py` 中的 `C_A`, `C_K1`

阶段目标：

- 在不引入显式地形时间导数项的前提下，建立可运行的 Boer exact 主体。

### 阶段 3: 补全 exact topographic terms

完成：

- `A_Z2`, `A_E2`
- `C_Z2`, `C_K2`
- `common/time_derivatives.py`
- `common/geopotential.py`

阶段目标：

- 引入所有显式地形项。
- 处理 `dps/dt` 与 `dPhi*/dt`。

### 阶段 4: 接入 SEBA 谱支路

完成：

- `spectral/seba_constants_mars.py`
- `spectral/seba_adapter.py`
- `spectral/spectra_validation.py`

阶段目标：

- 输出火星谱诊断。
- 对总水平 KE 和部分统计量做交叉校验。

---

## 必做验证清单

### 1. Flat-surface 极限测试

将地形压平并令 `ps` 为常数，检查：

```text
AZ_2 -> 0
AE_2 -> 0
CZ_2 -> 0
CK_2 -> 0
```

### 2. KE 分拆测试

验证：

```text
K_Z + K_E = int_M 0.5 Theta (u^2 + v^2) dm
```

这是第一优先级测试，不依赖参考态，最适合最早验错。

### 3. `[.]_R` 算子测试

需要覆盖以下极限：

- 整层完全高于地表时，`[X]_R` 退化为普通 zonal mean。
- 整层完全位于地下时，积分贡献自动归零。
- 部分经度被地形截断时，`[Theta X] / [Theta]` 工作正常。

### 4. 预算闭合测试

至少在 global-mean / time-mean 层面检查：

```text
dA_Z/dt = G_Z - C_Z - C_A
dA_E/dt = G_E - C_E + C_A
dK_Z/dt = C_Z - C_K - F_Z
dK_E/dt = C_E + C_K - F_E
```

### 5. 参考态守恒测试

Koehler solver 必须满足“等熵层间质量守恒”这一硬约束，否则后续 `N(pi)` 与 surface terms 全部不可信。

### 6. SEBA 交叉验证边界

只对下面对象做强约束交叉验证：

- 总水平 KE。
- 部分谱分布统计。

不对下面对象做强制一致性要求：

- Boer exact APE 与 SEBA APE。
- `(K_Z, K_E)` 与 SEBA 的 rotational / divergent 分解。

---

## 代码实施原则

### 1. 保持 SEBA 主体最小侵入

- 尽量不直接重写 `src/seba` 内核。
- 如果必须为适配火星常数增加扩展点，优先做小而清晰的接口修改。

### 2. Boer 物理与谱诊断彻底分层

- Boer 分支必须能在没有 SEBA 的情况下独立运行。
- SEBA 分支失败时，不应阻塞 exact 四储库预算。

### 3. 所有约定集中管理

应集中定义：

- 符号约定。
- 积分约定。
- mask 约定。
- 地形截断约定。
- 正负号约定。

避免“同一个记号在多个模块里含义不同”。

### 4. 先闭合，再扩展

优先得到一个可闭合、可测试、只包含基础项的最小 Boer 实现，再逐步补上参考态和地形项，不要一开始同时推进全部公式。

---

## 建议的第一批实际代码骨架

如果后续开始落地编码，建议第一批直接创建以下文件：

```text
src/mars_exact_lec/constants_mars.py
src/mars_exact_lec/io/mask_below_ground.py
src/mars_exact_lec/common/grid_weights.py
src/mars_exact_lec/common/integrals.py
src/mars_exact_lec/common/zonal_ops.py
src/mars_exact_lec/boer/reservoirs.py
```

理由：

- 这些模块共同定义了常数、mask、质量积分和 zonal/eddy split。
- 它们决定后续所有能量库与转换项是否可正确实现。
- 这一步即可先完成 `K_Z + K_E` 的封闭性测试。

---

## 参考文献与链接

- SEBA repository: <https://github.com/DataWaveProject/SEBA>
- SEBA `seba.py`: <https://raw.githubusercontent.com/DataWaveProject/SEBA/main/src/seba/seba.py>
- SEBA `constants.py`: <https://github.com/DataWaveProject/SEBA/blob/main/src/seba/constants.py>
- Tabataba-Vakili et al. (2015): <https://gemelli.colorado.edu/~lmontabone/Tabataba-Vakili_et_al_2015.pdf>
- Supplementary material: <https://oro.open.ac.uk/45065/2/__userdata_documents9_srl89_Desktop_supplementarymaterial_FTV_lorenz_mars.pdf>
- Koehler / Tellus reference-state material: <https://tellusjournal.org/articles/1834/files/submission/proof/1834-1-43486-1-10-20220929.pdf>

---

## 最终执行摘要

本方案的最终目标不是“让 SEBA 假装支持 Boer 1989”，而是：

1. 在当前仓库中保留 SEBA 作为谱诊断后端。
2. 新增一个独立的 `mars_exact_lec` 包实现火星 Boer-exact 四储库预算。
3. 共享预处理与数学基础层。
4. 用 SEBA 做 KE 与谱尺度诊断的交叉验证，但不让它替代 exact APE 与地形项实现。

后续若进入编码阶段，应从 `constants_mars.py`、`mask_below_ground.py`、`zonal_ops.py`、`reservoirs.py` 的第一阶段骨架开始。
