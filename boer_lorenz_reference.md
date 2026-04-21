# Boer/Lorenz 方案参考公式文档

## 目的

本文档用于整理 Boer (1989) exact pressure-coordinate energetics 在火星地形修正 Lorenz/Boer 能量循环中的参考公式，作为实现、复现、论文引用和附录撰写时的统一公式基线。

文档结构分为三层：

1. 规范版公式总表。
2. 与原始补充材料逐字对照时需要特别说明的排版/记号问题。
3. 符号表与注释版说明。

理论来源说明：

- 基础理论来自 Boer (1989) 的 exact pressure-coordinate energetics。
- 火星实现来源于 Tabataba-Vakili et al. (2015) 及其 supplementary material。
- APE 参考态的 uneven-topography 思路来自 Taylor (1979)。
- terrain-dependent reference state 的数值求解思路来自 Koehler (1986)。

---

## 一、规范版公式总表

### 0. 总能量与总收支

总 exact APE：

$$
A
=
\int_M c_p \, \Theta \, N \, T \, dm
+
\int_S (p_s - \pi_s)\,\Phi_s\,\frac{d\sigma}{g}
$$

总动能：

$$
K
=
\int_M \frac12 \,[\Theta\,\mathbf{V}\cdot\mathbf{V}]\,dm
$$

总收支：

$$
\frac{dK}{dt}=C-D,
\qquad
\frac{dA}{dt}=-C+G
$$

其中

$$
D=-\int_M \Theta\,\mathbf{V}\cdot\mathbf{F}\,dm,
\qquad
G=\int_M \Theta\,N\,Q\,dm
$$

以及效率因子

$$
N
=
1-\left(\frac{\pi}{p}\right)^\kappa
=
\frac{T-T_r}{T}
$$

四箱结构写法为

$$
\frac{\partial A_Z}{\partial t}=G_Z-C_Z-C_A,
\qquad
\frac{\partial A_E}{\partial t}=G_E-C_E+C_A
$$

$$
\frac{\partial K_Z}{\partial t}=C_Z-C_K-F_Z,
\qquad
\frac{\partial K_E}{\partial t}=C_E+C_K-F_E
$$

其中此处的 $C_K$ 定义方向为 $K_Z \to K_E$。

---

### 1. 动能储库

$$
K = K_Z + K_E
$$

$$
K_Z
=
\int_M \frac12 \,[\Theta]\,[\mathbf{V}]_R\cdot[\mathbf{V}]_R\,dm
\tag{1}
$$

$$
K_E
=
\int_M \frac12 \,[\Theta\,\mathbf{V}^*\cdot\mathbf{V}^*]\,dm
\tag{2}
$$

---

### 2. APE 储库

$$
A = A_Z + A_E
$$

$$
A_Z = A_{Z1} + A_{Z2}
=
\int_M c_p\,\Theta\,N_Z\,[T]_R\,dm
+
\int_S (p_s-\pi_{sZ})\,\Phi_s\,\frac{d\sigma}{g}
\tag{3}
$$

$$
A_E = A_{E1} + A_{E2}
=
\int_M c_p\,\Theta\,(N-N_Z)\,T\,dm
+
\int_S (\pi_{sZ}-\pi_s)\,\Phi_s\,\frac{d\sigma}{g}
\tag{4}
$$

---

### 3. 动能之间的转换

$$
C_K=C_{K1}+C_{K2}
$$

$$
\begin{aligned}
C_{K1}
=
-\int_M a\cos\phi \Bigg\{
&\left(
[\Theta u^*\mathbf{V}^*]\cdot\nabla
+
[\Theta u^*\omega^*]\frac{\partial}{\partial p}
\right)
\left(
\frac{[u]_R}{a\cos\phi}
\right)
\\[4pt]
&+
\left(
[\Theta v^*\mathbf{V}^*]\cdot\nabla
+
[\Theta v^*\omega^*]\frac{\partial}{\partial p}
-\frac{\tan\phi}{a}[\Theta\,\mathbf{V}^*\cdot\mathbf{V}^*]
\right)
\left(
\frac{[v]_R}{a\cos\phi}
\right)
\Bigg\}\,dm
\end{aligned}
\tag{5}
$$

$$
\begin{aligned}
C_{K2}
=
\int_M \Bigg\{
&\left[
\Theta\frac{\partial\Phi^*}{\partial t}
\right]
+
[\mathbf{V}]_R\cdot[\Theta\nabla\Phi^*]
+
[\omega]_R
\left[
\Theta\frac{\partial\Phi^*}{\partial p}
\right]
\Bigg\}\,dm
\end{aligned}
\tag{5'}
$$

---

### 4. APE 与 KE 之间的转换

$$
C_Z=C_{Z1}+C_{Z2}
=
-\int_M [\Theta]\,[\omega]_R\,[\alpha]_R\,dm
-\int_S
\left[
\frac{\partial p_s}{\partial t}\Phi_s
\right]
\frac{d\sigma}{g}
\tag{6}
$$

$$
C_E
=
-\int_M [\Theta\,\omega^*\alpha^*]\,dm
\tag{7}
$$

---

### 5. zonal APE 与 eddy APE 之间的转换

$$
\begin{aligned}
C_A
=
-\int_M
c_p\left(\frac{\theta}{T}\right)
\Bigg(
[\Theta T^*\mathbf{V}^*]\cdot\nabla
+
[\Theta T^*\omega^*]\frac{\partial}{\partial p}
\Bigg)
\left(
\frac{T}{\theta}N_Z
\right)\,dm
\end{aligned}
\tag{8}
$$

---

### 6. diabatic generation 与 frictional dissipation

$$
G_Z
=
\int_M \Theta\,N_Z\,[Q]_R\,dm
\tag{9}
$$

$$
G_E
=
\int_M \Theta\,(N-N_Z)\,Q\,dm
\tag{10}
$$

$$
F_Z
=
-\int_M [\Theta]\,[\mathbf{V}]_R\cdot[\mathbf{F}]_R\,dm
\tag{11}
$$

$$
F_E
=
-\int_M [\Theta\,\mathbf{V}^*\cdot\mathbf{F}^*]\,dm
\tag{12}
$$

说明：

- 上式中的 $F_Z$ 采用 implementation-consistent 的规范写法。
- 该写法与四箱预算中的 $-F_Z$、Boer 的总耗散定义、以及 representative decomposition 是一致的。

---

### 7. 定义式

地形 mask：

$$
\Theta(\lambda,\phi,p,t)=
\begin{cases}
1, & p<p_s(\lambda,\phi,t),\\
0, & p>p_s(\lambda,\phi,t)
\end{cases}
\tag{13}
$$

representative zonal mean：

$$
[X]_R=
\begin{cases}
[\Theta X]/[\Theta], & [\Theta]\neq 0,\\
[X], & [\Theta]=0
\end{cases}
\tag{14}
$$

扰动定义：

$$
X^*=X-[X]_R
\tag{15}
$$

效率因子：

$$
N(\pi)=1-\left(\frac{\pi}{p}\right)^\kappa
\tag{16}
$$

$$
N_Z=N(\pi_Z)
\tag{17}
$$

$$
\pi_Z=\pi([\theta]_R,t)
\tag{18}
$$

$$
[\theta]_R=[T]_R\left(\frac{p_{00}}{p}\right)^\kappa
\tag{19}
$$

水平梯度算子：

$$
\nabla=
\begin{pmatrix}
\partial_x\\[2pt]
\partial_y
\end{pmatrix}
=
\begin{pmatrix}
\dfrac{1}{a\cos\phi}\partial_\lambda\\[6pt]
\dfrac{1}{a}\partial_\phi
\end{pmatrix}
\tag{20}
$$

---

## 二、与原始补充材料逐字对照时必须注明的两处问题

### 1. eq. (5) 的排版问题

原补充材料中的 eq. (5) 在版面上容易被读成“两个 conversion block 的乘积”，但规范实现应理解为“两个对 mean shear 的贡献相加”。

理由：

- 若按乘积解释，量纲和结构都会变得不自洽。
- 在 Boer 的 mean/eddy kinetic-energy derivation 中，$C_{K1}$ 应是对 mean zonal wind shear 与 mean meridional wind shear 的线性和。
- 因此本文档将 eq. (5) 规范写为“第一块 $+$ 第二块”的形式。

这不是改理论，而是把明显不自洽的排版恢复为可推导、可计算的表达式。

### 2. eq. (11) 的印刷版与实现式不一致

原补充材料印刷版的 eq. (11) 若直接照抄，会与四箱预算中的符号约定以及 Boer 的总耗散定义不一致。

印刷版容易被读为：

$$
F_Z^{\mathrm{printed}}
=
\int_M [\Theta]\,[\mathbf{V}]_R\cdot[\mathbf{F}]\,dm
$$

而为了与

$$
\frac{\partial K_Z}{\partial t}=C_Z-C_K-F_Z
$$

以及

$$
D=-\int_M \Theta\,\mathbf{V}\cdot\mathbf{F}\,dm
$$

保持一致，最稳妥的 component split 应写成：

$$
D = F_Z + F_E
=
-\int_M [\Theta]\,[\mathbf{V}]_R\cdot[\mathbf{F}]_R\,dm
-\int_M [\Theta\,\mathbf{V}^*\cdot\mathbf{F}^*]\,dm
$$

因此：

- 正式公式表建议采用本文档中的规范版 eq. (11)。
- 原印刷版 eq. (11) 应视为排版或记号不一致点，而不是最终的实现式。

---

## 三、符号表

### 1. 坐标、几何与平均

- $\lambda$：经度。
- $\phi$：纬度。
- $p$：压力坐标。
- $t$：时间。
- $a$：行星半径。
- $M$：大气质量积分域。
- $S$：行星表面积分域。
- $dm$：大气质量元。
- $d\sigma$：表面积元。
- $[X]$：普通 zonal mean。
- $[X]_R$：representative zonal mean，定义为 $[\Theta X]/[\Theta]$，当 $[\Theta]\neq 0$ 时成立。
- $X^*=X-[X]_R$：相对于 representative mean 的扰动。
- $\nabla=((a\cos\phi)^{-1}\partial_\lambda,\ a^{-1}\partial_\phi)^T$：水平梯度算子。

### 2. 力学变量

- $\mathbf{V}=(u,v)$：水平风矢量。
- $u$：纬向风。
- $v$：经向风。
- $\omega=Dp/Dt$：pressure coordinates 下的垂直速度标量。
- $\Phi$：位势。
- $\Phi_s$：地表位势。
- $\alpha=1/\rho$：比容。
- $\rho$：密度。
- $\mathbf{F}$：进入动量方程的摩擦项向量。

实现注记：

- 方程结构要求 $\omega$ 是标量，而不是向量。
- 方程结构要求 $\mathbf{F}$ 是能与 $\mathbf{V}$ 点积的摩擦向量，而不是一个 scalar dissipation。

### 3. 热力学与参考态变量

- $T$：温度。
- $\theta$：位温。
- $Q$：加热率。
- $c_p$：定压比热。
- $R$：气体常数。
- $\kappa=R/c_p$。
- $p_{00}$：位温定义中的参考压强。
- $p_s$：实际地表压强。
- $\pi$：参考态压强。
- $\pi_s$：参考态地表压强。
- $\pi_Z=\pi([\theta]_R,t)$：由 representative zonal-mean 位温映射到参考态后的压强。
- $\pi_{sZ}$：对应 zonal reference state 的 surface pressure。
- $N(\pi)=1-(\pi/p)^\kappa=(T-T_r)/T$：efficiency factor。

注记：

- Boer 明确指出 $N$ 度量观测态与参考态温度的差异。
- $N$ 可以为负，因此不能把它当成一个恒正权重。

### 4. 能量库与转换项

- $A$：total exact available potential energy。
- $K$：total kinetic energy。
- $A_Z, A_E$：zonal 与 eddy APE。
- $K_Z, K_E$：zonal 与 eddy kinetic energy。
- $A_{Z1}, A_{E1}$：APE 的体积分部分。
- $A_{Z2}, A_{E2}$：APE 的 surface/topographic 部分。
- $C_Z$：$A_Z \to K_Z$ 的转换。
- $C_E$：$A_E \to K_E$ 的转换。
- $C_A$：$A_Z \to A_E$ 的转换。
- $C_K$：$K_Z \to K_E$ 的转换。
- $C_{Z2}, C_{K2}$：因显式保留地形而出现的额外项。
- $G_Z, G_E$：非绝热加热导致的 APE 生成项。
- $F_Z, F_E$：摩擦耗散项。

### 5. 地形 mask

- $\Theta$：Heaviside-type mask。
- 当某一 pressure point 位于地表之上时，$\Theta=1$。
- 当某一 pressure point 位于地表之下时，$\Theta=0$。

实现上建议直接按照分段定义理解，不要机械套用某个固定的 Heaviside 自变量记号约定。

---

## 四、注释版：真正需要记住的五点

### 注释 1：这套 exact equations 是相对于 hydrostatic primitive equations 且在 isobaric coordinates 中的 exact

- 这里的 “exact” 是相对于 hydrostatic primitive equations 的 exact。
- 它是在 pressure coordinates 中推导的，不是一个对所有坐标系普适不变的 universal exact mean/eddy equation set。
- 一旦做 mean/eddy split，mean terms 会依赖所选择的垂直坐标。

### 注释 2：$K$ 的分解是标准 quadratic decomposition；$A$ 的分解不是

- 动能是二次型，因此 $K=K_Z+K_E$ 来自标准的 mean/eddy decomposition。
- APE 不是简单二次型，因此 $A_Z$ 不是某个简单的 variance-like quantity。
- 更准确地说，$A_Z$ 是“把 total exact APE functional 作用在 mean state 上得到的量”，而 $A_E=A-A_Z$。

### 注释 3：为什么会出现 topographic terms

- uneven topography 会直接改变 APE 定义与其时间变化率。
- 因此在显式保留地形的 exact formulation 中，必须出现 $A_{Z2}, A_{E2}, C_{Z2}, C_{K2}$ 等项。
- 这些项对火星不是小修正，而是主方程结构的一部分。

### 注释 4：哪些 topographic terms 在 flat-surface approximation 中会消失

在 flat-surface approximation 中，下列“2”类项应退化为零：

- $A_{Z2}$
- $A_{E2}$
- $C_{Z2}$
- $C_{K2}$

这也是后续数值验证中最重要的极限测试之一。

### 注释 5：参考态的物理意义

- Taylor (1979) 讨论了 uneven topography 下 APE 的参考态问题。
- Koehler (1986) 给出了 terrain-dependent iterative algorithm，用来从真实资料求出包含地形效应的 reference atmosphere。
- 火星论文明确说明其参考态采用了 Koehler 的 terrain-dependent 方法。

---

## 五、引用时最稳妥的写法

英文建议写法：

> Following the exact pressure-coordinate formulation of Boer (1989), as implemented for Mars by Tabataba-Vakili et al. (2015, supporting information), the topographical Lorenz/Boer energy-cycle equations are given by equations (1)–(20) above, with the implementation-consistent normalizations of eq. (5) and eq. (11) noted in the text.

中文建议写法：

> 本文采用 Boer (1989) 的 pressure-coordinate exact energy equations，并按 Tabataba-Vakili et al. (2015) 补充材料中的火星实现式写成地形修正的 Lorenz/Boer 能量循环方程；其中 eq. (5) 与 eq. (11) 采用与总预算、representative decomposition 和耗散定义一致的规范化写法。

这样写的优点是：

- 理论来源交代清楚。
- 火星实现来源交代清楚。
- 对两处排版/记号不一致的处理方式交代清楚。

---

## 六、参考文献与链接

- Tabataba-Vakili et al. (2015): <https://gemelli.colorado.edu/~lmontabone/Tabataba-Vakili_et_al_2015.pdf>
- Boer exact pressure-coordinate energetics: <https://tellusjournal.org/articles/1724/files/submission/proof/1724-1-43268-1-10-20220928.pdf>
- Taylor (1979) uneven-topography APE: <https://tellusjournal.org/en/articles/10.3402/tellusa.v31i3.10430>

---

## 七、使用建议

若后续要把本仓库中的实现与该文档对齐，建议优先将下列对象写成显式代码接口：

- `Theta`
- `[X]_R`
- `X^*`
- `K_Z`, `K_E`
- `A_Z`, `A_E`
- `C_Z`, `C_E`, `C_A`, `C_K`
- `G_Z`, `G_E`, `F_Z`, `F_E`
- `pi(theta, t)` 与 `N(pi)`

并在实现文档或测试中明确说明：

- eq. (5) 按规范加法结构实现。
- eq. (11) 按 implementation-consistent 的正耗散率写法实现。
- flat-surface limit 下所有 “2” 类地形项应消失。
