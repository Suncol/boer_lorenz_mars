# AGENTS.md

## Python 环境

本项目的 Python 虚拟环境位于仓库根目录下的 `.venv`。

- 进入虚拟环境时，必须使用：`source .venv/bin/activate`
- 执行 Python 时，应使用解释器路径：`.venv/bin/python`

请不要默认使用系统 Python，也不要假设其他虚拟环境与本仓库兼容。

## 项目目标

当前仓库的核心基础仍然是 `src/seba`，它提供面向干绝热、静力大气的谱能量预算分析能力，是本库现有的谱诊断主干。

本项目后续的目标不是把 SEBA 强行改造成 Boer (1989) 的四储库预算器，而是在同一仓库中扩展出一条面向火星的 `exact / topography-aware` Lorenz/Boer energetics 支路。最终形态应当是：在保留 SEBA 作为谱能量诊断后端的同时，新增一套火星 exact Lorenz energy cycle 实现，并让两条支路共享同一套预处理与公共数学层。

## 实现原则

本仓库推荐采用“双支路架构”：

- 支路 A：Boer-exact / topography-aware 四储库预算
- 支路 B：SEBA 谱能量诊断

两条支路共享同一套火星预处理核心，包括 pressure-level 数据准备、surface pressure 驱动的 below-ground masking，以及公共积分、梯度和热力学工具。

SEBA 在这里的角色是可复用的谱分析后端与交叉诊断工具，而不是 Boer exact 物理本体的替代品。任何后续实现都应避免把 SEBA 的谱 APE、代表性平均或地形处理直接等同为 Boer exact 四储库框架。
