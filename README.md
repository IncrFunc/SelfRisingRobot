# SelfRisingRobot

二轴舵机起身机器人「poco」，使用 MuJoCo 和强化学习训练起身动作，并通过搭载 M5Atom 的实物机器人运行策略。本地实物使用 ATOM Matrix。

本项目基于 [homemadegarbage/SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot)，保留原项目的目录结构，增加从随机策略开始的多角度课程训练。机器人质量和惯量已按本地模型调整。

原作者详细介绍：[https://homemadegarbage.com/rl13](https://homemadegarbage.com/rl13)。

## Contents

- `3Dmodel/` - 实物机器人使用的 3D 打印模型
- `RL/` - MuJoCo 模型、强化学习环境、多角度训练脚本和 PPO 策略
- `Arduino/` - M5Atom 控制程序和导出的策略网络

各部分的详细使用方法见对应目录中的 README。

## 3D Model

`3Dmodel/` 包含实物机器人使用的 STL 文件：

- `footP.stl`
- `arm1P.stl`
- `arm2P.stl`
- `armhornP.stl`

## Reinforcement Learning

`RL/` 包含仿真模型、训练和评估脚本。

在原项目的基础上增加了多角度课程训练：从小倾角开始，逐步提高难度，同时评估四个完全倒地姿态。训练使用 PPO 和 gSDE 探索，不加载原作者权重，不使用教师动作；也支持从自己的检查点继续训练。

本次训练修改如下：

| 项目 | 原项目 | 本项目新增流程 |
| --- | --- | --- |
| 策略初始化 | 提供教师轨迹预训练及 PPO 追加训练，也支持随机初始化 | 默认从随机网络开始，不使用教师轨迹 |
| 初始状态 | 四个固定倒地方向 | 四个方向随机采样倾角，从 5° 逐步增加至 90° |
| 奖励 | 基础环境奖励 | 使用直立势能变化、未站立代价及稳定站姿奖励 |
| 探索 | 原 PPO 训练配置 | 启用 gSDE，调整 PPO 超参数 |
| 评估 | 独立评估脚本 | 每个训练块检查当前难度和四个完全倒地姿态 |
| 实验保存 | 训练模型输出 | 每次保存模型、训练源码、配置和评估记录 |

观测仍为 4 维，动作仍为 2 维，控制频率仍为 50 Hz；完全倒地验收沿用原环境的成功条件。课程训练的重置、奖励和回合长度有修改，模型质量与惯量也有调整。

仓库不附带本地课程训练的成品权重，需要自行训练。历史本地实验累计训练 294912 步后，四个固定倒地姿态评估通过 4/4。额外初始倾角扰动测试通过 20/20；该结果仅代表仿真，尚未完成新策略的实物起身验证。

安装、训练、回放和验证说明见 [RL/README.md](RL/README.md)。

## Arduino

`Arduino/` 包含原项目的 M5Atom 起身控制程序：

- `robo03/robo03.ino`
- `robo03/policy_network.h`

`policy_network.h` 是用于 M5Atom 推理的策略网络。当前 Arduino 目录仍保留原作者的头文件，未替换为本次课程训练结果。部署新策略前，需要重新导出并校准实机舵机方向、中位和行程。

固件依赖和上传方法见 [Arduino/README.md](Arduino/README.md)。
