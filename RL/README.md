# robo1 getup PPO

在 MuJoCo 中训练二自由度机器人 `robo1` 从倒地姿态起身。保留原项目的教师预训练流程，并新增从随机网络开始的多角度课程训练。

## Files

- `robo1.xml` - MuJoCo 模型，质量和惯量已按本地实验调整
- `assets/` - 模型引用的 STL 网格
- `robo1_env.py` - 基础 Gymnasium/MuJoCo 环境及成功条件
- `train_curriculum.py` - 新增的多角度 PPO 课程训练
- `eval_curriculum_policy.py` - 指定检查点的四姿态评估和动画回放
- `robo1_getup_ppo.zip` - 原作者提供的策略，未用新策略覆盖
- `play_robo1_policy.py` - 原项目策略回放入口
- `eval_robo1_policy.py` - 原项目策略评估入口
- `search_all_getup.py` - 搜索起身轨迹候选
- `getup_reference.py` - 教师动作轨迹
- `scripted_getup.py` - 教师动作回放
- `pretrain_robo1_from_scripted.py` - 教师数据预训练
- `train_robo1.py` - 原项目 PPO 训练入口
- `export_policy_header.py` - 导出 Arduino 策略头文件
- `requirements.txt` - Python 依赖
- `runs/` - 课程训练生成的实验目录，不提交到 Git

## Setup

以下命令使用 Windows PowerShell。先在仓库根目录创建虚拟环境，再进入 `RL/` 安装依赖；后续命令均在 `RL/` 中执行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd RL
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

本地成功实验使用 Python 3.13.2、MuJoCo 3.17.0、Gymnasium 1.4.0、Stable-Baselines3 2.9.2、PyTorch 2.15.0、NumPy 2.5.4。当前依赖文件未锁定版本，安装时会解析可用版本。

## Play

本仓库不附带本地课程训练的成品权重。先按下方 Training 章节训练，再把自己实验的检查点复制到 `RL/`（替换示例时间戳）：

```powershell
Copy-Item runs/curriculum_时间戳/best_policy.zip curriculum_policy.zip
```

`curriculum_policy.zip` 是上述命令生成的本地文件，不随仓库提供。回放它，依次循环四个倒地方向：

```powershell
python eval_curriculum_policy.py play --model curriculum_policy.zip --pose all
```

只回放一个方向：

```powershell
python eval_curriculum_policy.py play --model curriculum_policy.zip --pose roll_pos
```

可选方向为 `roll_pos`、`roll_neg`、`pitch_pos`、`pitch_neg`。这是已训练策略的回放；课程训练脚本本身不显示实时训练动画。

原项目的 `python play_robo1_policy.py` 仍加载原作者的 `robo1_getup_ppo.zip`，与新策略分开保存。由于物理参数已修改，原作者策略在当前模型上的表现需要重新评估。

## Evaluate

完成训练并按 Play 章节复制检查点后，使用基础环境的四个完全倒地姿态和原成功条件进行独立验收：

```powershell
python eval_curriculum_policy.py evaluate --model curriculum_policy.zip --output acceptance.json
```

成功要求连续 50 个控制步满足直立、关节角、底座高度和速度条件，在 50 Hz 下约为 1 秒。

以下是 2026-11-04 本地历史实验结果，作为训练方法的参考；对应成品权重不包含在发布目录中。步数包含保持成功姿态的时间：

| 初始姿态 | 成功 | 控制步数 |
| --- | --- | --- |
| `roll_pos` | 是 | 104 |
| `roll_neg` | 是 | 105 |
| `pitch_pos` | 是 | 110 |
| `pitch_neg` | 是 | 113 |

额外对每个方向分别加入 −4°、−2°、0°、2°、4° 的初始倾角扰动，本地测试通过 20/20。由于先进行落稳，这些扰动可能收敛到相似状态，不能据此认为策略能应对任意跌倒姿态。新策略尚未完成实物起身验证。

## Training

### 相对原项目的修改

新增训练在 `train_curriculum.py` 中继承 `Robo1GetupEnv`，保留基础环境文件和原训练入口。修改集中在课程环境的 `reset()`、`step()` 和 PPO 训练循环：

| 修改位置 | 具体变化 | 目的 |
| --- | --- | --- |
| 初始化 | 默认随机初始化，不运行教师预训练、不加载原策略 | 从零学习起身 |
| `reset()` | 随机倒地方向，按课程阶段采样倾角 | 先学小倾角恢复，再学习完全倒地起身 |
| `step()` | 替换基础环境返回的奖励 | 减少维持半起身状态的收益，鼓励恢复直立并稳定 |
| 回合上限 | 训练由基础环境的 700 步改为 400 步 | 减少无进展回合的采样开销 |
| PPO | 启用 gSDE，并调整学习率、折扣等参数 | 使用具有时间连续性的探索动作 |
| 训练循环 | 分块训练、逐级提升难度，每块检查四种完全倒地姿态 | 根据通过率推进课程并保存检查点 |

保留 4 维观测、2 维动作和 50 Hz 控制接口。验收不使用课程环境的奖励和重置，而是使用基础环境的四个固定倒地姿态，并保持连续 50 步成功的条件。

课程奖励的具体形式为：

```text
gate = clip((upright - 0.65) / 0.35, 0, 1)
potential = upright - 0.15 × gate × sum(target²)
reward = 20 × (0.995 × potential_next - potential_prev)
         - 0.2 - 0.8 × (1 - upright) - 0.001 × sum(action²)
满足原 goal_pose 条件：额外 +8
连续满足成功条件并终止：额外 +100
```

`upright` 是基础环境的直立指标，`target` 是两个舵机的目标角。基础环境原先返回的奖励在课程训练中被替换，不与上述奖励相加。

当前 PPO 使用 `learning_rate=2e-4`、`gamma=0.995`、`n_steps=512`、`batch_size=512`、`ent_coef=0.002`、`use_sde=True`、`sde_sample_freq=16` 和 `log_std_init=-0.5`，在 CPU 上训练。

### 多角度课程训练

从随机网络开始训练，不加载原作者策略，也不使用教师数据：

```powershell
python train_curriculum.py --blocks 24 --chunk 32768 --envs 8 --seed 7
```

课程阶段为 `5、10、15、20、30、35、40、45、55、65、75、90°`。每回合随机选择四个方向之一，并在当前阶段角度的 70%～100% 之间采样初始倾角，落稳 200 个物理步后开始控制。

每个训练块结束后，同时评估当前难度和四个完全倒地姿态：当前难度至少通过 3/4 时进入下一阶段；完全倒地通过 4/4 时提前停止。

训练保留 4 维观测（roll、pitch、两个舵机目标角）、2 维目标角增量动作和 50 Hz 控制接口。每步目标角最大增量为 0.08 rad，目标角限制为 ±1.55 rad。8 个环境通过 `DummyVecEnv` 顺序采样，不是 8 个并行进程。

课程环境改变了重置姿态、奖励和回合长度（400 个控制步）。奖励使用直立势能变化、未站立代价、动作代价、稳定站姿奖励和成功奖励，减少策略停留在半起身状态的问题。完全倒地验收仍使用基础环境的原成功条件。

| 参数 | 含义与默认值 |
| --- | --- |
| `--blocks` | 最多训练块数，默认 18；上例设为 24 |
| `--chunk` | 每块请求采样步数，默认 32768 |
| `--envs` | 采样环境数，默认 8 |
| `--seed` | 新策略随机种子，默认 7 |
| `--start-angle` | 起始课程阶段，默认 5，需使用已有阶段值 |
| `--resume` | 续训检查点；默认不加载 |

每次训练创建 `runs/curriculum_时间戳/`，包含模型和训练源码快照、`configuration.json`、`history.json`、`logs/progress.csv`、`latest_policy.zip` 和 `best_policy.zip`。当前脚本还会在退出清理流程中写入 `training_result.json`。

`best_policy.zip` 只在完全倒地成功数首次达到新高时更新，同分不替换。训练输出仅保存在对应实验目录，不会自动复制为 `curriculum_policy.zip`，也不会自动导出头文件或烧录。PPO 按完整批次采样，实际步数可能多于请求值；强制结束进程可能丢失尚未保存的进度。

从自己的检查点继续训练：

```powershell
python train_curriculum.py --resume runs/curriculum_时间戳/latest_policy.zip --start-angle 45 --blocks 16 --chunk 32768 --envs 8
```

续训创建新实验目录，并保留累计采样步数。课程阶段需手动指定；模型会重新从当前 `RL/` 创建快照，续训前确认物理参数与检查点匹配。

评估新实验时，应使用该实验的模型快照。将示例时间戳替换为实际目录：

```powershell
$run = (Resolve-Path runs/curriculum_时间戳).Path
$tool = (Resolve-Path eval_curriculum_policy.py).Path
Push-Location (Join-Path $run 'model')
try {
    python $tool evaluate --model (Join-Path $run 'best_policy.zip') --output (Join-Path $run 'acceptance.json')
    python $tool play --model (Join-Path $run 'best_policy.zip') --pose all
} finally {
    Pop-Location
}
```

本地成功实验先从随机网络训练 196608 步，再续训至累计 294912 步。第一版课程在 30° 后直接进入 45°；当前版本增加了 35°、40° 阶段，因此当前默认命令并非该次实验的逐步精确复现。不同随机种子和设置不保证相同步数收敛。

### 原项目训练流程

原项目通过教师轨迹预训练，再使用 PPO 追加学习。以下命令会生成或更新 `robo1_getup_ppo.zip`：

```powershell
python pretrain_robo1_from_scripted.py
python train_robo1.py --model-in robo1_getup_ppo.zip --timesteps 200000 --n-envs 6 --model-out robo1_getup_ppo
```

需要重新搜索教师轨迹时，运行 `python search_all_getup.py`，再把输出轨迹写入 `getup_reference.py`。多角度课程训练不依赖此流程。

## Export for Arduino

先训练、按 Play 章节复制自己的检查点并完成评估，再导出 Arduino 头文件：

```powershell
python export_policy_header.py curriculum_policy.zip -o policy_network.h
```

将生成的 `policy_network.h` 放入 `Arduino/robo03/`，替换该目录中的原头文件。当前仓库 Arduino 目录保留原作者头文件，尚未替换为课程策略。

## Notes

- 策略依赖对应的 `robo1.xml`、`robo1_env.py` 和 `assets/`，应将检查点与实验快照配套使用。
- 自己的机器人需同步质量、重心、惯量、关节轴和接触参数；仅改变质量时按比例缩放惯量是保持质量分布不变的近似。
- 实物部署前应校准舵机方向、中位、脉宽、行程和控制周期。仿真通过不代表实物必然成功。
- 机器人结构、基础环境、教师轨迹和 Arduino 部署代码来自 [原项目](https://github.com/homemadegarbage/SelfRisingRobot)。
