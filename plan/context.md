# 背景与任务上下文（性能回退排查：模型/数据侧）

1. 必要上下文
- 近期对模型代码进行了重构，重构后的主模型文件为 models/t2m_unet.py，重构前对应文件为 /data/kuimou/backup/models/unet.py。
- 观察到重构后训练/推理出现显著性能下降，需要定位是模型结构/权重映射还是数据/训练流程差异导致。
- 已有两份同步训练步数的检查点（ckpt）：
  - 重构后（新）：/data/kuimou/openANT/checkpoints/t2m/ant_t2m/model/latest_120000.tar
  - 重构前（旧）：/data/kuimou/backup/checkpoints/t2m/t2m_t5/model/latest_120000.tar
- 重要先验：在论文术语对齐下，重构后 t2m_unet.py 中的 sta_model 与重构前 unet.py 中的 ella_model 结构一致，仅是命名更改。因此在权重对齐与 diff 时，需要将二者视为等价模块进行比对（权重名做等价映射）。

2. 目标
- 梳理并记录重构前后模型命名与结构的关键差异点（尤其是 sta_model ≡ ella_model 的一一对应关系）。
- 编写权重对比脚本，不忽略任何权重 key，输出树状结构的差异报告，区分：
  - 相同的权重（形状与数值完全一致/近似一致）
  - 不同的权重（仅存在于一方/形状不同/数值不同）。
- 在 diff 中对“不同”的项标注来源（仅新/仅旧/两者皆有但数值不同），并在开头汇总统计。

3. 产出物
- /debug/diff_weight.txt：两份 ckpt 的权重树状 diff 报告（考虑 sta=ella 的同构映射）。
- /tmp_scripts/diff_weights/run_compare.sh：一键运行脚本（含必要的注释与最小化的版本管理动作）。
- 本文件 /plan/context.md：记录背景信息、先验与目标，便于后续 Review 与复现实验。

4. 注意事项与约定
- 不忽略任何权重：如存在多级字典/模块化 state_dict，脚本应尽可能递归提取叶子张量进行对比；如能识别出 state_dict/模型权重优先以其为主。
- 命名规范：对比时将 key 中的“ella_model”标准化为“sta_model”进行匹配，避免命名差异导致的误判。
- 输出要求：树状结构可按 key 的点号分层，缩进展示；每个叶子结点附上状态说明（相同/不同）与来源。