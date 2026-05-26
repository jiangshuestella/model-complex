# ANL 轻量化改造建议（针对井下连续相位调制 BPSK）

## 关键问题定位

你当前 `FFTDb4WaveletANL` 表现差，核心不是“容量不够”，而是存在几个**结构性错误/不匹配**：

1. **相位被破坏（严重）**  
   代码里 `phase_corr = phase * self.phase_gain`，而 `phase_gain` 初始化为 0，会把相位直接压成 0，等价于强行重构成近似零相位信号。这会显著伤害 BPSK 的相位信息。

2. **没有 residual 回路（与注释不一致）**  
   你注释写的是 `adapted = x_fft + rho * tanh(residual)`，但 `forward` 实际 `return out`，导致网络输出不是“轻微校正”，而是“直接重建”，小样本目标域下容易崩。

3. **BatchNorm 在小样本导频下不稳定**  
   元学习 + 少量导频时，BN 统计量偏差很大，常导致域偏移更严重。建议换 GroupNorm/LayerNorm。

4. **参数预算被错误分配**  
   你把很多预算放在 db4 多层可训练分解/重构上，但目标域核心偏移通常在“窄带周期干扰 + 增益/偏置漂移 + 相位微扰”。更应优先做“参数化干扰抑制 + 小 residual”。

---

## 小参数量 ANL 的推荐结构（可直接替换）

推荐一个 **TinyANL（<30k 参数）**：

- **Stage A：频域幅度门控（保相位）**
  - `A' = A * (1 + α*tanh(g(f)))`
  - **不要改相位**（phase passthrough）
- **Stage B：时域深度可分离卷积 residual（2~3 层）**
  - DWConv(k=9/15) + PWConv + SiLU + GroupNorm
- **Stage C：有界输出残差**
  - `y = x_fft + rho * tanh(r)`
- **Stage D：可选低秩谐波抑制头（很省参数）**
  - 对估计泵频及其 2~4 次谐波做软掩蔽（非硬陷波）

这样比 wavelet 栈更稳，且参数更小。

---

## 训练目标（冻结 recovery_model）

设冻结恢复器为 `R(·)`，ANL 为 `A(·)`。

- 导频监督：`L_pilot = CE(R(A(x_pilot)), bits_pilot)`
- 一致性：`L_cons = ||A(x) - x||_1`（限制 ANL 改动幅度）
- 频域正则：`L_spec = || log|FFT(A(x))| - log|FFT(x_src_style)| ||_1`
- 谐波稀疏：`L_h = Σ_k w_k * E_k(A(x))`

总损失：
`L = L_pilot + λ1 L_cons + λ2 L_spec + λ3 L_h`

> 关键：`L_cons` 与 `tanh` 有界残差一起，用来防止 ANL 把信号“过拟合成导频模板”。

---

## 元学习设置建议

- **内环**：只更新 ANL 的最后 1~2 层（或者仅更新门控参数）
- **外环**：更新 ANL 全部参数
- 任务构造：按工况（深度、温度、泵速区间）分 task，而不是随机切片
- BN 全部冻结或改 GN

---

## 你当前代码最小改动清单（高优先级）

1. 删除/禁用 `phase_gain`，相位直接透传。  
2. 把 `return out` 改成 `return x_fft + rho * tanh(out)`。  
3. 所有 BN 换成 GN（如 `GroupNorm(4, C)`）。  
4. `gain_scale` 降到 0.05~0.2，并对 `amp_gain` 加 L2 或 TV 正则。  
5. wavelet 先设 `trainable_wavelet=False`，减少自由度。  

仅这 5 条，通常就能比当前版本明显稳。

---

## 参考方向（用于对齐思路）

- DANN / CDAN（域对齐思想，适合你“源域预训练 + 目标域少量标签”）
- FiLM / AdaIN（小参数条件调制，比大 ANL 更适合 few-shot 适配）
- Complex spectral mapping（强调保相位，对调制信号更友好）
- Noise2Noise / 自监督一致性（导频极少时可增强训练信号）

