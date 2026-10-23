# LFS 审稿意见 · Reviewer #2 验证复现代码

本项目重跑了应对 Reviewer #2 三项质疑所需的全部验证实验，结果与数据均已如实写入
`to Reviewers - 20241123 (R2回复已补全).docx`。本目录代码可独立复现全部数字。

## 一、运行环境

- 本机已确认可用环境：`C:\Users\86158\AppData\Local\Programs\Python\Python311\python.exe`
- 依赖（Python311 中已全部安装）：`pandas numpy xgboost scikit-learn joblib openpyxl`
- **不需要** tensorflow / rdkit / mordred —— 32 维特征直接复用训练时已保存的特征文件，
  无需重新生成分子描述符。

## 二、所需数据文件（原始位置，脚本内 `BASE` 常量指向这里）

| 文件 | 位置 | 作用 |
|---|---|---|
| `features_with_predictions-3754.xlsx` | `D:\wrapper\LFS\` | 3754 条样本的 32 维特征 + LFS + SMILES + train/test 标记 |
| `voting_regressor_model-LFS-3754.pkl` | `D:\wrapper\LFS\重新做不含RP-3的-3754条数据\` | 已训练 VotingRegressor 主模型（错配实验用） |

> 若你的数据在其他路径，修改每个脚本顶部的 `BASE` 常量即可。

## 三、三个验证脚本（按顺序运行）

### 1. 全量留一燃料验证 —— `lofo_leave_one_fuel_out.py`

对 **39 种燃料逐一留出**：每次用其余 38 种燃料的全部样本、以主模型完全相同超参数
重新训练 VotingRegressor，预测留出的那种燃料，输出 R²/RMSE/MAE/MAPE。

```
python lofo_leave_one_fuel_out.py
```

- 输出：`lofo_full_results.xlsx`（39 行逐燃料）+ 控制台逐条打印 + 汇总 JSON
- 耗时：约 4 分钟（训练 39 次，每次 5 个基模型）
- 汇总数值：mean MAE ≈ 20.3 cm/s，mean R² ≈ -13.8，69% 燃料 MAE>5

### 2. 错误分子描述符错配实验 —— `mismatch_descriptor_test.py`

用已训练主模型做审稿人点名的测试：**保留工况(T,p,φ)，把分子部分（One-Hot 嵌入 +
Mordred）替换为错误燃料的**，观察预测是否崩坏。

```
python mismatch_descriptor_test.py
```

- 输出：`mismatch_test_results.xlsx`
- 核心结论：错配后全部样本 MAE 从 2.05 升至 24.45（约 12 倍），R² 从 0.995 崩为 -0.511
- 秒级完成

### 3. 按化学族整族留出验证 —— `family_leave_out.py`

把某一整族燃料（烷烃/烯烃/醇类/环烷烃/芳烃/H₂）全部留出，用其余族训练，预测留出族。

```
python family_leave_out.py
```

- 输出：`family_leaveout_results.xlsx`
- 结论：同系物外推（醇类）尚可，跨族外推（芳烃/环烷烃/H₂）基本失败
- 耗时：约 4 分钟

## 四、输出文件与审稿回复的对照

| 回复中的表格 | 由哪个脚本产出 | 数据文件 |
|---|---|---|
| Table R2-1（LOFO 全 39 燃料） | `lofo_leave_one_fuel_out.py` | `lofo_full_results.xlsx` |
| Table R2-2（错配实验） | `mismatch_descriptor_test.py` | `mismatch_test_results.xlsx` |
| Table R2-3（族留出） | `family_leave_out.py` | `family_leaveout_results.xlsx` |

## 五、可复现性说明

- 所有基模型超参（XGBR: depth=4, n_estimators=250, lr=0.25; SVR: rbf, C=30000, γ=0.1）
  与主模型一致，随机种子固定（random_state=42）。
- VotingRegressor 权重 [0.5, 0.5]（对应主模型 weights=[0.5,0,0,0,0.5] 中仅 XGBoost+SVR 有效）。
- 错配实验随机错配部分使用 `np.random.RandomState(42)`，结果可复现。
