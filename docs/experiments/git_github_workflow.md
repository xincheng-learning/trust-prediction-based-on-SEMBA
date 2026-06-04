# SEMBA Git/GitHub 实验工作流

这份文档说明本项目后续如何保持“原项目干净、实验线清楚、结果可回溯”。

## 1. 当前分支结构

```text
main
  稳定基线，只放确认可保留的代码和文档。

dev/current-env
  当前 conda 环境兼容开发线。
  例如 processed2、训练 bug 修复、smoke 脚本、统一指标。

exp/processed2-smoke
  processed2 与小规模复现实验。
  工作区：.worktrees/exp-processed2-smoke

exp/xgb-history
  XGBoost + 在线历史统计特征实验。
  工作区：.worktrees/exp-xgb-history
```

## 2. 哪些东西进 Git

应该提交：

```text
代码
测试
实验脚本
配置模板
实验计划
结果摘要
README/说明文档
```

不应该提交：

```text
data/
processed2/
results/
模型权重 .pt/.pth/.ckpt
pickle/joblib 模型
__pycache__/
.worktrees/
```

## 3. 后续与 Codex 协作的标准说法

开始一个新实验时，你可以这样说：

```text
请在 exp/xgb-history 对应 worktree 中实现 XGB 历史特征 baseline。
完成后先跑 5000 events smoke，再提交到该分支，不要合并。
```

或者：

```text
请从 dev/current-env 新建 exp/asym-decoder 分支和 worktree，
实现 coverage/status decoder，跑 BitcoinOTC-1 小规模实验。
```

## 4. 一个实验的标准流程

```text
1. 从 dev/current-env 新建 exp/<name> 分支
2. 在 .worktrees/<name> 中修改
3. 跑最小测试
4. 跑小规模实验
5. 保存结果摘要到 docs/experiments/
6. git commit
7. 如果稳定，再合并回 dev/current-env
8. 多个稳定实验积累后，再由 dev/current-env 合并回 main
```

## 5. 常用命令

查看当前分支：

```powershell
git branch --show-current
```

查看当前修改：

```powershell
git status --short
```

查看所有工作区：

```powershell
git worktree list
```

提交一次实验修改：

```powershell
git add <files>
git commit -m "feat: add xgb history baseline"
```

把实验分支合并回开发线：

```powershell
git switch dev/current-env
git merge exp/xgb-history
```

把开发线合并回稳定主线：

```powershell
git switch main
git merge dev/current-env
```

## 6. 重要安全规则

```text
不使用 git reset --hard，除非用户明确要求。
不删除 data/*/raw 或 data/*/processed。
不 force push。
大文件、模型权重、实验输出不进 GitHub。
每次合并前先看 git status。
```

## 7. GitHub 配合方式

推荐：

```text
main              -> GitHub 默认分支，稳定版
dev/current-env   -> 当前开发主线
exp/*             -> 实验分支，需要时推送备份
```

GitHub 上最理想的流程：

```text
exp/* 分支完成实验
  -> pull request 到 dev/current-env
  -> 讨论和检查结果
  -> 合并
  -> 阶段稳定后 pull request 到 main
```

