# DSG²former 开源代码

英文主说明：[README.md](README.md)。目录形式参考 [RUDG](https://github.com/yzhouli/RUDG)，将安装、配置、数据准备、训练、测试和引用分开。

本目录独立于原实验工程。模型与现有训练模块已保留，提供六个数据集的原版/稀疏版配置，以及 `preprocess.py`、`train.py`、`test.py` 三个入口。原始社交数据、预训练模型、服务器调度脚本和模拟图表没有打包。

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/verify_release.py
```

当前公开入口从**审计后的 184 维因果图缓存**开始。`preprocess.py` 检查并导入已有缓存，不是从原始平台数据自动生成全部特征。所需字段见 [DATA.md](docs/DATA.md)。SocialNet、TikTok 的公开获取或分发方式仍需确定，因此目前应称为“预处理数据接口版代码发布”，不能声称六个数据集都可从公开原始数据一键复现。

```bash
python preprocess.py --config configs/pheme.json \
  --graphs /path/to/graphs.pt --threads /path/to/threads.jsonl
python train.py --config configs/pheme.json --device cuda:0
python test.py --config configs/pheme.json --device cuda:0
```

其他数据集换对应配置；稀疏版换成 `*_sparse.json`。各版本输出目录独立，输入、源码或配置改变时拒绝混用已有检查点。`--stage micro`、`macro`、`optimized` 可分别运行。`test.py --replay` 重新推理并检查与保存数组是否完全一致。

本包不附带实验指标结果、来源哈希清单、服务器验证报告或模型权重。训练和评估代码保留，运行后可自行生成结果。详细数据划分、弱标签及 Twitter16 人工平衡说明见 [PROTOCOL.md](docs/PROTOCOL.md)。

MIT 许可证仅覆盖本包原创代码；数据、教师模型和外部资源保留各自条款。作者名单来自提供的论文；发表元数据确定后更新 `CITATION.cff`。
