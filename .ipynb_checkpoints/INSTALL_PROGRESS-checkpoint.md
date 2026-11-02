# Pointcept 1.7.0 安装配置进度记录

> 目标：在 base conda 环境中复现跑通 Pointcept 1.7.0 的安装配置。
> 代码目录：`/root/Pointcept-1.7.0`，GitHub: https://github.com/Pointcept/Pointcept

## 环境现状（2026-09-19 检查）

| 项目 | 状态 |
|---|---|
| GPU | NVIDIA RTX 4080（32GB，sm_89）✅ |
| 驱动 / CUDA | Driver 595.71.05，nvcc CUDA 12.8 ✅ |
| conda | 24.4.0（`/root/miniconda3`），已配清华镜像 ✅ |
| base Python | 3.12.3 |
| PyTorch | base 已装 **2.8.0+cu128**，`torch.cuda.is_available()=True` ✅ |
| pip 镜像 | 已配阿里云镜像 ✅ |
| 编译器 | gcc/g++ 11.4.0 |

**关键决策**：不降级到官方 environment.yml 的 torch 2.5.0+cu12.4，直接复用 base 中已可用的 torch 2.8.0+cu128，安装与之兼容的依赖，避免大体积重复下载。

## 总计划

1. [x] 阅读 README / environment.yml，摸清安装要求
2. [x] 检查本机 GPU / CUDA / conda / base 环境现状
3. [x] 安装基础 Python 依赖（h5py, pyyaml, tensorboardx, wandb, yapf, addict, einops, scipy, plyfile, termcolor, timm, ftfy, regex, tqdm, matplotlib, ninja, transformers==4.50.3, peft 等）✅ 2026-09-19
4. [x] 安装 PyG 相关：torch-scatter 2.1.2+pt28cu128 / torch-sparse 0.6.18+pt28cu128 / torch-cluster 1.6.3+pt28cu128 / torch-geometric 2.8.0 ✅ 2026-09-19（data.pyg.org 官方预编译 wheel）
5. [x] 安装 spconv-cu126 2.3.8 ✅ 2026-09-19（无 cu128 版，cu126 与 torch 2.8+cu128 实测兼容，`import spconv.pytorch` 通过）
6. [x] 安装 OpenAI CLIP ✅ 2026-09-19（GitHub 直连超时，改用 ghfast.top 代理下载 main.zip 安装，clip-1.0）
7. [x] 编译安装 libs/pointops ✅ 2026-09-19（TORCH_CUDA_ARCH_LIST=8.9，FarthestPointSampling CUDA kernel GPU 实测通过）
8. [x] 编译安装 libs/pointgroup_ops ✅ 2026-09-19（先 conda 装 bioconda 的 google-sparsehash，再编译）
9. [x] （可选）flash-attention 2.8.3 ✅ 2026-09-19（经 ghfile.geekertao.top 代理下载官方 cu12torch2.8cxx11abiTRUE-cp312 预编译 wheel 安装，PTv3 flash 路径实测通过）
10. [x] （可选）open3d 0.20.0 可视化 ✅ 2026-09-19（补装系统库 libegl1/libgl1 + conda-forge 新版 libstdcxx-ng 后导入通过）
11. [x] 验证 ✅ 2026-09-19：`import pointcept` 通过；`smoke_test.py` 用真实 config 构建 SpUNet(39.2M) 与 PTv3(46.2M)，GPU 前向+反向均通过（flash_attn=on）；引擎/数据集模块全量导入通过
12. [x] （可选）libs/pointrope 编译 ✅ 2026-09-19（LitePT CUDA 加速）

## 已做内容

- 2026-09-19：阅读 README.md 与 environment.yml，确认依赖清单与安装方式（官方推荐 conda 环境或手动安装）。
- 2026-09-19：检查本机环境（见上表），确认 pip/conda 均已配置国内镜像，无需额外配置。
- 2026-09-19：决策复用 base 环境现有 torch 2.8.0+cu128。
- 2026-09-19：pip 安装基础依赖一批（ninja/h5py/tensorboardx/wandb/yapf/addict/einops/scipy/plyfile/termcolor/timm/ftfy/regex/tqdm/matplotlib）。
- 2026-09-19：pip 安装 `transformers==4.50.3` + `peft 0.21.0`（huggingface_hub 自动降至 0.36.2 兼容版）。
- 2026-09-19：从 data.pyg.org 安装 torch2.8+cu128 预编译的 torch-scatter/torch-sparse/torch-cluster，pip 安装 torch-geometric。
- 2026-09-19：pip 安装 spconv-cu126 2.3.8（PyPI 无 cu128 版），导入实测通过。
- 2026-09-19：conda 从清华 bioconda 镜像安装 google-sparsehash。
- 2026-09-19：编译安装 `libs/pointops`（TORCH_CUDA_ARCH_LIST="8.9"），GPU 上 FPS kernel 测试通过。
- 2026-09-19：编译安装 `libs/pointgroup_ops`（--include_dirs=${CONDA_PREFIX}/include），导入通过。
- 2026-09-19：`import pointcept` 全量导入通过（models/__init__ 默认启用 sparse_unet/PTv1/PTv2/PTv3/stratified/spvcnn/octformer/oacnns/litept/point_group/msc/ppt/sonata/concerto/utonia 等）。
- 2026-09-19：pip 安装 open3d 0.20.0。
- 2026-09-19：CLIP 经 ghfast.top 代理安装成功（GitHub 直连 git clone 超时失败后的替代方案）。
- 2026-09-19：编写 `smoke_test.py`（项目根目录），用 `configs/scannet/semseg-spunet-v1m1-0-base.py` 与 `configs/scannet/semseg-pt-v3m1-0-base.py` 构建模型，随机点云 GPU 训练步测试：**SpUNet loss=3.02、PTv3 loss=4.48，前向+反向全部通过**（此时 flash_attn 未装，PTv3 自动走非 flash 路径）。
- 2026-09-19：编译安装 libs/pointrope（LitePT 的 PointROPE CUDA 实现），导入通过。
- 2026-09-19：flash-attn 首次经 ghfast.top 下载中断（55kB/s 太慢导致 wheel 截断），改用 ghfile.geekertao.top（约 3MB/s）下载完整 256MB wheel 安装成功；注意 pip 安装本地 wheel 必须保留原始文件名。
- 2026-09-19：修复 open3d 两个系统级问题：① 缺 libEGL.so.1 → `apt-get install -y libegl1 libgl1`；② conda 自带 libstdc++ 缺 GLIBCXX_3.4.30 → 清华 conda-forge 镜像安装新版 `libstdcxx-ng`。
- 2026-09-19：重跑 `smoke_test.py`，**SpUNet/PTv3 均 flash_attn=on 通过**；open3d、engines、atasets 全量导入通过。安装配置全部完成。

## 未做内容 / 待办

- 数据集准备与正式训练（需要 ScanNet/S3DIS 等原始数据，超出本次"安装配置"范围）。后续按 README `Data Preparation` 节下载数据 → 预处理 → `ln -s` 到 `data/` → `sh scripts/train.sh ...` 即可。

## 问题与解决记录

1. **GitHub 直连超时**：`pip install git+https://github.com/openai/CLIP.git` 卡住失败 → 改用 `https://ghfast.top/` 代理下载 zip 安装，成功。
2. **SpUNet 输入需 grid_coord**：`spconv_unet_v1m1_base.py` forward 直接取 `input_dict["grid_coord"]`，冒烟脚本中需先用 `grid_size=0.02` 算出 grid_coord 再传入。
3. **torch 版本取舍**：官方 yml 锁 torch 2.5.0+cu12.4，但 base 已有 torch 2.8.0+cu128 且 GPU 可用 → 复用现有版本，实测 spconv-cu126、PyG pt28cu128 wheel、pointops/pointgroup_ops/pointrope 源码编译均与之兼容。
4. **LitePT 提示 pointrope 缺失**：`[PointROPE] CUDA implementation unavailable ... Using slower Pytorch fallback`，编译 libs/pointrope 后消除。
5. **flash-attn 代理下载中断**：ghfast.top 仅 55kB/s，256MB wheel 下载被截断报 "Wheel is invalid" → 测速对比后换 ghfile.geekertao.top（约 3MB/s）下载完整；pip 安装本地 wheel 时文件名必须保留官方命名（含 cu12torch2.8cxx11abiTRUE-cp312 标签）。
6. **open3d 缺系统库**：① `libEGL.so.1` 缺失 → apt 装 libegl1/libgl1；② conda libstdc++ 缺 `GLIBCXX_3.4.30` → conda 清华 conda-forge 镜像装新版 libstdcxx-ng。

## 网络加速说明（重要）

本机自带 GitHub/HuggingFace 加速代理，用环境变量开关：

- **需要访问 GitHub/HuggingFace 时**（clone 代码、下权重等）：先运行 `source /etc/network_turbo`（会设置 `http_proxy/https_proxy=http://172.20.0.113:12798`，并对 modelscope/aliyuncs/tencentyun/wisemodel 等国内站点走 no_proxy 直连）。
- **访问国内镜像时**（pip/conda 安装包等）：运行 `unset http_proxy && unset https_proxy` 关掉加速，否则走代理访问国内镜像**会更慢**。
- 若代理不可用，备选 GitHub 代理镜像：`https://ghfast.top/`、`https://gh-proxy.com/`、`https://ghfile.geekertao.top/`（直接在 GitHub URL 前加前缀，实测 ghfile.geekertao.top 速度最快约 3MB/s）。
