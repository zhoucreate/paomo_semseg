#### Paomo Dataset Preprocessing for Pointcept
将自定义泡沫修边点数据集（`.ply`点云 + 修边索引`.txt`）预处理为 Pointcept 框架兼容的 ScanNet 格式，可直接使用`ScanNetDataset`加载训练。
Input Data Structure（输入数据集目录结构）
```bash
dataset_root/
├─ train/
│  ├─ sample01.ply        # 点云文件，必须包含xyz坐标；法向nx/ny/nz、RGB颜色为可选
│  └─ sample01.txt        # 修边点索引文件，一行一个点索引；允许空文件（无修边点）
└─ val/
   ├─ sample02.ply
   └─ sample02.txt
```
```bash
python /Pointcept-1.7.0/convert_to_pointcept.py \
    --dataset_root <path/to/paomo_box_dataset2> \
    --output_root data/paomo
```
#### Point Transformers
- **LitePT**

LitePT (CVPR 2026) is a state-of-the-art point cloud backbone that delivers superior or competitive performance with significantly improved efficiency compared to prior point Transformers. 

1. Additional requirements:

Compile CUDA implementation of PointROPE. Otherwise, the system will automatically fall back to a slower PyTorch implementation. 

```bash
cd libs/pointrope
python setup.py install
cd ../..
```

2. Example running scripts:

The model is registered as `LitePT-v1` and shared across small/base/large variants. In config filenames, `v1m1` refers to the lightweight decoder (no conv/attn), while `v1m2` uses a decoder with conv or attention at selected stages. See **Decoder design** in Sec 4.1 in the paper for details.

```bash
### paomo + LitePT-S
sh scripts/train.sh -g 4 -d paomo -c semseg-litept-v1m1-0-small -n semseg-litept-v1m1-0-small
```
