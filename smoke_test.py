# Pointcept 安装验证冒烟测试
# 用真实 config 构建模型,随机点云在 GPU 上跑一次训练步(前向 + 反向)
import importlib.util

import torch
from pointcept.models import build_model
from pointcept.utils.config import Config


def make_batch(num_points=8192, in_channels=6, num_classes=20, device="cuda"):
    coord = torch.rand(num_points, 3, device=device) * 5.0
    feat = torch.rand(num_points, in_channels, device=device)
    segment = torch.randint(0, num_classes, (num_points,), device=device)
    # 两个场景拼成一个 batch
    half = num_points // 2
    offset = torch.tensor([half, num_points], dtype=torch.long, device=device)
    grid_size = 0.02
    grid_coord = torch.div(
        coord - coord.min(0)[0], grid_size, rounding_mode="trunc"
    ).int()
    return dict(
        coord=coord,
        feat=feat,
        segment=segment,
        offset=offset,
        grid_size=grid_size,
        grid_coord=grid_coord,
    )


def run_smoke(cfg_path, model_name):
    cfg = Config.fromfile(cfg_path)
    model_cfg = cfg.model
    # flash-attn 未安装时,PTv3 退回非 flash 路径
    try:
        import flash_attn  # noqa: F401

        flash_ok = True
    except ImportError:
        flash_ok = False

    def tune(d):
        if isinstance(d, dict):
            if "enable_flash" in d and not flash_ok:
                d["enable_flash"] = False
            for v in d.values():
                tune(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                tune(v)

    tune(model_cfg)
    model = build_model(model_cfg).cuda()
    model.train()
    batch = make_batch(in_channels=model_cfg.backbone.in_channels)
    out = model(batch)
    loss = out["loss"]
    loss.backward()
    torch.cuda.synchronize()
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[PASS] {model_name}: forward+backward OK, loss={loss.item():.4f}, "
        f"params={n_param:.1f}M, flash_attn={'on' if flash_ok else 'off'}"
    )


if __name__ == "__main__":
    run_smoke("configs/scannet/semseg-spunet-v1m1-0-base.py", "SpUNet(ScanNet config)")
    run_smoke("configs/scannet/semseg-pt-v3m1-0-base.py", "PTv3(ScanNet config)")
    print("ALL SMOKE TESTS PASSED")
