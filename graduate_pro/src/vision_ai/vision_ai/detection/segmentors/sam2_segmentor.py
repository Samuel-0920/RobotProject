# detection/segmentors/sam2_segmentor.py
import logging
import os
import numpy as np
import torch
from typing import List
from ..interfaces.segmentor_interface import ObjectSegmentor


def _sam2_hydra_config_name(config_name: str) -> str:
    """
    Map JSON values like sam2_hiera_l.yaml to Hydra names under sam2/configs/
    (e.g. sam2/sam2_hiera_l). Root-level *.yaml stubs in the sam2 package break
    compose() on Hydra 1.3+.
    """
    name = config_name.strip()
    for suf in (".yaml", ".yml"):
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
            break
    if "/" in name:
        return name
    if name.startswith("sam2.1_"):
        return f"sam2.1/{name}"
    return f"sam2/{name}"


def _build_sam2_hydra13(
    config_name: str,
    ckpt_path: str,
    device: str,
    apply_postprocessing: bool = True,
):
    import sam2
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    hydra_overrides = []
    if apply_postprocessing:
        hydra_overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    cfg_key = _sam2_hydra_config_name(config_name)
    config_dir = os.path.join(sam2.__path__[0], "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=cfg_key, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error("%s", missing_keys)
            raise RuntimeError(f"SAM2 checkpoint missing keys: {missing_keys}")
        if unexpected_keys:
            logging.error("%s", unexpected_keys)
            raise RuntimeError(f"SAM2 checkpoint unexpected keys: {unexpected_keys}")
        logging.info("Loaded SAM2 checkpoint successfully")
    model = model.to(device)
    model.eval()
    return model


class SAM2Segmentor(ObjectSegmentor):
    """SAM2分割器实现"""
    
    def __init__(self, checkpoint_path: str, config_name: str = "sam2_hiera_l.yaml", device: str = "cuda"):
        """
        初始化SAM2分割器
        
        Args:
            checkpoint_path: SAM2模型检查点路径
            config_name: SAM2配置文件名称 (例如: "sam2_hiera_l.yaml")
            device: 设备 ("cuda" 或 "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Hydra 1.3+: 勿用 sam2.build_sam.build_sam2 + 包内根目录 *.yaml 桩文件（compose 会得到无效 cfg）
            self.model = _build_sam2_hydra13(
                config_name, checkpoint_path, device=self.device, apply_postprocessing=True
            )
            self.predictor = SAM2ImagePredictor(self.model)
            
            print(f"[SAM2] Model loaded successfully")
            print(f"[SAM2] Configuration: {config_name}")
            print(f"[SAM2] Checkpoint: {checkpoint_path}")
            
        except ImportError as e:
            raise ImportError(
                "无法导入SAM2库。请确保已安装：\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            ) from e
        except Exception as e:
            raise RuntimeError(f"SAM2模型加载失败: {e}") from e
    
    def set_image(self, image: np.ndarray):
        """
        设置当前处理的图像
        
        Args:
            image: RGB图像 (H, W, 3)
        """
        self.predictor.set_image(image)
    
    def segment(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        对检测到的目标进行分割
        
        Args:
            image: RGB图像 (H, W, 3)
            boxes: 检测框 (N, 4) [x1, y1, x2, y2]
            
        Returns:
            masks: 分割掩码列表，每个掩码为 (H, W) 的boolean数组
        """
        if len(boxes) == 0:
            return []
        
        # 设置图像
        self.set_image(image)
        
        masks = []
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = box.astype(int)
                input_box = np.array([x1, y1, x2, y2])
                
                # 运行SAM2预测
                mask, _, _ = self.predictor.predict(
                    box=input_box, 
                    multimask_output=False
                )
                
                # 添加到结果列表
                masks.append(mask[0])  # mask[0] 是最佳掩码
                
            except Exception as e:
                print(f"[SAM2] Failed to segment target {i}: {e}")
                empty_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                masks.append(empty_mask)
        return masks
    
    def segment_single(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        分割单个目标 (便利方法)
        
        Args:
            image: RGB图像 (H, W, 3)
            box: 单个检测框 (4,) [x1, y1, x2, y2]
            
        Returns:
            mask: 分割掩码 (H, W) boolean数组
        """
        masks = self.segment(image, box.reshape(1, -1))
        return masks[0] if masks else np.zeros((image.shape[0], image.shape[1]), dtype=bool)