import torch
import torch.nn as nn
from PIL import Image
import yaml
import argparse
from torchvision import transforms
from diffusion_trainer import Diffusion, p_sample_loop 
from model import ConditionalModel
from pretraining.dcg import DCG
import os
import re
from ml_collections import ConfigDict

''' 使用: python inference.py \
            --config exp/logs/aptos/split_0/config.yml \
            --ckpt exp/logs/aptos/split_0/ckpt_best.pth \
            --aux_ckpt exp/logs/aptos/split_0/aux_ckpt_best.pth \  
            --image your_test_image.jpg \
            --device cuda
            
            这里用哪个数据集就写哪个aptos或者isic
'''
def load_config(config_path):
    """加载配置文件并转换为ConfigDict格式"""
    with open(config_path, 'r') as f:
        # 读取文件内容
        content = f.read()
        
        # 移除Python对象标签
        content = re.sub(r'!!python/object:[\w\.]+', '', content)
        
        try:
            # 解析处理后的YAML内容
            config = yaml.safe_load(content)
            
            # 转换为ConfigDict格式
            def dict_to_config(d):
                if isinstance(d, dict):
                    return ConfigDict({k: dict_to_config(v) for k, v in d.items()})
                elif isinstance(d, list):
                    return [dict_to_config(v) for v in d]
                return d
            
            return dict_to_config(config)
            
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            raise

class DiffMICPredictor:
    def __init__(self, config_path, ckpt_path, aux_ckpt_path, device='cuda'):
        self.device = torch.device(device)
        self.config = load_config(config_path)
        self.config.device = device
        self.args = argparse.Namespace()
        self.args.device = device
        self.args.log_path = os.path.dirname(ckpt_path)
        self.diffusion = Diffusion(self.args, self.config)

        # 打印配置信息进行调试
        print("Config keys:", self.config.keys())
        if hasattr(self.config, 'data'):
            print("Data config keys:", self.config.data.keys())

        # 设置默认图像尺寸
        self.image_size = 224  # 默认值
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'image_size'):
            self.image_size = self.config.model.image_size
        elif hasattr(self.config, 'data') and hasattr(self.config.data, 'image_size'):
            self.image_size = self.config.data.image_size
        print(f"Using image size: {self.image_size}")

        
        # 初始化模型
        self.model = ConditionalModel(self.config, guidance=self.config.diffusion.include_guidance).to(self.device)
        self.model = self.model.to(self.device)

        self.cond_pred_model = DCG(self.config).to(self.device)
        self.cond_pred_model = self.cond_pred_model.to(self.device)
        # 加载模型参数
        states = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(states[0])
        
        aux_states = torch.load(aux_ckpt_path, map_location=self.device)
        self.cond_pred_model.load_state_dict(aux_states[0])
        
        # 设置为评估模式
        self.model.eval()
        self.cond_pred_model.eval()
        
        # 设置预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        对单张图片进行预测
        Args:
            image_path: 输入图片路径
        Returns:
            pred_label: 预测的类别
            confidence: 预测的置信度
        """
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取条件预测
            target_pred, _, _ = self.cond_pred_model(image)
            target_pred = target_pred.softmax(dim=1)
            
            # 设置先验分布
            y_T_mean = target_pred.to(self.device)
            if self.config.diffusion.noise_prior:
                y_T_mean = torch.zeros_like(target_pred).to(self.device)

            test_timesteps = self.config.diffusion.test_timesteps
            alphas = self.diffusion.alphas.to(self.device)
            one_minus_alphas_bar_sqrt = self.diffusion.one_minus_alphas_bar_sqrt.to(self.device)
            
            
            # 进行采样
            label_t_0 = p_sample_loop(
                self.model, 
                image, 
                target_pred.to(self.device), 
                y_T_mean,
                test_timesteps,
                alphas,
                one_minus_alphas_bar_sqrt,
                only_last_sample=True
            )
            
            # 获取预测结果
            pred_probs = label_t_0.softmax(dim=-1)
            pred_label = torch.argmax(pred_probs, dim=1)
            confidence = torch.max(pred_probs, dim=1)[0]
            
            return pred_label.item(), confidence.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--aux_ckpt', type=str, required=True, help='Path to auxiliary model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # 初始化预测器
    predictor = DiffMICPredictor(
        config_path=args.config,
        ckpt_path=args.ckpt,
        aux_ckpt_path=args.aux_ckpt,
        device=args.device
    )

    # 进行预测
    label, confidence = predictor.predict(args.image)
    
    # 输出结果
    print(f'Predicted class: {label}')
    print(f'Confidence: {confidence:.4f}')

if __name__ == '__main__':
    main()
