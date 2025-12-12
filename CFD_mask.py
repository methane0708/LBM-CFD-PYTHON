import torch
import numpy as np
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
import math


IMG_SIZE = 750  # 統一矩陣大小

def get_coordinates(size, device):# 取得座標矩陣
    y = torch.arange(size, device=device, dtype=torch.float32)
    x = torch.arange(size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    cy, cx = size // 2, size // 2
    y_shifted = grid_y - cy
    x_shifted = grid_x - cx
    return x_shifted, y_shifted

def rotate_coordinates(x, y, angle_deg, device):# 旋轉座標矩陣
    theta = torch.tensor(math.radians(angle_deg), device=device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    return x_rot, y_rot
def create_ellipse_tensor(angle_deg, a, b):# 建立橢圓遮罩矩陣
    x, y = get_coordinates(IMG_SIZE, device)
    x_rot, y_rot = rotate_coordinates(x, y, angle_deg, device)
    ellipse_eq = (x_rot / a)**2 + (y_rot / b)**2
    bool_tensor = ellipse_eq <= 1
    return bool_tensor
def create_naca_tensor(angle_deg, chord, thickness):# 建立 NACA 遮罩矩陣
    x  , y = get_coordinates(IMG_SIZE, device)
    x_rot, y_rot = rotate_coordinates(x, y, angle_deg, device)
    x_norm = (x_rot + chord / 2) / chord
    valid_mask = (x_norm >= 0) & (x_norm <= 1)
    x_safe = torch.where(valid_mask, x_norm, torch.zeros_like(x_norm))
    term1 =  0.2969 * torch.sqrt(x_safe)
    term2 = -0.1260 * x_safe
    term3 = -0.3516 * (x_safe ** 2)
    term4 =  0.2843 * (x_safe ** 3)
    term5 = -0.1015 * (x_safe ** 4)
    half_thk = 5 * thickness * (term1 + term2 + term3 + term4 + term5) * chord
    bool_tensor = valid_mask & (torch.abs(y_rot) <= half_thk)
    return bool_tensor

def mask_building(type,param):# 主遮罩建立函式
    if type=="upload":
        mask_path=param
        img = Image.open(mask_path).convert('L')# 開啟檔案並轉為灰階
        img = img.resize((750, 750))# 調整大小         
        tensor = np.array(img)              
        tensor = tensor < 127     # 將像素值轉為布林值                      
        mask_tensor=torch.from_numpy(tensor).to(device).double()
        return mask_tensor
    elif type=="ellipse":
        a = param['a']
        b = param['b']
        angle_deg = param['angle']
        mask_tensor = create_ellipse_tensor(angle_deg, a, b).to(device)
        return mask_tensor
    elif type=="NACA":
        naca_code=int(param['naca'])/100
        chord=param['chord']//2
        angle_deg = param['angle']
        thickness = naca_code
        mask_tensor = create_naca_tensor(angle_deg, chord, thickness).to(device)
        return mask_tensor
        
        