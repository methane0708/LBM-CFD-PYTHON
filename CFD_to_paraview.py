import h5py
import numpy as np
import os
import shutil
from pyevtk.hl import imageToVTK
import time
def cfd_to_paraview(fluid_v,fluid_type,irr,bd_condition,mask,name, progress_callback=None):
    #匯入模擬參數
    name=str(name)
    H5_INPUT = f"{name}.h5"
    OUTPUT_DIR = f"{name}"
    FLUID_PROPERTIES = {
            'Water': {'rho': 1000.0,  'nu': 1.0e-6},
            'Air':   {'rho': 1.225,   'nu': 1.5e-5},
            'Honey': {'rho': 1420.0,  'nu': 1.0e-4},
            'Blood': {'rho': 1060.0,  'nu': 3.5e-6},
        }
    FLUID_DENSITY=FLUID_PROPERTIES[fluid_type]['rho']
    LBM_MAX_U = 0.01 *fluid_v
    def calculate_derived_physics(ux, uy, rho_fluc, dx, C_u, C_p, C_t):# 計算衍生物理量
        u_phys = ux * C_u# 將 LBM 速度轉換為實際速度
        v_phys = uy * C_u# 將 LBM 速度轉換為實際速度
        w_phys = np.zeros_like(u_phys)# Z 方向速度為零
        Cs_sq = 1.0 / 3.0# LBM 聲速平方
        p_phys = (rho_fluc * Cs_sq * C_p)# 將 LBM 密度擾動轉換為實際壓力
        grad_u_y, grad_u_x = np.gradient(u_phys, dx)# 計算速度梯度
        grad_v_y, grad_v_x = np.gradient(v_phys, dx)# 計算速度梯度
        vorticity = grad_v_x - grad_u_y# 計算渦度
        S11 = grad_u_x# 計算應變率張量分量
        S22 = grad_v_y
        S12 = 0.5 * (grad_u_y + grad_v_x)
        norm_S_sq = S11**2 + S22**2 + 2 * S12**2
        norm_Omega_sq = 0.5 * vorticity**2
        q_criterion = 0.5 * (norm_Omega_sq - norm_S_sq)# 計算 Q 準則
        vel_mag = np.sqrt(u_phys**2 + v_phys**2)# 計算速度大小
        return {
            "Velocity": (u_phys, v_phys, w_phys),
            "Pressure_Pa": p_phys,
            "Vorticity": vorticity,
            "Q_Criterion": q_criterion,
            "Speed_m_s": vel_mag
        }
    def convert():
        if os.path.exists(OUTPUT_DIR):# 清理舊資料夾
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)# 建立新資料夾
        with h5py.File(H5_INPUT, 'r') as f:
            NX = f.attrs['NX']
            NY = f.attrs['NY']
            try:
                REAL_U = f.attrs['Real_U']
                dx_real = f.attrs['dx_real']
            except KeyError:# 若無此屬性，使用預設值
                REAL_U = 5.0
                dx_real = 0.002
            C_u = REAL_U / LBM_MAX_U
            C_t = dx_real / C_u
            C_p = FLUID_DENSITY * (C_u ** 2)
            steps = sorted(list(f.keys()))# 取得所有模擬步數
            for i, step_name in enumerate(steps):# 逐步轉換並儲存
                if progress_callback:
                    progress_callback(i+1, len(steps), "正在轉換格式")
                ux = np.flip(f[step_name]['ux'][:], axis=0)
                uy = -np.flip(f[step_name]['uy'][:], axis=0)
                rho_fluc = np.flip(f[step_name]['rho_fluc'][:], axis=0)
                data = calculate_derived_physics(ux, uy, rho_fluc, dx_real, C_u, C_p, C_t)
                point_data = {}
                for key, val in data.items():
                    if isinstance(val, tuple):
                        vx = np.ascontiguousarray(val[0].T.reshape(NX, NY, 1)).astype(np.float32)
                        vy = np.ascontiguousarray(val[1].T.reshape(NX, NY, 1)).astype(np.float32)
                        vz = np.ascontiguousarray(val[2].T.reshape(NX, NY, 1)).astype(np.float32)
                        point_data[key] = (vx, vy, vz)
                    else:
                        point_data[key] = np.ascontiguousarray(val.T.reshape(NX, NY, 1)).astype(np.float32)
                file_path = os.path.join(OUTPUT_DIR, f"sim_{i:06d}")                
                imageToVTK(
                    file_path,
                    origin=(0.0, 0.0, 0.0),
                    spacing=(dx_real, dx_real, dx_real),
                    pointData=point_data
                )
    convert()# 執行轉換