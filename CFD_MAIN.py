import torch
import numpy as np
import h5py
import os
import time

def cfd_setup(fluid_v,fluid_type,irr,bd_condition,mask,name,progress_callback=None):
    #設定預設型態與裝置
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NX, NY = 3000, 750
    #匯入模擬參數
    FLUID_TYPE = fluid_type 
    REAL_VELOCITY = fluid_v
    OUTPUT_FILE = f"{name}.h5"
    SAVE_INTERVAL = 1000
    step_limit=irr
    FLUID_PROPERTIES = {
        'Water': {'rho': 1000.0,  'nu': 1.0e-6},
        'Air':   {'rho': 1.225,   'nu': 1.5e-5},
        'Honey': {'rho': 1420.0,  'nu': 1.0e-4},
        'Blood': {'rho': 1060.0,  'nu': 3.5e-6},
    }
    REAL_NU = FLUID_PROPERTIES[FLUID_TYPE]['nu']
    REAL_RHO = FLUID_PROPERTIES[FLUID_TYPE]['rho']
    dx_real = 0.002# 真實空間格子間距mm
    U_LBM_MAX = 0.01*REAL_VELOCITY
    dt_real = 0.00002# LBM 真實時間s/步
    nu_lbm = REAL_NU * dt_real / (dx_real**2)# LBM 黏度
    tau_lbm = 3.0 * nu_lbm + 0.5# LBM 鬆弛時間
    #建立阻尼區域(sponge layer)，模擬外牆吸收流體動能
    sponge_width = NX // 15 # 200 個格子寬度
    sponge_sigma = 0.6# 阻尼強度
    #初始化 LBM 變數
    tau_field = torch.full((NY, NX), tau_lbm, dtype=torch.float64, device=device)
    y_grid, x_grid = torch.meshgrid(torch.arange(NY, device=device), 
                                    torch.arange(NX, device=device), indexing='ij')
    is_sponge = x_grid > (NX - sponge_width)
    dist_from_start = (x_grid - (NX - sponge_width)).float()
    profile = (dist_from_start / sponge_width) ** 2
    target_tau = tau_lbm + sponge_sigma * profile
    tau_field = torch.where(is_sponge, target_tau, tau_field)
    Re_domain = (REAL_VELOCITY * 6) / REAL_NU
    C_smago = 0.18
    cs_sq = 1.0 / 3.0
    w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)# LBM 權重
    c_np = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
                    [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)# LBM 速度集

    w = torch.from_numpy(w_np).to(device).view(9, 1, 1)# LBM 權重張量
    cx = torch.tensor(c_np[:, 0], dtype=torch.float64, device=device).view(9, 1, 1)# LBM 速度張量
    cy = torch.tensor(c_np[:, 1], dtype=torch.float64, device=device).view(9, 1, 1)# LBM 速度張量
    cx_bc = cx.view(9, 1)
    cy_bc = cy.view(9, 1)
    w_bc  = w.view(9, 1)
    opp_idx = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=device)
    #邊界反彈索引
    reflect_top_idx = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.long, device=device)
    reflect_bot_idx = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.long, device=device)
    f = torch.zeros((9, NY, NX), dtype=torch.float64, device=device)
    rho = torch.ones((NY, NX), dtype=torch.float64, device=device)
    ux = torch.zeros((NY, NX), dtype=torch.float64, device=device)
    uy = torch.zeros((NY, NX), dtype=torch.float64, device=device)
    obstacle_mask = mask
    def get_equilibrium(rho, ux, uy):# 計算平衡分佈函式
        u2 = ux**2 + uy**2
        eu = cx * ux + cy * uy
        return w * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u2)
    f = get_equilibrium(rho, ux, uy)
    if os.path.exists(OUTPUT_FILE):# 刪除舊檔案
        os.remove(OUTPUT_FILE)
    with h5py.File(OUTPUT_FILE, 'w') as f_h5:# 建立新檔案並寫入屬性
        f_h5.attrs['NX'] = NX; f_h5.attrs['NY'] = NY
        f_h5.attrs['Real_U'] = REAL_VELOCITY
        f_h5.attrs['dx_real'] = dx_real 
    step_counter = 0
    def save():# 儲存當前狀態至 HDF5 檔案
        ux_cpu = ux.cpu().numpy().astype(np.float32)
        uy_cpu = uy.cpu().numpy().astype(np.float32)
        rho_fluc = (rho - 1.0).cpu().numpy().astype(np.float32)                
        duy_dx = (torch.roll(uy, -1, dims=1) - torch.roll(uy, 1, dims=1)) / 2.0
        dux_dy = (torch.roll(ux, -1, dims=0) - torch.roll(ux, 1, dims=0)) / 2.0
        curl = (duy_dx - dux_dy).cpu().numpy().astype(np.float32)
        curl[obstacle_mask.cpu().numpy()] = 0
        with h5py.File(OUTPUT_FILE, 'a') as f_h5:
            grp_name = f"step_{step_counter:06d}"
            grp = f_h5.create_group(grp_name)
            grp.create_dataset('ux', data=ux_cpu, compression='lzf')
            grp.create_dataset('uy', data=uy_cpu, compression='lzf')
            grp.create_dataset('rho_fluc', data=rho_fluc, compression='lzf')
            grp.create_dataset('curl', data=curl, compression='lzf')
        return
    save()# 儲存初始狀態
    while True:
        RAMP_STEPS = 2000# 入口速度 RAMP 步數，緩慢加速以防流速設定太快而壞掉
        if step_counter < RAMP_STEPS:
            current_u_inlet = U_LBM_MAX * (step_counter / RAMP_STEPS)
        else:
            current_u_inlet = U_LBM_MAX
        for _ in range(100):# 每次大迴圈執行20步，提高運算效率
            # 計算宏觀變數
                rho = f.sum(dim=0)
                rho_inv = 1.0 / rho
                ux = (f * cx).sum(dim=0) * rho_inv
                uy = (f * cy).sum(dim=0) * rho_inv
                f_eq = get_equilibrium(rho, ux, uy)
                f_neq = f - f_eq               # 計算非平衡分佈函式
                Pkx = (cx * cx * f_neq).sum(dim=0)# 計算應力張量
                Pky = (cy * cy * f_neq).sum(dim=0)
                Pkxy = (cx * cy * f_neq).sum(dim=0)
                Pi_norm_sq = Pkx**2 + Pky**2 + 2.0 * Pkxy**2# 計算應力張量範數平方
                Q = torch.sqrt(Pi_norm_sq) / (2.0 * rho * cs_sq * tau_field)  # 計算 Q 張量
                # Smagorinsky 模型調整鬆弛時間
                prefactor = 18.0 * (C_smago**2)# Smagorinsky 預因子
                tau_eff = tau_field + 0.5 * (torch.sqrt(tau_field**2 + prefactor * Q) - tau_field)  # 有效鬆弛時間             
                omega_eff = 1.0 / tau_eff# 有效鬆弛頻率
                f = (1.0 - omega_eff) * f + omega_eff * f_eq# 碰撞步驟
                # 反彈邊界條件
                f_boundary = f[:, obstacle_mask]
                f[:, obstacle_mask] = f_boundary[opp_idx]
                # 平移步驟
                for i in range(9):
                    f[i] = torch.roll(f[i], shifts=(int(c_np[i, 1]), int(c_np[i, 0])), dims=(0, 1))
                # 邊界條件處理
                if bd_condition[0] == True:
                    f[:, 0, :] = f[reflect_top_idx, 0, :]
                if bd_condition[1] == True:
                    f[:, -1, :] = f[reflect_bot_idx, -1, :]
                # 入口速度邊界條件
                rho_in = 1.0
                ux_in_val = torch.full((1, NY), current_u_inlet, device=device)
                uy_in_val = torch.zeros((1, NY), device=device)
                u2_in = ux_in_val**2 + uy_in_val**2
                eu_in = cx_bc * ux_in_val + cy_bc * uy_in_val
                f_eq_in = w_bc * rho_in * (1.0 + 3.0*eu_in + 4.5*eu_in**2 - 1.5*u2_in)
                f[:, :, 0] = f_eq_in
                f[:, :, -1] = f[:, :, -2]# 開放式出口邊界條件
                if progress_callback:# 更新進度
                    progress_callback(step_counter+1, step_limit, "LBM計算中")            
        step_counter += 100
            
        if RAMP_STEPS==step_counter:# RAMP 結束強制儲存
            save()          
        elif step_counter % SAVE_INTERVAL == 0 and step_counter>RAMP_STEPS:# 定期儲存
            save()
            if step_counter>=step_limit:# 達到步數上限結束模擬
                break    
