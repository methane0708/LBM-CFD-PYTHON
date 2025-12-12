import tkinter as tk
from tkinter import ttk
import threading
import time
from CFD_gui import launch_gui
from CFD_MAIN import cfd_setup
from CFD_to_paraview import cfd_to_paraview

class ProgressWindow(tk.Tk):
    def __init__(self, sim_params, mask):
        #匯入參數
        super().__init__()
        self.params = sim_params
        self.mask = mask
        self.fluid_v = sim_params['Velocity']
        self.fluid_type = sim_params['Fluid']
        self.irr = sim_params['Iterations']
        self.name = sim_params['Output File']
        bd_raw = sim_params['bd_condition']
        if bd_raw == "上下界": self.bd_condition = [1, 1]
        elif bd_raw == "上界": self.bd_condition = [1, 0]
        elif bd_raw == "下界": self.bd_condition = [0, 1]
        else: self.bd_condition = [0, 0]
        #運行進度條
        self.title("CFD 任務執行中")
        self.geometry("450x250")
        self.lbl_title = tk.Label(self, text="任務執行中", font=("微軟正黑體", 16, "bold"))
        self.lbl_title.pack(pady=15)
        self.status_var = tk.StringVar(value="正在初始化...")
        self.lbl_status = tk.Label(self, textvariable=self.status_var, font=("微軟正黑體", 12))
        self.lbl_status.pack(pady=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=350, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=20)
        self.detail_var = tk.StringVar(value="")
        self.lbl_detail = tk.Label(self, textvariable=self.detail_var, fg="gray")
        self.lbl_detail.pack(pady=5)
        self.btn_close = tk.Button(self, text="關閉", command=self.destroy, state="disabled")
        self.btn_close.pack(pady=10)
        self.start_thread()
    def start_thread(self):
        t = threading.Thread(target=self.run_process)
        t.daemon = True
        t.start()
    def update_ui(self, current, total, stage_name):
        if total > 0:
            percent = (current / total) * 100
        else:
            percent = 0
        self.after(0, lambda: self._update_widgets(percent, stage_name, current, total))
    def _update_widgets(self, percent, stage_name, current, total):
        self.progress_var.set(percent)
        self.status_var.set(f"{stage_name} ({percent:.1f}%)")
        elapsed= time.time() - start_time
        fps = current / elapsed
        self.detail_var.set(f"進度: {current} / {total} | Speed: {fps:.2f} steps/s | Elapsed: {elapsed:.0f} s")
    def run_process(self):
            global start_time 
            start_time = time.time()# 記錄開始時間
            # 執行 CFD 設定與模擬
            cfd_setup(
                self.fluid_v, 
                self.fluid_type, 
                self.irr, 
                self.bd_condition, 
                self.mask, 
                self.name,
                progress_callback=self.update_ui
            )
            # 執行轉檔至 ParaView
            start_time = time.time()# 記錄開始時間
            elapsed = time.time() - start_time
            cfd_to_paraview(
                self.fluid_v, 
                self.fluid_type, 
                self.irr, 
                self.bd_condition, 
                self.mask, 
                self.name,
                progress_callback=self.update_ui 
            )
            self.after(0, lambda: self.status_var.set("所有任務完成！用 ParaView 開啟資料夾"))
            self.after(0, lambda: self.progress_var.set(100))
            self.after(0, lambda: self.btn_close.config(state="normal", bg="#ddffdd", text="完成 (點擊關閉)"))
            
info, mask = launch_gui()# 啟動 GUI 並獲取參數與遮罩
app = ProgressWindow(info, mask)# 啟動進度視窗
app.mainloop()# 啟動主事件迴圈
