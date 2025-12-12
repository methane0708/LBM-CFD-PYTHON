import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
import torch

class LBM_GUI:
    def __init__(self, root):
        self.root = root
        root.title("LBM GUI")
        root.geometry("500x800")
        self.build_gui()

    def build_gui(self):# 建立 GUI 元件
        title = ttk.Label(self.root, text="LBM Simulation Config", font=("Arial", 18))
        title.pack(pady=10)
        main = ttk.Frame(self.root)
        main.pack(pady=5)
        ttk.Label(main, text="流體種類").pack()
        self.fluid_var = tk.StringVar(value="Water")
        fluids = ["Water", "Air", "Honey", "Blood"]
        ttk.OptionMenu(main, self.fluid_var, fluids[0], *fluids).pack(pady=3)
        self.real_u_var = tk.StringVar(value="5.0")
        self.add_entry(main, "流速 (m/s)", self.real_u_var)
        self.irr = tk.StringVar(value="100000")
        self.add_entry(main, "迭迨次數(五萬次=1秒)", self.irr)
        ttk.Label(main, text="遮罩來源").pack(pady=4)
        self.mask_mode = tk.StringVar(value="upload")

        mode_frame = ttk.Frame(main)
        mode_frame.pack()

        ttk.Radiobutton(mode_frame, text="上傳圖片", variable=self.mask_mode, value="upload", 
                        command=self.update_mask_ui).pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="幾何生成", variable=self.mask_mode, value="shape",
                        command=self.update_mask_ui).pack(side="left", padx=10)
        self.upload_frame = ttk.Frame(main)
        self.upload_frame.pack(pady=5)
        self.mask_path = tk.StringVar(value="None")
        entry = ttk.Entry(self.upload_frame, textvariable=self.mask_path, width=32)
        entry.pack(side="left", padx=3)
        ttk.Button(self.upload_frame, text="選擇檔案", command=self.browse_image).pack(side="left")
        self.shape_frame = ttk.Frame(main)
        ttk.Label(self.shape_frame, text="形狀類型").pack()
        self.shape_type = tk.StringVar(value="Ellipse")

        shape_menu = ttk.OptionMenu(self.shape_frame, self.shape_type, "Ellipse",
                                    "Ellipse", "NACA Airfoil",
                                    command=lambda e: self.update_shape_params())
        shape_menu.pack(pady=5)
        ttk.Label(main, text="邊界").pack()  
        bd = ["上下界", "上界", "下界", "無界"]
        self.bd_var = tk.StringVar(value="上下界")
        ttk.OptionMenu(main, self.bd_var, bd[0], *bd).pack(pady=6)
        
        self.param_frame = ttk.Frame(self.shape_frame)
        self.param_frame.pack()
        self.update_shape_params()
        self.output_var = tk.StringVar(value="cfd")
        self.add_entry(main, "輸出檔案名稱", self.output_var)
        ttk.Button(self.root, text="預覽遮罩 ",
                   command=self.preview).pack(pady=20)
        ttk.Button(self.root, text="開始模擬",
                   command=self.start_simulation).pack(pady=20)


    def add_entry(self, parent, text, var):
        ttk.Label(parent, text=text).pack()
        ttk.Entry(parent, textvariable=var).pack(pady=3)
    def update_mask_ui(self):# 更新遮罩 UI
        mode = self.mask_mode.get()
        if mode == "upload":
            self.shape_frame.pack_forget()
            self.upload_frame.pack(pady=5)
        else:
            self.upload_frame.pack_forget()
            self.shape_frame.pack(pady=5)
    def browse_image(self):# 瀏覽圖片檔案
        filename = filedialog.askopenfilename(
            title="選擇遮罩圖片",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if filename:
            self.mask_path.set(filename)
    def update_shape_params(self):# 更新形狀參數 UI
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        shape = self.shape_type.get()

        if shape == "Ellipse":
            self.ellipse_a = tk.StringVar(value="150")
            self.ellipse_b = tk.StringVar(value="375")
            self.ellipse_angle = tk.StringVar(value="0.0")

            self.add_entry(self.param_frame, "長軸 a (mm)，最大75cm", self.ellipse_a)
            self.add_entry(self.param_frame, "短軸 b (mm)，最大75cm", self.ellipse_b)
            self.add_entry(self.param_frame, "攻角 (deg)", self.ellipse_angle)

        else:  # NACA
            ttk.Label(self.param_frame, text="NACA 對稱翼型").pack()
            self.naca_type = tk.StringVar(value="0008")

            ttk.OptionMenu(self.param_frame, self.naca_type,
                           "0008", "0008", "0012", "0015", "0018", "0020").pack(pady=5)

            self.chord = tk.StringVar(value="100")
            self.aoa = tk.StringVar(value="0.0")

            self.add_entry(self.param_frame, "弦長 (mm)，最大150cm", self.chord)
            self.add_entry(self.param_frame, "攻角 (deg)", self.aoa)

    def start_simulation(self):# 開始模擬，將參數整理輸出
        info = {
            "Fluid": self.fluid_var.get(),
            "Velocity": float(self.real_u_var.get()),
            "Output File": self.output_var.get(),
            "Iterations": int(self.irr.get()),
            'bd_condition': self.bd_var.get(),
        }
        if self.mask_mode.get() == "upload":
            info["Mask Mode"] = "Image Upload"
            info["Mask Path"] = None if self.mask_path.get() == "None" else self.mask_path.get()

        else:
            info["Mask Mode"] = "Generated Shape"
            shape = self.shape_type.get()
            info["Shape Type"] = shape

            if shape == "Ellipse":
                info["Ellipse a"] = float(self.ellipse_a.get())
                info["Ellipse b"] = float(self.ellipse_b.get())
                info["Ellipse AOA"] = float(self.ellipse_angle.get())

            else:
                info["NACA Type"] = self.naca_type.get()
                info["Chord"] = float(self.chord.get())
                info["AOA"] = float(self.aoa.get())
        from CFD_mask import mask_building
        mode = self.mask_mode.get()
        if mode == "upload":
            path = self.mask_path.get()
            mask = mask_building("upload", path)
        else:
            shape = self.shape_type.get()
            if shape == "Ellipse":
                    params = {
                        'a': float(self.ellipse_a.get()) //2, 
                        'b': float(self.ellipse_b.get()) //2,
                        'angle': float(self.ellipse_angle.get())
                    }
                    mask = mask_building("ellipse", params)
            else:
                    params = {
                        'naca': self.naca_type.get(),
                        'chord': float(self.chord.get()),
                        'angle': float(self.aoa.get())
                    }
                    mask = mask_building("NACA", params)
        self.info = info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        expanded_tensor = torch.zeros((750, 3000), dtype=torch.bool, device=device)
        start_row = 375
        end_row = start_row + 750
        expanded_tensor[:,start_row:end_row] = mask
        self.mask = expanded_tensor
        self.root.destroy()
    def preview(self):# 預覽遮罩
            from CFD_mask import mask_building
            mode = self.mask_mode.get()
            if mode == "upload":
                path = self.mask_path.get()
                mask = mask_building("upload", path)
            else:
                shape = self.shape_type.get()
                if shape == "Ellipse":
                    params = {
                        'a': float(self.ellipse_a.get())//2 ,  
                        'b': float(self.ellipse_b.get())//2,
                        'angle': float(self.ellipse_angle.get()) 
                    }
                    mask = mask_building("ellipse", params)
                else:
                    params = {
                        'naca': self.naca_type.get(),
                        'chord': float(self.chord.get()),
                        'angle': float(self.aoa.get())
                    }
                    mask = mask_building("NACA", params)
            plt.figure(figsize=(4, 4))
            mask = mask.cpu().numpy()
            plt.title("Boolean Tensor Preview")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.show()
def launch_gui():  #整理本程式內容，方便主程式呼叫  
    root = tk.Tk()
    app = LBM_GUI(root)
    root.mainloop()
    return app.info, app.mask
