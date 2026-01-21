

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import PhotoImage
import threading
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, List


import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import VAE
from training.trainer import load_model
from interpolation.morphing import MorphingPipeline
from utils.helpers import get_device, get_latest_checkpoint, load_custom_image
from data.dataset import MNISTDataLoader


class VAEMorphingGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Morphing Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')  
        
        
        self.source_image_path = tk.StringVar()
        self.target_image_path = tk.StringVar()
        self.num_frames = tk.IntVar(value=15)
        self.model = None
        self.device = None
        self.morphing_pipeline = None
        
        
        self.generated_gif_path = None
        self.frame_images = []
        self.image_references = []  
        
        
        self.init_model()
        
        
        self.create_widgets()
        
    def init_model(self):
        
        try:
            self.device = get_device()
            
            
            model_path = get_latest_checkpoint('./checkpoints')
            if not model_path:
                messagebox.showwarning(
                    "Model Not Found", 
                    "No trained model found. Please train the model first using:\npython main.py --train --epochs 5"
                )
                return
                
            self.model = load_model(model_path, self.device)
            self.model.eval()
            
            
            self.morphing_pipeline = MorphingPipeline(self.model, self.device)
            
            print(f" Model loaded successfully from {model_path}")
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
            
    def create_widgets(self):
        
        
        title_label = tk.Label(
            self.root, 
            text="Digit Morphing Tool", 
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        
        left_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        
        right_panel = ttk.LabelFrame(main_frame, text="Results", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_panel)
        self.create_results_panel(right_panel)
        
    def create_control_panel(self, parent):
        
        
        
        source_frame = ttk.LabelFrame(parent, text="Source Image", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            source_frame, 
            text="Upload Source Image",
            command=self.upload_source_image,
            width=25
        ).pack(pady=5)
        
        self.source_preview_label = ttk.Label(source_frame, text="No image selected")
        self.source_preview_label.pack(pady=5)
        
        
        target_frame = ttk.LabelFrame(parent, text="Target Image", padding="10")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            target_frame, 
            text="Upload Target Image",
            command=self.upload_target_image,
            width=25
        ).pack(pady=5)
        
        self.target_preview_label = ttk.Label(target_frame, text="No image selected")
        self.target_preview_label.pack(pady=5)
        
        
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="Number of Frames:").pack(anchor=tk.W)
        
        frames_frame = ttk.Frame(params_frame)
        frames_frame.pack(fill=tk.X, pady=5)
        
        ttk.Scale(
            frames_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.num_frames,
            command=self.update_frames_label
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.frames_label = ttk.Label(frames_frame, text="15")
        self.frames_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        
        generate_frame = ttk.Frame(parent)
        generate_frame.pack(fill=tk.X, pady=20)
        
        self.generate_button = ttk.Button(
            generate_frame,
            text="Generate Morphing",
            command=self.generate_morphing,
            style='Accent.TButton'
        )
        self.generate_button.pack(fill=tk.X)
        
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            parent, 
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        
        self.status_label = ttk.Label(parent, text="Ready", foreground='green')
        self.status_label.pack(pady=5)
        
    def create_results_panel(self, parent):
        
        
        
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        
        gif_frame = ttk.Frame(notebook)
        notebook.add(gif_frame, text="🎬 Generated GIF")
        
        self.gif_label = ttk.Label(gif_frame, text="No GIF generated yet")
        self.gif_label.pack(expand=True, fill=tk.BOTH)
        
        
        frames_frame = ttk.Frame(notebook)
        notebook.add(frames_frame, text="🖼️ Frame Sequence")
        
        
        canvas = tk.Canvas(frames_frame)
        scrollbar = ttk.Scrollbar(frames_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def upload_source_image(self):
        
        file_path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.source_image_path.set(file_path)
            self.preview_image(file_path, self.source_preview_label, "Source")
            
    def upload_target_image(self):
        
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.target_image_path.set(file_path)
            self.preview_image(file_path, self.target_preview_label, "Target")
            
    def preview_image(self, file_path: str, label_widget, image_type: str):
        
        try:
            
            img = Image.open(file_path)
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            
            
            photo = ImageTk.PhotoImage(img)
            
            
            label_widget.configure(image=photo, text=f"{image_type}: {os.path.basename(file_path)}")
            label_widget.image = photo  
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load {image_type.lower()} image: {str(e)}")
            
    def update_frames_label(self, value):
        
        self.frames_label.configure(text=str(int(float(value))))
        
    def generate_morphing(self):
        
        if not self.source_image_path.get() or not self.target_image_path.get():
            messagebox.showwarning("Missing Images", "Please upload both source and target images.")
            return
            
        if not self.model:
            messagebox.showerror("Model Error", "Model not loaded. Please restart the application.")
            return
            
        
        self.generate_button.configure(state='disabled')
        self.progress_var.set(0)
        self.status_label.configure(text="Generating morphing...", foreground='blue')
        
        
        thread = threading.Thread(target=self._generate_morphing_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_morphing_thread(self):
        
        try:
            
            self.root.after(0, lambda: self.progress_var.set(10))
            
            
            source_image = load_custom_image(self.source_image_path.get())
            target_image = load_custom_image(self.target_image_path.get())
            
            self.root.after(0, lambda: self.progress_var.set(30))
            
            
            output_dir = './outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            results = self.morphing_pipeline.create_morph_sequence(
                source_image=source_image,
                target_image=target_image,
                num_steps=self.num_frames.get(),
                sequence_name="gui_morph",
                save_frames=True,  
                create_video=True
            )
            
            self.root.after(0, lambda: self.progress_var.set(80))
            
            
            self.frame_images, frames_dir, self.generated_gif_path = results
            
            self.root.after(0, lambda: self.progress_var.set(100))
            
            
            self.root.after(0, self._display_results)
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
            
    def _display_results(self):
        
        try:
            
            self.image_references.clear()
            if hasattr(self, 'gif_frames'):
                delattr(self, 'gif_frames')
            
            
            if self.generated_gif_path and os.path.exists(self.generated_gif_path):
                self.display_gif()
                
            
            self.display_frames()
            
            
            self.status_label.configure(text="Morphing generated successfully!", foreground='green')
            
        except Exception as e:
            self._show_error(f"Display error: {str(e)}")
        finally:
            
            self.generate_button.configure(state='normal')
            
    def display_gif(self):
        
        try:
            
            gif_image = Image.open(self.generated_gif_path)
            
            
            frames = []
            try:
                while True:
                    
                    display_size = (400, 400)
                    frame = gif_image.copy().resize(display_size, Image.Resampling.LANCZOS)
                    frames.append(ImageTk.PhotoImage(frame))
                    gif_image.seek(gif_image.tell() + 1)
            except EOFError:
                pass  
            
            if frames:
                
                self.gif_frames = frames
                self.current_frame = 0
                
                
                self._animate_gif()
            else:
                self.gif_label.configure(text="No frames found in GIF")
                
        except Exception as e:
            self.gif_label.configure(text=f"Error displaying GIF: {str(e)}")
            
    def _animate_gif(self):
        
        if hasattr(self, 'gif_frames') and self.gif_frames:
            
            self.gif_label.configure(image=self.gif_frames[self.current_frame], text="")
            self.gif_label.image = self.gif_frames[self.current_frame]
            
            
            self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
            
            
            self.root.after(200, self._animate_gif)
            
    def display_frames(self):
        
        try:
            
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
                
            if self.frame_images is None or len(self.frame_images) == 0:
                ttk.Label(self.scrollable_frame, text="No frames to display").pack()
                return
                
            
            frames_per_row = 4
            for i, frame_tensor in enumerate(self.frame_images):
                row = i // frames_per_row
                col = i % frames_per_row
                
                
                frame_np = frame_tensor.squeeze().cpu().numpy()
                
                
                if frame_np.shape == (784,):
                    frame_np = frame_np.reshape(28, 28)
                elif len(frame_np.shape) == 3 and frame_np.shape[0] == 1:
                    
                    frame_np = frame_np.squeeze()
                elif len(frame_np.shape) == 1 and len(frame_np) == 784:
                    frame_np = frame_np.reshape(28, 28)
                    
                
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    
                frame_img = Image.fromarray(frame_np, mode='L')
                
                
                frame_img = frame_img.resize((80, 80), Image.Resampling.LANCZOS)
                frame_photo = ImageTk.PhotoImage(frame_img)
                
                
                self.image_references.append(frame_photo)
                
                
                frame_widget = ttk.Frame(self.scrollable_frame)
                frame_widget.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                
                
                img_label = ttk.Label(frame_widget, image=frame_photo)
                img_label.pack()
                img_label.image = frame_photo  
                
                
                ttk.Label(frame_widget, text=f"Frame {i+1}").pack()
                
        except Exception as e:
            ttk.Label(self.scrollable_frame, text=f"Error displaying frames: {str(e)}").pack()
            
    def _show_error(self, error_message: str):
        
        messagebox.showerror("Generation Error", f"Failed to generate morphing: {error_message}")
        self.status_label.configure(text="Generation failed", foreground='red')
        self.generate_button.configure(state='normal')
        self.progress_var.set(0)


def main():
    
    root = tk.Tk()
    
    
    style = ttk.Style()
    style.theme_use('clam')
    
    
    app = VAEMorphingGUI(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()