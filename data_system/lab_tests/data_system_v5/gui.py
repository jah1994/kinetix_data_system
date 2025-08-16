import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from astropy.io import fits
import subprocess
import threading
import os
import glob
from pathlib import Path
import time
import psutil

def load_config(path="config.txt"):
    config = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and full-line comments

            # Remove inline comment
            line = line.split("#", 1)[0].strip()

            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip('"').strip("'")  # remove quotes
    return config

config = load_config()


class TextImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinetix data system")
        self.root.geometry("1000x650")

        self.current_text_file = None
        self.static_image_ref = None
        self.dynamic_image_ref = None

        # --- Top Buttons ---
        quick_menu = tk.Frame(self.root)
        quick_menu.pack(fill=tk.X)

        tk.Button(quick_menu, text="Load Config file", command=lambda: self.open_text_file("config.txt")).pack(side=tk.LEFT, padx=5)
        tk.Button(quick_menu, text="Run Script", command=self.run_external_script).pack(side=tk.LEFT, padx=5)
        tk.Button(quick_menu, text="Stop Script", command=self.stop_external_script).pack(side=tk.LEFT, padx=5)

        # --- Main Split: Left (text/log), Right (images) ---
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # LEFT SIDE (text + log)
        self.left_container = ttk.PanedWindow(self.paned, orient=tk.VERTICAL)

        self.text_frame = ttk.Frame(self.left_container)
        self.text_widget = tk.Text(self.text_frame, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.left_container.add(self.text_frame, weight=3)

        self.log_frame = ttk.Frame(self.left_container)
        self.log_widget = tk.Text(self.log_frame, height=10, bg="#f0f0f0", state=tk.DISABLED)
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        self.left_container.add(self.log_frame, weight=1)

        self.paned.add(self.left_container, weight=1)

        # RIGHT SIDE (dynamic images)
        self.right_container = ttk.PanedWindow(self.paned, orient=tk.VERTICAL)

        # --- TOP: two FITS stamps side by side ---
        self.dynamic_stamp_frame = ttk.LabelFrame(self.right_container, text="Source stamp(s)")

        # a horizontal panedwindow inside the label frame
        self.stamps_container = ttk.PanedWindow(self.dynamic_stamp_frame, orient=tk.HORIZONTAL)
        self.stamps_container.pack(fill=tk.BOTH, expand=True)

        # first stamp canvas
        self.dynamic_stamp_canvas1 = tk.Canvas(self.stamps_container, bg="white", height=300, width=300)
        self.stamps_container.add(self.dynamic_stamp_canvas1, weight=1)

        # second stamp canvas
        self.dynamic_stamp_canvas2 = tk.Canvas(self.stamps_container, bg="white", height=300, width=300)
        self.stamps_container.add(self.dynamic_stamp_canvas2, weight=1)

        self.right_container.add(self.dynamic_stamp_frame, weight=1)

        # --- BOTTOM: reference image (unchanged) ---
        self.dynamic_img_frame = ttk.LabelFrame(self.right_container, text="Reference image")
        self.dynamic_canvas = tk.Canvas(self.dynamic_img_frame, bg="white", height=1200)
        self.dynamic_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_container.add(self.dynamic_img_frame, weight=4)

        # add the right container to the main paned window
        self.paned.add(self.right_container, weight=3)


        # --- Menu ---
        self.create_menu()

        # Load defaults
        self.open_text_file("config.txt")

        # store the external process so it can be accessed later
        self.external_process = None

        # dynamic plotting
        self.poll_for_latest_reference_image()
        self.poll_for_latest_stamp_image()


    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Text File", command=self.open_text_file)
        file_menu.add_command(label="Save Text File", command=self.save_text_file)
        file_menu.add_command(label="Save Text As...", command=self.save_text_as)
        file_menu.add_separator()
        file_menu.add_command(label="Run Script", command=self.run_external_script)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def open_text_file(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_widget.delete("1.0", tk.END)
                    self.text_widget.insert(tk.END, content)
                    self.current_text_file = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file:\n{e}")

    def save_text_file(self):
        if self.current_text_file:
            try:
                content = self.text_widget.get("1.0", tk.END)
                with open(self.current_text_file, 'w', encoding='utf-8') as file:
                    file.write(content.strip() + '\n')
                messagebox.showinfo("Saved", f"Saved to {self.current_text_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")
        else:
            self.save_text_as()

    def save_text_as(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                content = self.text_widget.get("1.0", tk.END)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content.strip() + '\n')
                self.current_text_file = file_path
                messagebox.showinfo("Saved", f"Saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def refresh_reference_image(self, file_path):
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                img = img.resize((800, 700), Image.LANCZOS)
                self.dynamic_image_ref = ImageTk.PhotoImage(img)
                self.dynamic_canvas.delete("all")
                self.dynamic_canvas.create_image(400, 350, image=self.dynamic_image_ref)
            except Exception as e:
                self.dynamic_canvas.delete("all")
                self.dynamic_canvas.create_text(400, 350, text=f"Image error:\n{e}", fill="red")

    def refresh_stamp_image(self, file_path, canvas, attr_name):
        def fits_to_photoimage(file_path, size=(200, 200)):
            with fits.open(file_path) as hdul:
                data = hdul[0].data
            if data is None:
                raise ValueError("FITS file has no image data")

            # normalize data for display (scale to 0-255)
            data = np.nan_to_num(data)  # replace NaNs/Infs
            data = data - data.min()
            if data.max() > 0:
                data = (255 * data / data.max()).astype(np.uint8)
            else:
                data = data.astype(np.uint8)

            # convert to PIL Image (grayscale "L" mode)
            img = Image.fromarray(data, mode="L")
            img = img.resize((200, 200), Image.LANCZOS)

            return ImageTk.PhotoImage(img)

        if os.path.exists(file_path):
            try:
                photo = fits_to_photoimage(file_path, size=(200, 200))
                setattr(self, attr_name, photo)  # keep reference so Tk doesn't GC
                canvas.delete("all")
                canvas.create_image(100, 100, image=photo)
            except Exception as e:
                canvas.delete("all")
                canvas.create_text(
                    100, 100, text=f"Image error:\n{e}", fill="red"
                )

    def run_external_script(self, script_path="auto.py"):
        def target():
            try:
                self.external_process = subprocess.Popen(
                    ["python", "-u", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows only
                )
                while True:
                    if self.external_process.poll() is not None:
                        break  # Process has finished
                    line = self.external_process.stdout.readline()
                    if line:
                        self.log_widget.config(state=tk.NORMAL)
                        self.log_widget.insert(tk.END, line)
                        self.log_widget.see(tk.END)
                        self.log_widget.config(state=tk.DISABLED)
                    else:
                        time.sleep(0.1)  # Avoid tight loop

                # Read remaining output after process ends
                remaining = self.external_process.stdout.read()
                if remaining:
                    self.log_widget.config(state=tk.NORMAL)
                    self.log_widget.insert(tk.END, remaining)
                    self.log_widget.see(tk.END)
                    self.log_widget.config(state=tk.DISABLED)

            except Exception as e:
                self.log_widget.config(state=tk.NORMAL)
                self.log_widget.insert(tk.END, f"Error: {e}\n")
                self.log_widget.see(tk.END)
                self.log_widget.config(state=tk.DISABLED)
            finally:
                self.external_process = None

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def stop_external_script(self):
        if self.external_process and self.external_process.poll() is None:
            try:
                parent = psutil.Process(self.external_process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                self.log_widget.config(state=tk.NORMAL)
                self.log_widget.insert(tk.END, "Script and child processes killed.\n")
                self.log_widget.see(tk.END)
                self.log_widget.config(state=tk.DISABLED)
            except Exception as e:
                self.log_widget.config(state=tk.NORMAL)
                self.log_widget.insert(tk.END, f"Failed to kill script: {e}\n")
                self.log_widget.see(tk.END)
                self.log_widget.config(state=tk.DISABLED)

    def get_latest_reference_image(self, out_path):
        try:
            # Step 1: Get list of RUN_* dirs, sorted by creation time (descending)
            run_dirs = sorted(
                Path(out_path).glob("RUN_*"),
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            if not run_dirs:
                return None

            latest_run = run_dirs[0]

            # Step 2: Get list of SCENE_* dirs inside the latest RUN dir
            scene_dirs = sorted(
                latest_run.glob("SCENE_*"),
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            if not scene_dirs:
                return None

            latest_scene = scene_dirs[0]

            # Step 3: Look for image files in images/ inside latest SCENE
            image_files = sorted(
                (latest_scene / "images").glob("ref_annotated.png"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            if not image_files:
                return None

            return str(image_files[0])

        except Exception as e:
            print(f"Error finding latest reference image: {e}")
            return None

    def get_latest_stamp_image(self, out_path):
        try:
            # Step 1: Get list of RUN_* dirs, sorted by creation time (descending)
            run_dirs = sorted(
                Path(out_path).glob("RUN_*"),
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            if not run_dirs:
                return None

            latest_run = run_dirs[0]

            # Step 2: Get list of SCENE_* dirs inside the latest RUN dir
            scene_dirs = sorted(
                latest_run.glob("SCENE_*"),
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            if not scene_dirs:
                return None

            latest_scene = scene_dirs[0]

            # Step 3: Look for image files in images/ inside latest SCENE
            image_files = sorted(
                (latest_scene / "fits").glob("source*.fits"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            if not image_files:
                return None

            return image_files

        except Exception as e:
            print(f"Error finding latest stamp image: {e}")
            return None

    def poll_for_latest_reference_image(self):
        out_path = Path(config['out_path'])
        latest_ref_img = self.get_latest_reference_image(out_path)
        if latest_ref_img:
            self.refresh_reference_image(latest_ref_img)
        self.root.after(10000, self.poll_for_latest_reference_image)  # poll every 10 seconds

    def poll_for_latest_stamp_image(self):
        out_path = Path(config['out_path'])
        latest_stamp_imgs = self.get_latest_stamp_image(out_path)
        for i, canvas in enumerate([self.dynamic_stamp_canvas1, self.dynamic_stamp_canvas2]):
            try:
                if latest_stamp_imgs is not None:
                    file_path = latest_stamp_imgs[i]
                    if os.path.exists(file_path):
                        self.refresh_stamp_image(file_path, canvas, str(file_path))
            except IndexError:
                pass
        self.root.after(5, self.poll_for_latest_stamp_image)  # poll every 5 seconds

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = TextImageApp(root)
    root.mainloop()
