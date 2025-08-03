import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess
import threading
import os
import glob
from pathlib import Path
import time
import psutil


class TextImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text, Log, and Image Viewer")
        self.root.geometry("1000x650")

        self.current_text_file = None
        self.static_image_ref = None
        self.dynamic_image_ref = None

        # --- Top Buttons ---
        quick_menu = tk.Frame(self.root)
        quick_menu.pack(fill=tk.X)

        tk.Button(quick_menu, text="Load Text", command=lambda: self.open_text_file("config.txt")).pack(side=tk.LEFT, padx=5)
        tk.Button(quick_menu, text="Load Static Image", command=lambda: self.open_static_image("")).pack(side=tk.LEFT, padx=5)
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

        self.paned.add(self.left_container, weight=2)

        # RIGHT SIDE (static + dynamic image)
        self.right_container = ttk.PanedWindow(self.paned, orient=tk.VERTICAL)

        self.dynamic_fits_frame = ttk.LabelFrame(self.right_container, text="Fits stamp")
        self.dynamic_fits_canvas = tk.Canvas(self.dynamic_fits_frame, bg="white", height=300)
        self.dynamic_fits_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_container.add(self.dynamic_fits_frame, weight=1)

        self.dynamic_img_frame = ttk.LabelFrame(self.right_container, text="Annotated reference")
        self.dynamic_canvas = tk.Canvas(self.dynamic_img_frame, bg="white", height=300)
        self.dynamic_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_container.add(self.dynamic_img_frame, weight=1)

        self.paned.add(self.right_container, weight=2)

        # --- Menu ---
        self.create_menu()

        # Load defaults
        self.open_text_file("config.txt")
        #self.open_static_image("images/ref_annotated.png")

        # store the external process so it can be accessed later
        self.external_process = None

        # dynamic plotting
        self.poll_for_latest_image()


    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Text File", command=self.open_text_file)
        file_menu.add_command(label="Save Text File", command=self.save_text_file)
        file_menu.add_command(label="Save Text As...", command=self.save_text_as)
        file_menu.add_separator()
        file_menu.add_command(label="Open Static Image", command=self.open_static_image)
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

    def open_static_image(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path and os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                img = img.resize((400, 300), Image.ANTIALIAS)
                self.static_image_ref = ImageTk.PhotoImage(img)
                self.static_canvas.delete("all")
                self.static_canvas.create_image(200, 150, image=self.static_image_ref)
            except Exception as e:
                self.static_canvas.delete("all")
                self.static_canvas.create_text(200, 150, text=f"Error:\n{e}", fill="red")

    def refresh_dynamic_image(self, file_path):
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                img = img.resize((400, 300), Image.ANTIALIAS)
                self.dynamic_image_ref = ImageTk.PhotoImage(img)
                self.dynamic_canvas.delete("all")
                self.dynamic_canvas.create_image(200, 150, image=self.dynamic_image_ref)
            except Exception as e:
                self.dynamic_canvas.delete("all")
                self.dynamic_canvas.create_text(200, 150, text=f"Image error:\n{e}", fill="red")

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

    def get_latest_image_in_latest_scene(self, out_path):
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
            print(f"Error finding latest image: {e}")
            return None

    def poll_for_latest_image(self):

        out_path = Path("D:/offline_tests/")  # replace with actual config or path
        latest_img = self.get_latest_image_in_latest_scene(out_path)
        if latest_img:
            self.refresh_dynamic_image(latest_img)

        self.root.after(10000, self.poll_for_latest_image)  # poll every 10 seconds

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = TextImageApp(root)
    root.mainloop()
