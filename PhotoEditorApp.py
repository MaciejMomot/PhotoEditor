import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Obrazów")
        self.root.geometry("850x600")

        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue")

        self.image = None
        self.processed_image = None
        self.current_effect = None  
        self.brightness_value = 0
        self.contrast_value = 0
        
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.left_frame.pack(side="left", padx=10, pady=10, expand=True)
        self.middle_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.middle_frame.pack(side="left", padx=10, pady=10, expand=True)
        self.right_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.right_frame.pack(side="left", padx=10, pady=10, expand=True)

        self.left_label = ctk.CTkLabel(self.left_frame, text="Oryginalne zdjęcie")
        self.left_label.pack(pady=10)
        self.middle_label = ctk.CTkLabel(self.middle_frame, text="Zdjęcie po obróbce")
        self.middle_label.pack(pady=10)
        
        self.right_label = ctk.CTkLabel(self.right_frame, text="Wykresy TODO")
        self.right_label.pack(pady=10)
        
        self.left_frame.pack_propagate(False)
        self.middle_frame.pack_propagate(False)
        self.right_frame.pack_propagate(False)

        self.button_frame = ctk.CTkFrame(root)
        self.button_frame.pack(pady=10)

        self.load_button = ctk.CTkButton(self.button_frame, text="Otwórz obraz", command=self.load_image)
        self.load_button.pack(side="left", padx=5)

        self.save_button = ctk.CTkButton(self.button_frame, text="Zapisz", command=self.save_image)
        self.save_button.pack(side="left", padx=5)

        self.filter_menu = ctk.CTkOptionMenu(self.button_frame, values=["Szarość", "Negatyw", "Binaryzacja"],
                                             command=self.apply_pixels_filter)
        self.filter_menu.pack(side="left", padx=5)
        self.filter_menu.set("Pixele")

        self.filter_menu2 = ctk.CTkOptionMenu(self.button_frame, values=["Wyostrzający", "Uśredniający", "Gaussa", "Znajdowanie krawędzi", "Sobel"],
                                             command=self.apply_filter)
        self.filter_menu2.pack(side="left", padx=5)
        self.filter_menu2.set("Filtry")
        
        self.refresh_button = ctk.CTkButton(self.button_frame, text="Odśwież", command=self.refresh)
        self.refresh_button.pack(side="left", padx=5)

        self.brightness_label = ctk.CTkLabel(root, text="Jasność")
        self.brightness_label.pack(pady=(10, 0))
        self.brightness_scale = ctk.CTkSlider(root, from_=-255, to=255, command=self.update_brightness)
        self.brightness_scale.pack(fill="x", padx=10, pady=10)
        
        self.contrast_label = ctk.CTkLabel(root, text="Kontrast")
        self.contrast_label.pack(pady=(10, 0))
        self.contrast_scale = ctk.CTkSlider(root, from_=-255, to=255, command=self.update_contrast)
        self.contrast_scale.pack(fill="x", padx=10, pady=10)
    
    # LOADING AND SAVING IMAGES

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.processed_image = self.image.copy()
            self.current_effect = self.image.copy()
            self.display_image(self.image, self.left_label)
            self.display_image(self.processed_image, self.middle_label)
            self.histograms()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                self.processed_image.save(file_path)

    # DISPLAYING IMAGES

    def display_image(self, img, label):
        self.root.update_idletasks()  
        frame_width = 400  # Stała szerokość
        img_ratio = img.width / img.height

        new_width = frame_width
        new_height = int(frame_width / img_ratio)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=resized_img, dark_image=resized_img, size=(new_width, new_height))
        label.configure(image=ctk_img, text="")
        label.image = ctk_img
        label.pack(expand = True, anchor = "center")
        
    def refresh(self):
        self.current_effect = self.image.copy()
        self.display_image(self.current_effect, self.middle_label)
        self.processed_image = self.image.copy()
        self.brightness_scale.set(0)
        self.contrast_scale.set(0)
        self.brightness_value = 0
        self.contrast_value = 0
        self.histograms()
        
    # APPLYING FILTERS

    def apply_pixels_filter(self, choice):
        if self.image:
            if choice == "Szarość":
                self.to_grayscale(False)
            elif choice == "Negatyw":
                self.to_negative()
            elif choice == "Binaryzacja":
                self.binarize()
            self.current_effect = self.processed_image.copy()
            self.histograms()

    def apply_filter(self, choice):
        if self.image:
            if choice == "Wyostrzający":
                self.apply_sharpen()
            elif choice == "Uśredniający":
                self.apply_blur()
            elif choice == "Gaussa":
                self.apply_gaussian(1)
            elif choice == "Znajdowanie krawędzi":
                self.apply_find_edges()
            elif choice == "Sobel":
                self.apply_gradient_magnitude("roberts")
            self.current_effect = self.processed_image.copy()
            self.histograms()

    def to_grayscale(self, trigger, img=None):
        if img is None:
            img = self.processed_image

        pixels = np.array(img)
        grey = np.mean(pixels, axis=2).astype(np.uint8)
        tmp = np.stack([grey]*3, axis=2)

        if trigger:
            return tmp
        else:
            self.processed_image = Image.fromarray(tmp)
            self.display_image(self.processed_image, self.middle_label)
            print("Grayscale applied")

    def to_negative(self):
        pixels = np.array(self.processed_image)
        neg = 255 - pixels
        self.processed_image = Image.fromarray(neg)
        self.display_image(self.processed_image, self.middle_label)

    def binarize(self):
        pixels = np.array(self.processed_image)
        gray = np.mean(pixels, axis=2)
        binary = (gray > 128) * 255
        self.processed_image = Image.fromarray(np.stack([binary]*3, axis=2).astype(np.uint8))
        self.display_image(self.processed_image, self.middle_label)

    def update_brightness(self, value):
        self.brightness_value = int(float(value))
        self.apply_brightness_contrast()
    
    def update_contrast(self, value):
        self.contrast_value = float(value) / 255
        self.apply_brightness_contrast()
    
    def apply_brightness_contrast(self):
        if self.current_effect:
            pixels = np.array(self.current_effect, dtype=np.float32)
            mean = np.mean(pixels, axis=(0, 1), keepdims=True)
            pixels = np.clip((pixels - mean) * (1 + self.contrast_value) + mean, 0, 255)
            pixels = np.clip(pixels + self.brightness_value, 0, 255)
            self.processed_image = Image.fromarray(pixels.astype(np.uint8))
            self.display_image(self.processed_image, self.middle_label)
            self.histograms()

    def apply_blur(self):
        if self.image:
            pixels = np.array(self.processed_image)
            kernel = np.ones((3, 3)) / 9
            blurred = self.convolve(pixels, kernel)
            self.processed_image = Image.fromarray(blurred)
            self.display_image(self.processed_image, self.middle_label)

    def apply_sharpen(self):
        if self.image:
            pixels = np.array(self.processed_image)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = self.convolve(pixels, kernel)
            self.processed_image = Image.fromarray(sharpened)
            self.display_image(self.processed_image, self.middle_label)
         
    def apply_find_edges(self):
        if self.image:
            pixels = np.array(self.processed_image)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            edges = self.convolve(pixels, kernel)
            self.processed_image = Image.fromarray(edges)
            self.display_image(self.processed_image, self.middle_label)


    def histograms(self):
        if self.processed_image:
            r, g, b = self.processed_image.copy().split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)

            self.display_histograms(r, g, b)

    def display_histograms(self, r, g, b):
        fig, axs = plt.subplots(3, 1, figsize=(3.5, 9), facecolor='none')
        fig.patch.set_alpha(0)  # Ustawienie całkowitej przejrzystości tła wykresów

        colors = ['red', 'green', 'blue']
        labels = ['R', 'G', 'B']
        channels = [r, g, b]

        for ax, channel, color, label in zip(axs, channels, colors, labels):
            ax.hist(channel.ravel(), bins=256, color=color, alpha=0.5)
            ax.set_title(label, color=color)
            ax.set_xlim([0, 256])
            ax.patch.set_alpha(0)  # Ustawienie przezroczystości dla każdego subplotu
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

        plt.tight_layout()

        # Usunięcie starych widgetów
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        # Osadzenie wykresów w tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(expand=True, anchor="center")
        
        # Usunięcie tła z płótna
        widget.config(bg='black', highlightthickness=0)

        plt.close(fig)


    # CONVOLUTION
    def convolve(self, img, kernel):
        height, width, channels = img.shape
        ksize = kernel.shape[0]
        pad = ksize // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        output = np.zeros_like(img)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    region = padded_img[y:y+ksize, x:x+ksize, c]
                    output[y, x, c] = np.clip(np.sum(region * kernel), 0, 255)

        return output.astype(np.uint8)
    

    # GRADIENT MAGNITUDE
    def apply_gradient_magnitude(self, type):
        if not self.image:
            return

        pixels = np.array(self.processed_image)

        if type == "sobel":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = kernel_x.T
        elif type == "prewitt":
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = kernel_x.T
        elif type == "roberts":
            kernel_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
            kernel_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        elif type == "sobel-feldman":
            kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
            kernel_y = kernel_x.T
        elif type == "scharr":
            kernel_x = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])
            kernel_y = kernel_x.T

        Gx = self.convolve(pixels, kernel_x)
        Gy = self.convolve(pixels, kernel_y)

        G = np.sqrt(Gx**2 + Gy**2)
        G = G / G.max() * 255

        treshold = np.mean(G)
        output = (G > treshold).astype(np.uint8) * 255

        self.processed_image = Image.fromarray(output)
        self.display_image(self.processed_image, self.middle_label)

        print(f"{type}'s gradient magnitude applied")

    # GAUSSIAN FILTER
    def apply_gaussian(self, sigma):
        if not self.image:
            return

        pixels = np.array(self.processed_image)

        size = int(6 * sigma + 1)
        kernel = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                kernel[x, y] = np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        output = self.convolve(pixels, kernel)

        self.processed_image = Image.fromarray(output)
        self.display_image(self.processed_image, self.middle_label)

        print("Gaussian filter applied")

if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageProcessorApp(root)
    root.mainloop()