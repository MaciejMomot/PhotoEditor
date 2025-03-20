import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("PhotoHub")
        self.root.geometry("850x600")

        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue")

        self.image = None
        self.processed_image = None
        self.current_effect = None  
        self.processed_image_history = []
        self.brightness_value = 0
        self.contrast_value = 0
        self.threshold_value = None
        self.threshold_text = ''
        self.selected_filter = "Choose operation"
        
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.left_frame.pack(side="left", padx=10, pady=10, expand=True)
        self.middle_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.middle_frame.pack(side="left", padx=10, pady=10, expand=True)
        self.right_frame = ctk.CTkFrame(self.main_frame, width=400, height=400)
        self.right_frame.pack(side="left", padx=10, pady=10, expand=True)

        self.left_label = ctk.CTkLabel(self.left_frame, text="Original image")
        self.left_label.pack(pady=10)
        self.middle_label = ctk.CTkLabel(self.middle_frame, text="Processed image")
        self.middle_label.pack(pady=10)
        
        self.right_label = ctk.CTkLabel(self.right_frame, text="Histograms")
        self.right_label.pack(pady=10)
        
        self.left_frame.pack_propagate(False)
        self.middle_frame.pack_propagate(False)
        self.right_frame.pack_propagate(False)

        self.button_frame = ctk.CTkFrame(root)
        self.button_frame.pack(pady=10)

        self.load_button = ctk.CTkButton(self.button_frame, text="Load image", command=self.load_image)
        self.load_button.pack(side="left", padx=5)

        self.save_button = ctk.CTkButton(self.button_frame, text="Save", command=self.save_image)
        self.save_button.pack(side="left", padx=5)

        self.filter_menu = ctk.CTkComboBox(self.button_frame, 
                                           values=["Choose operation",
                                                    "grayscale",
                                                    "negative", 
                                                    "binarization", 
                                                    "sharpening", 
                                                    "blur", 
                                                    "gaussian", 
                                                    "laplacian", 
                                                    "Robert's cross", 
                                                    "Sobel operator", 
                                                    "Prewitt operator", 
                                                    "Scharr operator", 
                                                    "Sobel-Feldman operator"], 
                                            command=self.on_select)
        self.filter_menu.pack(side="left", padx=5)
        self.filter_menu.set("Choose operation")

        self.apply_button = ctk.CTkButton(self.button_frame, text="Apply", command=self.apply_operation)
        self.apply_button.pack(side="left", padx=5)
        
        self.reset_button = ctk.CTkButton(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side="left", padx=5)
        
        self.undo_button = ctk.CTkButton(self.button_frame, text="Undo", command=self.undo)
        self.undo_button.pack(side="left", padx=5)

        self.refresh_button = ctk.CTkButton(self.button_frame, text="Refresh Plots", command=self.prepare_plots)
        self.refresh_button.pack(side="left", padx=5)
        
        self.slider_frame = ctk.CTkFrame(root)
        self.slider_frame.pack(pady=10, fill="x")
        
        self.slider_frame.grid_rowconfigure(0, weight=1)
        self.slider_frame.grid_columnconfigure(0, weight=1)
        self.slider_frame.grid_columnconfigure(1, weight=1)

        self.brightness_label = ctk.CTkLabel(self.slider_frame, text="Brightness")
        self.brightness_label.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.brightness_scale = ctk.CTkSlider(self.slider_frame, from_=-255, to=255, command=self.set_brightness)
        self.brightness_scale.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.contrast_label = ctk.CTkLabel(self.slider_frame, text="Contrast")
        self.contrast_label.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.contrast_scale = ctk.CTkSlider(self.slider_frame, from_=-255, to=255, command=self.set_contrast)
        self.contrast_scale.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Tworzymy główną ramkę
        self.customization_frame = ctk.CTkFrame(root)
        self.customization_frame.pack(pady=10, fill="x")

        # Konfiguracja wierszy i kolumn w ramach
        self.customization_frame.grid_rowconfigure(0, weight=1)
        self.customization_frame.grid_rowconfigure(1, weight=1)
        self.customization_frame.grid_rowconfigure(2, weight=1)
        self.customization_frame.grid_rowconfigure(3, weight=1)  # Dla wiersza przycisku
        self.customization_frame.grid_columnconfigure(0, weight=3)
        self.customization_frame.grid_columnconfigure(1, weight=1)
        self.customization_frame.grid_columnconfigure(2, weight=1)
        self.customization_frame.grid_columnconfigure(3, weight=1)
        self.customization_frame.grid_columnconfigure(4, weight=1)

        # Próg binarnego filtra
        self.threshold_label = ctk.CTkLabel(self.customization_frame, text = self.threshold_text)
        self.threshold_label.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Pole do wprowadzenia wartości progu
        self.threshold_entry = ctk.CTkEntry(self.customization_frame, width=10)
        self.threshold_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Macierz do ustawiania filtra w 2. kolumnie
        self.kernel_entries = []  # Lista przechowująca pola dla macierzy 3x3
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = ctk.CTkEntry(self.customization_frame, width=40)
                entry.grid(row=i, column=j+1, padx=5, pady=5)
                entry.insert(0, "0")  # Ustawiamy początkowo wartość na 0
                row_entries.append(entry)
            self.kernel_entries.append(row_entries)

        # Przycisk do zatwierdzenia filtra w 3. kolumnie
        self.apply_kernel_button = ctk.CTkButton(self.customization_frame, text="Zatwierdź filtr", command=self.apply_custom_filter)
        self.apply_kernel_button.grid(row=1, column=4, rowspan=3, padx=10, pady=20, sticky="ew")
        
        
    
    # LOADING AND SAVING IMAGES

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.processed_image = self.image.copy()
            self.processed_image_history.append(self.image.copy())
            self.current_effect = self.image.copy()
            self.display_image(self.image, self.left_label)
            self.display_image(self.processed_image, self.middle_label)
            self.reset_labels()
            self.prepare_plots()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                self.processed_image.save(file_path)
                self.reset_labels()
                print("Image saved")


    # DISPLAYING IMAGES

    def display_image(self, img, label):
        self.root.update_idletasks()  
        frame_width = 400 
        img_ratio = img.width / img.height

        new_width = frame_width
        new_height = int(frame_width / img_ratio)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=resized_img, dark_image=resized_img, size=(new_width, new_height))
        label.configure(image=ctk_img, text="")
        label.image = ctk_img
        label.pack(expand = True, anchor = "center")

    def reset_labels(self):
        self.filter_menu.set("Choose operation")
        self.selected_filter = None
        self.threshold_text = ""
        self.threshold_label.configure(text=self.threshold_text)
        self.threshold_entry.delete(0, "end")

    def reset(self):
        self.current_effect = self.image.copy()
        self.display_image(self.current_effect, self.middle_label)
        self.processed_image = self.image.copy()
        self.brightness_scale.set(0)
        self.contrast_scale.set(0)
        self.brightness_value = 0
        self.contrast_value = 0
        self.reset_labels()
        self.processed_image_history.append(self.image.copy())
        print("Reset applied")


    def undo(self):
        
        if len(self.processed_image_history) > 0:
            self.processed_image = self.processed_image_history.pop()
            self.current_effect = self.processed_image.copy()
            self.brightness_scale.set(0)
            self.contrast_scale.set(0)
            self.brightness_value = 0
            self.contrast_value = 0
            self.reset_labels()
            self.display_image(self.processed_image, self.middle_label)
            print("Undo applied")
        else:
            self.show_error_message("No more undo steps available")
    

    def on_select(self, choice):
        self.selected_filter = choice
        if choice == "binarization":
            self.threshold = 128
            self.threshold_text = "Threshold: "
        elif choice == "gaussian":
            self.threshold = 1
            self.threshold_text = "Sigma: "
        elif choice == "blur":
            self.threshold = 1
            self.threshold_text = "Parameter: "
        else:
            self.threshold = None
            self.threshold_text = ''

        self.threshold_label.configure(text=self.threshold_text)
        if self.threshold is not None:
            self.threshold_entry.delete(0, "end")
            self.threshold_entry.insert(0, str(self.threshold))
        else:
            self.threshold_entry.delete(0, "end")

    def apply_operation(self):

        if not self.image or self.selected_filter == "Choose operation":
            return
        
        self.processed_image_history.append(self.processed_image.copy())
        self.current_effect = self.processed_image.copy()
        
        if self.selected_filter == "grayscale":
            self.grayscale()
        elif self.selected_filter == "negative":
            self.negative()
        elif self.selected_filter == "binarization":
            self.set_threshold()
            self.binarization(self.threshold)
        elif self.selected_filter == "sharpening":
            self.sharpen()
        elif self.selected_filter == "blur":
            self.set_threshold()
            self.blur(self.threshold)
        elif self.selected_filter == "gaussian":
            self.set_threshold()
            self.gaussian(self.threshold)
        elif self.selected_filter == "laplacian":
            self.laplacian()
        elif self.selected_filter == "Robert's cross":
            self.gradient(self.selected_filter)
        elif self.selected_filter == "Sobel operator":
            self.gradient(self.selected_filter)
        elif self.selected_filter == "Prewitt operator":
            self.gradient(self.selected_filter)
        elif self.selected_filter == "Scharr operator":
            self.gradient(self.selected_filter)
        elif self.selected_filter == "Sobel-Feldman operator":
            self.gradient(self.selected_filter)

        self.reset_labels()
        self.prepare_plots()

    
    def set_brightness(self, value):
        self.brightness_value = int(float(value))
        self.brightness_contrast()
    
    def set_contrast(self, value):
        self.contrast_value = float(value) / 255
        self.brightness_contrast()


    
    def set_threshold(self):

        try:
            threshold = int(self.threshold_entry.get())        

            if self.selected_filter == "binarization" and 0 <= threshold <= 255:
                self.threshold = threshold
            elif self.selected_filter == "binarization":
                self.show_error_message("Please insert threshold value between 0 and 255.")
            elif self.selected_filter == "gaussian" and threshold > 0:
                self.threshold = threshold
            elif self.selected_filter == "gaussian":
                self.show_error_message("Please insert sigma value greater than 0.")
            elif self.selected_filter == "blur" and threshold > 0 and type(threshold) == int:
                self.threshold = threshold
            elif self.selected_filter == "blur":
                self.show_error_message("Please insert threshold value greater than 0.")

        except ValueError:
            self.show_error_message("Please insert a valid number.")

    def show_error_message(self, message):

        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")
        error_label = ctk.CTkLabel(error_window, text=message)
        error_label.pack(padx=10, pady=10)
        ok_button = ctk.CTkButton(error_window, text="OK", command=error_window.destroy)
        ok_button.pack(pady=10)
        error_window.grab_set()
        error_window.wait_window()


    """ PIXEL OPERATIONS """

    # GRAYSCALE

    def grayscale(self):
        pixels = np.array(self.processed_image)
        grey = np.mean(pixels, axis=2).astype(np.uint8)
        tmp = np.stack([grey]*3, axis=2)
        self.processed_image = Image.fromarray(tmp)
        self.display_image(self.processed_image, self.middle_label)
        print("Grayscale applied")


    # NEGATIVE

    def negative(self):
        pixels = np.array(self.processed_image)
        neg = 255 - pixels
        self.processed_image = Image.fromarray(neg)
        self.display_image(self.processed_image, self.middle_label)
        print("Negative applied")


    # BINARIZATION

    def binarization(self, threshold):
        pixels = np.array(self.processed_image)
        gray = np.mean(pixels, axis=2)
        binary = (gray > threshold) * 255
        self.processed_image = Image.fromarray(np.stack([binary]*3, axis=2).astype(np.uint8))
        self.display_image(self.processed_image, self.middle_label)
        print("Binarization applied")

    
    # CHANGING BRIGHTNESS AND CONTRAST

    def brightness_contrast(self):
        if self.current_effect:
            pixels = np.array(self.current_effect, dtype=np.float32)
            mean = np.mean(pixels, axis=(0, 1), keepdims=True)
            pixels = np.clip((pixels - mean) * (1 + self.contrast_value) + mean, 0, 255)
            pixels = np.clip(pixels + self.brightness_value, 0, 255)
            self.processed_image = Image.fromarray(pixels.astype(np.uint8))
            self.display_image(self.processed_image, self.middle_label)


    """ FILTERS """

    # CONVOLUTION

    def convolve(self, img, kernel):

        height, width, channels = img.shape
        ksize = kernel.shape[0]
        pad = ksize // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        output = np.zeros_like(img)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    region = padded_img[y:y+ksize, x:x+ksize, c]
                    output[y, x, c] = np.clip(np.sum(region * kernel), 0, 255)

        return output.astype(np.uint8)
    

    # SHARPENING

    def sharpen(self):
        pixels = np.array(self.processed_image)
        kernel = np.array([[0, -3, 0], [-3, 16, -3], [0, -3, 0]])
        sharpened = self.convolve(pixels, kernel)
        self.processed_image = Image.fromarray(sharpened)
        self.display_image(self.processed_image, self.middle_label)
        print("Sharpening applied")


    # BLURRING

    def blur(self, threshold):
        pixels = np.array(self.processed_image)
        kernel = np.array([[1, 1, 1], [1, threshold, 1], [1, 1, 1]])
        kernel = kernel / np.sum(kernel)
        blurred = self.convolve(pixels, kernel)
        self.processed_image = Image.fromarray(blurred)
        self.display_image(self.processed_image, self.middle_label)
        print("Blurring applied")

    
    # GAUSSIAN FILTER

    def gaussian(self, sigma):

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


    # LAPLACE OPERATOR

    def laplacian(self):
        pixels = np.array(self.processed_image)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edges = self.convolve(pixels, kernel)
        self.processed_image = Image.fromarray(edges)
        self.display_image(self.processed_image, self.middle_label)
        print("Laplacian operator applied")
    

    # GRADIENT EDGE DETECTION

    def gradient(self, type):

        pixels = np.array(self.processed_image)

        if type == "Sobel operator":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = kernel_x.T
        elif type == "Prewitt operator":
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = kernel_x.T
        elif type == "Robert's cross":
            kernel_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
            kernel_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        elif type == "Sobel-Feldman operator":
            kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
            kernel_y = kernel_x.T
        elif type == "Scharr operator":
            kernel_x = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])
            kernel_y = kernel_x.T

        norm = np.sum(np.abs(kernel_x))
        kernel_x = kernel_x / norm
        kernel_y = kernel_y / norm

        Gx = self.convolve(pixels, kernel_x)
        Gy = self.convolve(pixels, kernel_y)

        G = np.sqrt(Gx**2 + Gy**2)

        if G.max() > 0:
            G = G / G.max() * 255 

        self.processed_image = Image.fromarray(G.astype(np.uint8))
        self.display_image(self.processed_image, self.middle_label)

        print(f"{type} applied")


    # CUSTOM FILTER

    def apply_custom_filter(self):

        kernel = []
        
        # Pobieramy wartości z pól tekstowych
        for row in self.kernel_entries:
            kernel_row = []
            for entry in row:
                try:
                    value = float(entry.get())
                    kernel_row.append(value)
                except ValueError:
                    self.show_error_message("Proszę wprowadzić poprawne wartości liczbowe w macierzy!")
                    return
            kernel.append(kernel_row)
        
        # Sprawdzamy, czy użytkownik wypełnił przynajmniej jedno pole
        if not any(any(value != 0 for value in row) for row in kernel):
            self.show_error_message("Proszę wprowadzić przynajmniej jedno pole z wartością różną od zera!")
        else:
            pixels = np.array(self.processed_image)
            kernel = np.array(kernel)
            modified = self.convolve(pixels, kernel)
            self.processed_image = Image.fromarray(modified)
            self.display_image(self.processed_image, self.middle_label)
            print("Zastosowano filtr:", kernel) 


    """ PLOTTING """

    
    def prepare_plot_data(self):

        pixels = np.array(self.processed_image)
        grey = np.mean(pixels, axis=2).astype(np.uint8)
        vertical = np.zeros(pixels.shape[0])
        horizontal = np.zeros(pixels.shape[1])
        R = np.zeros(256)
        G = np.zeros(256)
        B = np.zeros(256)
        K = np.zeros(256)

        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                vertical[i] += grey[i, j]
                horizontal[j] += grey[i, j]
                R[pixels[i, j, 0]] += 1
                G[pixels[i, j, 1]] += 1
                B[pixels[i, j, 2]] += 1
                K[grey[i, j]] += 1

        return R, G, B, K, vertical, horizontal

    def prepare_plotsdasdadadfe(self):

        if self.processed_image is None:
            return

        R, G, B, K, vertical, horizontal = self.prepare_plot_data()

        # Create a figure for each plot
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig3, ax3 = plt.subplots(figsize=(5, 5))

        # Plot 1: Histogram
        x = np.arange(256)
        ax1.plot(x, R, color='red')
        ax1.plot(x, G, color='green')
        ax1.plot(x, B, color='blue')
        ax1.plot(x, y, color='gray')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='darkgray')
        ax1.spines['bottom'].set_color('darkgray')
        ax1.spines['left'].set_color('darkgray')

        # Plot 2: Horizontal intensity
        x = np.arange(len(horizontal))
        y = horizontal
        ax2.plot(x, y, color='gray')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='darkgray')
        ax2.spines['bottom'].set_color('darkgray')
        ax2.spines['left'].set_color('darkgray')

        # Plot 3: Vertical intensity
        x = vertical
        y = np.arange(len(vertical))
        ax3.plot(x, y, color='gray')
        ax3.set_facecolor('black')
        ax3.tick_params(colors='darkgray')
        ax3.spines['bottom'].set_color('darkgray')
        ax3.spines['left'].set_color('darkgray')

        # Display the plots on separate canvases
        for widget in self.right_frame.winfo_children():
            widget.destroy()  # Clear previous plots

        # Temporary canvas names
        canvas1 = FigureCanvasTkAgg(fig1, master=self.right_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="top", fill='both', expand=True)

        canvas2 = FigureCanvasTkAgg(fig2, master=self.right_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="top", fill='both', expand=True)

        canvas3 = FigureCanvasTkAgg(fig3, master=self.right_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(side="top", fill='both', expand=True)

    
    def prepare_plots(self):
        if self.processed_image is None:
            return

        R, G, B, K, vertical, horizontal = self.prepare_plot_data()

        # Create a single figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed
        fig.patch.set_facecolor('black')  # Set the background color of the figure

        # Plot 1: Histogram
        x = np.arange(256)
        axes[0].plot(x, R, color='red')
        axes[0].plot(x, G, color='green')
        axes[0].plot(x, B, color='blue')
        axes[0].plot(x, K, color='gray')
        axes[0].set_facecolor('black')
        axes[0].tick_params(colors='darkgray')
        axes[0].spines['bottom'].set_color('darkgray')
        axes[0].spines['left'].set_color('darkgray')

        # Plot 2: Horizontal intensity
        x = np.arange(len(horizontal))
        axes[1].plot(x, horizontal, color='gray')
        axes[1].set_facecolor('black')
        axes[1].tick_params(colors='darkgray')
        axes[1].spines['bottom'].set_color('darkgray')
        axes[1].spines['left'].set_color('darkgray')

        # Plot 3: Vertical intensity
        y = np.arange(len(vertical))
        axes[2].plot(vertical, y, color='gray')
        axes[2].set_facecolor('black')
        axes[2].tick_params(colors='darkgray')
        axes[2].spines['bottom'].set_color('darkgray')
        axes[2].spines['left'].set_color('darkgray')

        # Clear previous plots in the right frame
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        # Embed the figure in the right frame
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageProcessorApp(root)
    root.mainloop()