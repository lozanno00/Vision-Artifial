import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def train_images():
    # Lógica de entrenamiento aquí
    status_label.config(text="Entrenando imágenes...")

def select_images():
    files = filedialog.askopenfilenames(title="Selecciona imágenes")
    status_label.config(text=f"{len(files)} imágenes seleccionadas.")

def show_prediction_diagram():
    # Aquí puedes mostrar una imagen o dibujar un diagrama simple
    diagram_canvas.delete("all")
    diagram_canvas.create_rectangle(50, 50, 250, 150, fill="lightblue")
    diagram_canvas.create_text(150, 80, text="Predicción", font=("Arial", 14))
    diagram_canvas.create_text(150, 120, text="Acierto: 92%", font=("Arial", 12))

root = tk.Tk()
root.title("Vision Artificial")

# Frame para la gráfica
frame_graph = tk.LabelFrame(root, text="Gráfica", padx=10, pady=10)
frame_graph.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

fig, ax = plt.subplots(figsize=(4,3))
ax.plot([1,2,3,4], [10,20,25,30])
ax.set_title("Ejemplo de Gráfica")
canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas.draw()
canvas.get_tk_widget().pack()

# Frame para entrenamiento de imágenes
frame_train = tk.LabelFrame(root, text="Entrenamiento de Imágenes", padx=10, pady=10)
frame_train.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

select_btn = tk.Button(frame_train, text="Seleccionar Imágenes", command=select_images)
select_btn.pack(pady=5)

train_btn = tk.Button(frame_train, text="Entrenar", command=train_images)
train_btn.pack(pady=5)

status_label = tk.Label(frame_train, text="Esperando acción...")
status_label.pack(pady=5)

# Frame para diagrama de predicción
frame_diagram = tk.LabelFrame(root, text="Diagrama de Predicción", padx=10, pady=10)
frame_diagram.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

diagram_canvas = tk.Canvas(frame_diagram, width=300, height=200, bg="white")
diagram_canvas.pack()
show_prediction_diagram()

root.mainloop()
