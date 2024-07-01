import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

# Function to upload image
def upload_image():
    global img, img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    if img_path:
        img = cv2.imread(img_path)
        display_image(img)

# Function to display image in the GUI
def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function to detect faces
def detect_faces():
    if img_path:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        display_image(img)
        messagebox.showinfo("Face Detection Result", f"Detected {len(faces)} face(s).")
    else:
        messagebox.showerror("Error", "Please upload an image first.")

# Function to detect objects using pre-trained MobileNet SSD model
def detect_objects():
    if img_path:
        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
        class_names = {15: 'person', 7: 'car'}
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if idx in class_names:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"{class_names[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        display_image(img)
        messagebox.showinfo("Object Detection Result", "Object detection completed.")
    else:
        messagebox.showerror("Error", "Please upload an image first.")

# Function to save the processed image
def save_image():
    if img_path:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("BMP files", "*.bmp")])
        if save_path:
            cv2.imwrite(save_path, img)
            messagebox.showinfo("Image Saved", f"Image saved to {save_path}")
    else:
        messagebox.showerror("Error", "Please upload and process an image first.")

# Initialize Tkinter window
root = tk.Tk()
root.title("Image Recognition App")
root.geometry("800x600")

# Create and place widgets
upload_btn = Button(root, text="Upload Image", command=upload_image, width=20, bg="lightblue")
upload_btn.pack(pady=10)

face_btn = Button(root, text="Detect Faces", command=detect_faces, width=20, bg="lightgreen")
face_btn.pack(pady=10)

object_btn = Button(root, text="Detect Objects", command=detect_objects, width=20, bg="lightyellow")
object_btn.pack(pady=10)

save_btn = Button(root, text="Save Image", command=save_image, width=20, bg="lightcoral")
save_btn.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
