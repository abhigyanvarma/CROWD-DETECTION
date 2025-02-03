from ultralytics import YOLO
import cv2
from tkinter import Tk, Button, Canvas, filedialog, Frame, Label
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import DBSCAN


class YOLOv8VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Video Detection with Clusters")
        self.root.configure(bg="#1e2a47") 

        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.main_frame = Frame(self.root, bg="#283747")
        self.main_frame.pack(fill="both", expand=True)

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.button_frame = Frame(self.main_frame, bg="#283747")
        self.button_frame.grid(row=1, column=0, pady=20)

        self.title_label = Label(self.main_frame, text="YOLOv8 Video Detection", font=("Arial", 20), fg="white", bg="#283747")
        self.title_label.grid(row=2, column=0, pady=10)

        self.load_button_image = self.create_rounded_button_image("#2980b9", "Load Video")
        self.quit_button_image = self.create_rounded_button_image("#c0392b", "Quit")

        self.load_button = Button(self.button_frame, image=self.load_button_image, command=self.load_video, borderwidth=0, relief="flat")
        self.load_button.grid(row=0, column=0, padx=20)

        self.quit_button = Button(self.button_frame, image=self.quit_button_image, command=root.destroy, borderwidth=0, relief="flat")
        self.quit_button.grid(row=0, column=1, padx=20)

        self.cap = None
        self.out = None
        self.frame = None
        self.model = YOLO("yolov8n.pt")  

    def create_rounded_button_image(self, color, text):
        width, height = 200, 60  
        button_image = Image.new("RGBA", (width, height), color)
        button_image = button_image.convert("RGBA")

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(button_image)
        font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text, font=font, fill="white")

        button_image = button_image.crop((10, 10, width-10, height-10))

        return ImageTk.PhotoImage(button_image)

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        self.out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

        self.process_video()

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.out.release()
            return

        results = self.model(frame)

        person_detections = []

        for box in results[0].boxes:
            class_id = int(box.cls)  
            confidence = float(box.conf)  
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
            if class_id == 0:
                person_detections.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Person {int(confidence * 100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if person_detections:
            points = []
            for x1, y1, x2, y2 in person_detections:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 
                points.append([cx, cy])

            dbscan = DBSCAN(eps=50, min_samples=3) 
            clusters = dbscan.fit(points)

            cluster_labels = clusters.labels_

            cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  
            risk_text = "Risk: Low"
            risk_color = (0, 255, 0)

            if cluster_count >= 3: 
                risk_text = "Risk: Medium"
                risk_color = (0, 255, 255)
            if cluster_count > 5: 
                risk_text = "Risk: High"
                risk_color = (0, 0, 255)

            cv2.putText(frame, risk_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, risk_color, 3)
            cv2.putText(frame, f"Cluster Count: {cluster_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue 
                cluster_points = [points[i] for i in range(len(points)) if cluster_labels[i] == cluster_id]
                x_coords, y_coords = zip(*cluster_points)

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
                cv2.putText(frame, f"Cluster {cluster_id + 1}: {len(cluster_points)}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.out.write(frame)

        frame_resized = cv2.resize(frame, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        self.root.after(10, self.process_video)


root = Tk()
app = YOLOv8VideoApp(root)
root.mainloop()







