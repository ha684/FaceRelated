import gradio as gr
import cv2
import threading
import queue
from functionality import process_frame,remove,clear

frame_queue = queue.Queue(maxsize=2)
stop = threading.Event()

def capture_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

def recognize(name_input):
    if not frame_queue.empty():
        frame = frame_queue.get()
        processed_frame, message = process_frame(frame, name_input)
        return processed_frame, message
    return None, "No frame available"

def start_cam():
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    return 'Camera started'

def stop_cam():
    stop.set()
    return 'Camera stopped'

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Face Recognition System")
    with gr.Row():
        with gr.Column():
            start_camm = gr.Button('Open Camera')
            name_input = gr.Textbox(label="Enter name")
            recognizer = gr.Button("Recognize")
            remover = gr.Button("Remove name")
            clearer = gr.Button("Clear")
            stop_camm = gr.Button("Stop Camera")
        with gr.Column():
            camera_feed = gr.Image(image_mode='RGB', sources="webcam", streaming=True)
            text_output = gr.Textbox(label="Recognition Result")
    
    def update_frame():
        if not frame_queue.empty():
            frame = frame_queue.get()
            return frame
    start_camm.click(start_camm,inputs=None,outputs = text_output)
    recognizer.click(recognize, inputs=[name_input], outputs=[camera_feed, text_output])
    remover.click(remove,inputs = [name_input], outputs = text_output)
    clearer.click(clear,inputs = None, outputs = text_output)
    stop_camm.click(stop_camm,inputs=None, outputs=text_output)
    demo.load(update_frame, outputs=[camera_feed], every=0.05)

demo.launch(debug=True)
