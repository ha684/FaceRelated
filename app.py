import gradio as gr
import cv2
import threading
import queue
from source._clip.functionality import Feature

frame_queue = queue.Queue(maxsize=2)
stop = threading.Event()
feature = Feature()
def capture_frames():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
    cap.release()

def recognize(name_input):
    if not frame_queue.empty():
        frame = frame_queue.get()
        crop, message = feature.process_frame(frame, name_input)
        if message.startswith('Hello'):
            look_like_message = feature.look_like(crop)
            message_last = message + ', ' +  look_like_message
        else:
            message_last = message
        return message_last
    return "No frame available, turn on the camera"

def start_cam():
    stop.clear()
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    return 'Camera started'

def stop_cam():
    cap.release()
    return 'Camera stopped'

def update_frame():
    if not frame_queue.empty():
        frame = frame_queue.get()
        return frame
    
with gr.Blocks() as demo:
    with gr.Tabs():
            with gr.TabItem("Section 1"):
                gr.Markdown("#  CLIP Recognizing Frame")
                with gr.Row():
                    with gr.Column():
                        text_output = gr.Textbox(label="Output")
                        name_input = gr.Textbox(label="Enter name")
                        recognizer = gr.Button("Recognize")
                        remover = gr.Button("Remove name")
                        clearer = gr.Button("Clear")
                    with gr.Column():
                        start_camm = gr.Button('Open Camera')
                        stop_camm = gr.Button("Stop Camera")
                        camera_feed = gr.Image(image_mode='RGB', sources="webcam",streaming=True)

                start_camm.click(start_cam, inputs=None, outputs=text_output)
                recognizer.click(recognize, inputs=[name_input], outputs=text_output)
                remover.click(feature.remove, inputs=[name_input], outputs=text_output)
                clearer.click(feature.clear, inputs=None, outputs=text_output)
                stop_camm.click(stop_cam, inputs=None, outputs=text_output)
                demo.load(update_frame, outputs=[camera_feed], every=0.1)
            
            with gr.TabItem("Section 2"):
                gr.Markdown("# CLIP Image to Text")
                with gr.Row():
                    with gr.Column():
                        output = gr.Textbox(label="Output")
                        button = gr.Button("Process")
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image here")
                button.click(feature.animal,inputs=image_input,outputs=output)
            
            with gr.TabItem("Section 3"):
                gr.Markdown("# CLIP Text to Image")
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(label="Enter Text", placeholder='e.g., a lion')
                        button = gr.Button("Process")
                    with gr.Column():
                        image_output = gr.Image(label="Best Matching Image")
                button.click(feature.animal_im,inputs=text_input,outputs=image_output)
                
demo.launch(debug=True)
