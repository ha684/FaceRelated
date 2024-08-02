from functionality import extract_image_features,concat,capture_image,add,find_closest
import gradio as gr
import cv2


def create_result():
    cap = cv2.VideoCapture(0)
    while True:
        capture_image(cap)
    cap.release()
    cv2.destroyAllWindows()
hash = dict()
with gr.Blocks() as demo:
    run_button = gr.Button("Run Demo")
#     run_button.click(fn=create_result, inputs=input_image, outputs=[detected_output, recog_output, detect_time_output, recog_time_output, kie_output, kie_time_output])
    run_button.click(fn=create_result)

demo.launch(debug=True)        
            
