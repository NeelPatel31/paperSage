import gradio as gr

# Gradio application setup
def create_demo():
    with gr.Blocks(title="paperSage: A RAG based Chatbot", theme="Soft") as demo:
        gr.Markdown("# paperSage: A RAG based Chatbot\nAsk questions about your PDF documents. 🚀")
        gr.Markdown("\nNote: The embedding model is running on CPU, so the processing time depends on the number of pages in the PDF.")
        
        with gr.Row():
            with gr.Column(scale=0.35):
                show_img = gr.Image(label='Overview', height=680)

            with gr.Column(scale=0.65):
                chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=680)

        with gr.Row():
                uploaded_pdf = gr.UploadButton("Upload PDF 📁", file_types=[".pdf"], file_count = "single", scale = 1)
                uploaded_pdf.upload(show_progress="full")

                text_input = gr.Textbox(
                    show_label = False,
                    placeholder = "Type here to ask your PDF",
                    container=False,
                    scale = 3,
                    max_lines = 3,
                    elem_id = "textbox")

                submit_button = gr.Button('Send 💬', elem_id="submit_button", scale = 0.5)

                clear_button = gr.Button('Clear Chat 🗑️', elem_id="clear_button", scale = 0.5)
                clear_button.click(lambda: [], None, chat_history)

        return demo, chat_history, show_img, text_input, submit_button, uploaded_pdf

if __name__ == '__main__':
    demo, chatbot, show_img, text_input, submit_button, uploaded_pdf = create_demo()
    demo.queue()
    demo.launch(debug=True)
