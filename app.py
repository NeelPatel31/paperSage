from interface import create_demo
from papersage import paperSage

demo, chat_history, show_img, txt_input, submit_button, uploaded_pdf = create_demo()

paperSageBot = paperSage()

with demo:
    print("[INFO] inside demo of app.py")
    uploaded_pdf.upload(paperSageBot.open_pdf, inputs=[uploaded_pdf, chat_history], outputs=[show_img, chat_history]).\
        success(paperSageBot.process_pdf_wrapper, inputs=[chat_history], outputs=[chat_history])

    txt_input.submit(paperSageBot.manage_history, inputs=[chat_history, txt_input], outputs=[chat_history], queue=False).\
        success(paperSageBot.ask_document, inputs=[chat_history, txt_input, uploaded_pdf], outputs=[chat_history, txt_input]).\
        success(paperSageBot.get_page, inputs=[], outputs=[show_img])

    submit_button.click(paperSageBot.manage_history, inputs=[chat_history, txt_input], outputs=[chat_history], queue=False).\
        success(paperSageBot.ask_document, inputs=[chat_history, txt_input, uploaded_pdf], outputs=[chat_history, txt_input]).\
        success(paperSageBot.get_page, inputs=[], outputs=[show_img])

if __name__ == "__main__":
    print("[INFO] App started")
    demo.launch(debug=True)