import fitz
import torch
import gradio as gr
from PIL import Image
from tqdm.auto import tqdm
from helper import text_formatter, split_list, prompt_formatter, retrieve_relevant_resources, generate_llm_response, get_summary_prompt
import pandas as pd
import numpy as np
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer
import textwrap
import os
from groq import Groq
from dotenv import load_dotenv, dotenv_values
from typing import IO
load_dotenv()


class paperSage:

    def __init__(self, env_path=".env"):
        
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.llm_client = Groq(api_key=self.GROQ_API_KEY)
        
        self.doc = None
        self.num_sentence_chunk_size = 10
        self.min_token_length = 30
        self.page_number = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model_name = "all-mpnet-base-v2"
        # self.embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name, device=self.device)
        self.embeddings_length = 0
        self.llm_model_name = "gemma2-9b-it"
        self.temperature = 0.2
        self.summary_history = ""
        self.processed = False

        
        self.pages_and_texts = None
        self.pages_and_chunks = None
        self.pages_and_chunks_over_min_token_len = None
        self.embedding_model = None
        self.text_chunks_and_embedding_df = None
        self.pages_and_chunks = None
        self.embeddings = None
        
        print("[INFO] inside chatpdf.py file")
        print(f"[INFO] Device: {self.device}")

    def get_page(self)-> Image.Image:
        """
        Returns the image of a page if the page_number is set else first page will be returned.
        """
        print(f"Page number: {self.page_number}")
        page = self.doc[self.page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

    def manage_history(self,
                       history: list[list[str]], 
                       text: str) -> list[list[str]]:
        """
            Add user-entered text to the chat history.

            Parameters:
                history (list): List of chat history tuples.
                text (str): User-entered text.

            Returns:
                list: Updated chat history.
            """
        if not text:
            raise gr.Error('Enter text')

        history.append((text, ''))
        return history

    def open_pdf(self, 
                 pdf_path: str, 
                 history: list[list[str]]) -> list:
        """
        Opens a PDF file, reads its text content page by page, and collects statistics.
        """
        try:
            doc = fitz.open(pdf_path)
            self.doc = doc
            gr.Info("Let me go through the PDF.")
            return self.get_page(), history
        except Exception as e:
            gr.Error(f"Error opening PDF: {str(e)}")
            return None, history

    def process_opened_pdf(self) -> None:
        """
        Split the pdf into pages and then converting the pages into sentences.
        """
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(self.doc),desc="Basic processing of the file."):  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            text = text_formatter(text)
            pages_and_texts.append(
                {"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                 "page_char_count": len(text),
                 "page_word_count": len(text.split(" ")),
                 "page_sentence_count_raw": len(text.split(". ")),
                 "page_token_count": len(text) / 4,
                 # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                 "text": text})

        nlp = English()
        nlp.add_pipe("sentencizer")

        for item in tqdm(pages_and_texts):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_count_spacy"] = len(item["sentences"])

            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                 slice_size=self.num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        self.pages_and_texts = pages_and_texts

    def process_doc_into_chunks(self) -> None:
        """
        Process the sentences into chunks and then create the embeddings of them.
        """
        pages_and_chunks = []

        for item in tqdm(self.pages_and_texts, desc="Chunking the document"):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                               joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                pages_and_chunks.append(chunk_dict)

        self.pages_and_chunks = pages_and_chunks
        df = pd.DataFrame(self.pages_and_chunks)
        self.pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > self.min_token_length].to_dict(
            orient="records")

        self.embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name, device=self.device)

        for item in tqdm(self.pages_and_chunks_over_min_token_len, desc="Encoding the document"):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

        text_chunks_and_embedding_df = pd.DataFrame(self.pages_and_chunks_over_min_token_len)
        print("============================================")
        # print(text_chunks_and_embedding_df.columns)
        print("============================================")
        print(text_chunks_and_embedding_df.head())
        print("============================================")
        # print(type(text_chunks_and_embedding_df['embedding'][0]))
        print("============================================")

        # text_chunks_and_embedding_df['embedding'] = text_chunks_and_embedding_df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        # embeddings = text_chunks_and_embedding_df['embedding'].tolist()
        # embeddings = np.stack(text_chunks_and_embedding_df['embedding'].tolist(), axis=0)
        # embeddings = np.stack(text_chunks_and_embedding_df['embedding'].tolist(), axis=0)
        self.embeddings = torch.tensor(text_chunks_and_embedding_df['embedding'].tolist(), dtype=torch.float32).to(self.device)
        self.embeddings_length = len(self.embeddings)

    def process_pdf_wrapper(self,history: list[list[str]]) -> list[list[str]]:
        """
        Main function that calls the different pdf functions sequentially.
        """
        self.process_opened_pdf()
        self.process_doc_into_chunks()
        self.processed = True
        history.append(('','Thanks for waiting. You can now ask questions related to the PDF.'))
        # gr.Info("Thanks for waiting.")
        return history


    def ask_document(self,
                     history: list[list[str]],
                     query: str,
                     file: list[IO]) -> list:
        """ 
        Takes a query, finds relevant resources/context and generates an answer to the query \
        based on the relevant resources.
        """

        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_pdf_wrapper()
            print("[INFO] File is processed.")

        print(f"[INFO] Query: {query}")
        
        # Retrieval
        retrieve_relevant_resources_res = retrieve_relevant_resources(query=query,
                                                      embeddings=self.embeddings,
                                                      model=self.embedding_model,
                                                      embeddings_length = self.embeddings_length)
        
        if (retrieve_relevant_resources_res['status'] == False):
            gr.Error(retrieve_relevant_resources_res['error'])
        
        scores, indices = retrieve_relevant_resources_res['data']

        
        context_item = [self.pages_and_chunks[i].copy() for i in indices]
        self.page_number = self.pages_and_chunks[indices[0]]['page_number']
        
        # remove embeddings
        for item in context_item:
            item.pop('embedding')

        for i, item in enumerate(context_item):
            item['score'] = scores[i]

        print(scores)
        
        # Augmentation
        prompt = prompt_formatter(query=query, history=self.summary_history, context_item=context_item)

        # Generation
        llm_output = generate_llm_response(self.llm_client, prompt, self.llm_model_name, self.temperature)
        
        if (llm_output['status'] == False):
            gr.Error("Error while generating the response from LLM.")
            return history, " "

        output_text = llm_output['data']
        output_text = output_text.replace(prompt, '')

        summary_prompt = get_summary_prompt(output_text)
        summary_llm_output = generate_llm_response(self.llm_client, summary_prompt, self.llm_model_name, self.temperature)
        
        if (summary_llm_output['status'] == False):
            gr.Error("Error while generating the summary from LLM.")
            return history, " "
        
        self.summary_history = summary_llm_output['data']

        print(f"___________________________________________________________\n{self.summary_history}\n___________________________________________________________")
        
        for char in output_text:
            history[-1][-1] += char
        
        return history, " "