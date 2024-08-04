import textwrap
import torch
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer


def text_formatter(text: str) -> str:
    """
    Performs minor formatting on text.
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sub-lists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def print_wrapped(text: set,
                  wrap_width: int = 80) -> None:
    """
    Print the text as per the wrap_width.
    """
    wrapped_text = textwrap.fill(text, wrap_width)
    print(wrapped_text)


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                embeddings_length: int,
                                n_resources_to_return: int = 5,
                                print_time: bool = True) -> dict:
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
    except Exception as e:
        print(f"[ERROR] Error converting the query into embeddings: {str(e)}")
        return {
            "status": False,
            "error": str(e)
        }

    # Get dot scores
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)}) embeddings: {end_time - start_time :.5f} seconds.\n")

    n_resources_to_return = min(n_resources_to_return, embeddings_length)
    if embeddings_length == 0:
        print("[WARNING] No embeddings available.")
        return None, None
        
    try:
        scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    except Exception as e:
        print(f"[ERROR] Error getting top k scores: {str(e)}")
        return {
            "status": False,
            "error": str(e)
        }
    # scores, indices = torch.topk(dot_scores, k=n_resources_to_return)

    return {
        "status": True,
        "data":[scores, indices]
    }


def get_top_results_scores_pg_num(query: str,
                                  embeddings: torch.tensor,
                                  pages_and_chunks: list[dict],
                                  n_resources_to_return: int = 5):
    """
    Finds relevant passages given a query and fetches relevant page numbers.
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    page_number_array = []
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text: ")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"], 100)
        page_number_array.append(pages_and_chunks[idx]['page_number'])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

    return scores, indices, page_number_array


def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int = 5):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """

    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Query: '{query}'\n")
    print("Results: ")

    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text: ")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"], 100)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")


def generate_llm_response(client,
                          prompt: str,
                          llm_model_name: str = "gemma2-9b-it", 
                          temperature: int = 0.2) -> dict:
    """
    Generate a response from LLM.
    """
    try:    
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=llm_model_name,
            temperature=temperature
        )
        return {
            "status": True,
            "data" : chat_completion.choices[0].message.content
        }
    except Exception as e:
        print(f"[ERROR] Error in generating LLM response: {str(e)}")
        return {
            "status": False,
            "error": str(e)
        }

def prompt_formatter(query: str,
                     history: str,
                     context_item: list[dict]) -> str:
    context = "- " + "\n- ".join([item['sentence_chunk'] for item in context_item])

    base_prompt = """Based on the following context and the previous history, please answer the query. If the context 
    don't contain the answer then return "Sorry, but I don't have the information you're looking for at the moment." 
    Give yourself room to think by extracting relevant passages from the context before answering the query. Previous 
    history is a part of the communication done previously, it may contain details related to current query,so use this
    history wisely. Use the history if and only if the context is not useful. Don't return the thinking, only return the
    answer and the answer should be in simple text not in markup language. Make sure your answers are as explanatory as 
    possible. 
    
    Context: {context}
    
    History: {history}

    Query: {query}
    Answer:
    """
    
    base_prompt = base_prompt.format(query=query, history=history, context=context)
    print("[INFO] Base Prompt after formatting: ",base_prompt)

    return base_prompt

def get_summary_prompt(context: str) -> str:
    """
    Get the summary of the generated response to use it for future queries.
    """
    prompt = """
    Based on the following context, create a concise summary. Summary should me as short as possible without losing the information.
    You must insert any other details, use the context only. Give output in simple text only.
    
    Context: {context}
    """
    prompt = prompt.format(context=context)
    return prompt
