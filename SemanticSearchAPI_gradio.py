import gradio as gr

from SemanticSearchUtilsFunction import *


def haystack_pipeline(pdf_file):
    DATA_INDEX = get_data_haystack_format(pdf_file)
    DOC_STORE = get_faiss_document_store(DATA_INDEX)
    RETRIEVER = get_retriever(DOC_STORE)
    DOC_STORE.update_embeddings(RETRIEVER)
    PIPELINE = get_pipeline(RETRIEVER)
    return PIPELINE


def predict(query, pdf_file):
    pipe = haystack_pipeline(pdf_file)
    result = pipe.run(query=query,
                      params={"Retriever": {"top_k": 3}})
    print(result)
    return [(x.to_dict()['content'], x.to_dict()['score']) for x in result["documents"]]


title = "Case Management Intelligent Search System"
description = """
<center>Sample Questions: What are potential duplicates? </center>
"""
iface = gr.Interface(fn=predict,
                     inputs=[gr.inputs.Textbox(lines=3, label='Ask a query!'),
                             gr.inputs.File(file_count="single", type="file", label="Upload a pdf"),
                             ],
                     outputs="text",
                     title=title, description=description,
                     interpretation="default",
                     theme="dark-grass",
                     server_name="0.0.0.0"
                     )
iface.launch()
