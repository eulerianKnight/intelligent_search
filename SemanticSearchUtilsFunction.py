from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PDFToTextConverter, PreProcessor
from haystack.pipelines import DocumentSearchPipeline


def get_data_haystack_format(filepath):
    """
    Convert PDF file to Haystack format
    :param: pdf_file_path
    :return:
    """
    converter = PDFToTextConverter(remove_numeric_tables=True,
                                   valid_languages=["en"])
    doc_pdf = converter.convert(file_path=filepath, meta=None)[0]
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=50,
        split_respect_sentence_boundary=True
    )
    docs_preprocessed = preprocessor.process([doc_pdf])
    return docs_preprocessed


def get_faiss_document_store(data_doc,
                             do_indexing=True):
    """
    Create FAISS document store
    :return:
    """
    doc_store = FAISSDocumentStore(faiss_index_factory_str="Flat",
                                   similarity='cosine',
                                   return_embedding=True)
    if do_indexing:
        try:
            doc_store.delete_all_documents()
            doc_store.write_documents(get_data_haystack_format(data_doc))
        except Exception as e:
            print(e)
    return doc_store


def get_retriever(doc_store):
    """
    Create retriever
    :return:
    """
    retriever = EmbeddingRetriever(document_store=doc_store,
                                   embedding_model='distilroberta-base-msmarco-v2',
                                   model_format='sentence_transformers')
    return retriever


def get_pipeline(retriever):
    """
    Create pipeline
    :return:
    """
    pipeline = DocumentSearchPipeline(retriever=retriever)
    return pipeline


def get_search_results(haystack_pipeline,
                       query,
                       top_k=3):
    """
     Get search results
     :param haystack_pipeline:
     :param query:
     :param top_k:
     :return:
     """
    results = haystack_pipeline.run(query=query,
                                    params={"Retriever": {"top_k": top_k}})
    return results


def print_answers(results):
    """
    Print answers
    :param results:
    :return:
    """
    return [(x.to_dict()['content'], x.to_dict()['score']) for x in results["documents"]]


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper

