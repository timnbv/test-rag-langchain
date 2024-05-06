def create_db(db_directory, from_file_path):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    import os
    from langchain_chroma import Chroma

    if not os.path.exists(db_directory):
        print(f"*****Started creating database {db_directory}*******")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        loader = PyPDFLoader(from_file_path)
        pages = loader.load_and_split()
        Chroma.from_documents(pages, embeddings, persist_directory=db_directory)
        print(f"*****Finished creating database {db_directory}*******")


def ask_llm(question, model="llama3", emb_model="nomic-embed-text", stream=False):
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser

    db_directory = "./chroma_db"

    create_db(db_directory, "docs/Understanding_LLMs.pdf")

    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know". Return result with Markdown formatting.

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = Ollama(model=model)

    parser = StrOutputParser()

    embeddings = OllamaEmbeddings(model=emb_model)
    chroma = Chroma(persist_directory=db_directory, embedding_function=embeddings)

    chain = (
        {"context": chroma.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser

    )
    if stream:
        return chain.stream(question)
    else:
        return chain.invoke(question)
