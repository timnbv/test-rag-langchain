def create_db(name, from_file_path):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    import os
    from langchain_chroma import Chroma
    db_name = f"./{name}"

    if not os.path.exists(db_name):
        print(f"*****Creating database {db_name}*******")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        loader = PyPDFLoader(from_file_path)
        pages = loader.load_and_split()
        Chroma.from_documents(pages, embeddings, persist_directory=db_name)


def ask_llm(question):
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser

    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know". Return result with Markdown formatting.

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = Ollama(model="llama3")

    parser = StrOutputParser()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    chain = (
        {"context": chroma.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser

    )
    return chain.invoke(question)
