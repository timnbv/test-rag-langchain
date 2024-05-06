from my_lib import ask_llm


def main():
    from halo import Halo
    default_question = "What is transformer?"
    while True:
        question = input("Enter your question or type \"exit\": ")
        if question.lower() == "exit":
            exit()
        result = None
        if not question:
            print(f"\n*** Answering default question: {default_question}***")
        with Halo(text='Processing your question...', spinner='dots') as spinner:
            result = ask_llm(question or default_question, stream=True)
            chunk = next(result)
            spinner.succeed("Your result is ready:\n")
        print(chunk, end="")
        for chunk in result:
            print(chunk, end="")
        print("\n")


if __name__ == "__main__":
    main()
