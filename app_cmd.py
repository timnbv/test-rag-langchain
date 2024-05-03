from my_lib import create_db, ask_llm

if __name__ == "__main__":
    while True:
        question = input("Enter your question or type \"exit\": ")
        if question.lower() == "exit":
            exit()

        print(ask_llm(question))
        print("\n")
