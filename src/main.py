from ai.agents import get_agent
from ai.doc_manager import store_pdf_doc, store_website
from ai.store import RESUME

import getpass
import os
import dotenv

dotenv.load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


def main():
    collection_name = RESUME
    doc_ids = store_pdf_doc(
        "./resume-darshan-rander.pdf", collection_name, data_id="resume"
    )
    doc_ids = store_website("https://blog.darshanrander.com/", data_id="blog")
    print(doc_ids)

    agent = get_agent()

    query = "Give summary of his blog posts and try to find any interesting insights about his experience and skills."
    for events in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        context={
            "admin_name": "Darshan Rander",
            "blog": "https://blog.darshanrander.com",
            "portfolio": "https://darshanrander.com",
        },
        config={"configurable": {"thread_id": "test_thread"}},
    ):
        events["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
