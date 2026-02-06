from ai.agents import get_agent
from ai.doc_manager import store_pdf_doc
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
    print(doc_ids)

    agent = get_agent()

    query = "What are Darshan's skills in backend development?"
    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode=["updates", "custom"],
        context={"admin_name": "Darshan Rander"},
    ):
        print(f"stream_mode: {mode}")
        print(f"content: {chunk}")
        print("\n")


if __name__ == "__main__":
    main()
