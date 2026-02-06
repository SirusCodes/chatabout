from typing import Literal, TypedDict
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field

from ai.store import RESUME, WEBSITE, get_vector_store


class Context(TypedDict):
    admin_name: str
    blog: str
    portfolio: str


type SiteType = Literal["blog", "portfolio"]


class ResumeDataTool(BaseModel):
    """Get data from the admin's resume."""

    query: str = Field(description="The query to get data from the resume.")


@tool(args_schema=ResumeDataTool, response_format="content_and_artifact")
def get_data_from_resume(query: str, runtime: ToolRuntime[Context]):
    runtime.stream_writer("Getting data from resume.")

    store = get_vector_store(collection_name=RESUME)
    results = store.similarity_search(query, k=5)
    content = "\n\n".join([doc.page_content for doc in results])
    return content, results


class WebDataTool(BaseModel):
    """Get data from the admin provided website."""

    query: str = Field(description="The query to get data from the website.")
    site_type: SiteType = Field(description="The type of the website")


@tool(args_schema=WebDataTool, response_format="content_and_artifact")
def get_data_from_website(
    query: str, site_type: SiteType, runtime: ToolRuntime[Context]
):
    runtime.stream_writer(f"Getting data from {site_type}.")

    store = get_vector_store(collection_name=WEBSITE)
    results = store.similarity_search(query, k=5)

    content = "\n\n".join([doc.page_content for doc in results])
    return content, results


def get_all():
    return [
        get_data_from_website,
        get_data_from_resume,
    ]
