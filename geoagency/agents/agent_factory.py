from smolagents import CodeAgent, tool, Tool
from geoagency.llm_manager import LLMManager
from smolagents.prompts import MANAGED_AGENT_PROMPT, CODE_SYSTEM_PROMPT 
from geoagency.agents.tools.retrieval_tools import MetadataRetrieverTool
from geoagency.agents.retriever.retriever import RepoRetriever
from geoagency.agents.tools.geo_tools import geocode_query, is_query_bbox_within_document
# Specify the model to be used 
llm = LLMManager.get_llm()
repo_retriever = RepoRetriever()
verbose = True

# comment this to show detailed output
#from smolagents.utils import console
#console.quiet = True

# Generic Agent factory
def create_agent(name, description, tools, model, additional_authorized_imports=[], system_message=None):
    """Create an agent that uses a given list of tools according to its system message."""
    if system_message is None:
        system_message = CODE_SYSTEM_PROMPT
    else:
        system_message = CODE_SYSTEM_PROMPT + "\n" + system_message
        
    agent = CodeAgent(tools=tools, 
                      model=model, 
                      system_prompt=system_message,
                      additional_authorized_imports=additional_authorized_imports)
    agent.name = name
    agent.description = description
    agent.logger.level = 1

    return agent

def agent_factory(*args, **kwargs):
    def create_instance():
        return create_agent(*args, **kwargs)
    return create_instance


metadata_retriever_agent_factory = agent_factory(
    name="metadata-retriever-assistant",
    description="Retrieval assistant who can search for datasets and related metadata using different repositories.",
    system_message="""You will be tasked to search for datasets and provide a summary of the retrieved metadata. 
    Provide an overwiew of the top hits. 
    If necessary ask a follow-up before conducting a search.""",
    tools=[MetadataRetrieverTool(repo_retriever), clarify_search_criteria],
    model=llm,
    additional_authorized_imports=["asyncio"],
)

system_message_bck="""You will be tasked to evaluate the search results. 
    This is useful to see if the result datasets have a spatial coverage that is relevant for the query.
        
    # Proceed like the following: 
    (1) For a query you can parse the geometry using the `geocode_query` tool. 
    (2)
        a) If the search results have a geometry propertey as GeoJSON <string> => Only then use the `is_query_bbox_within_document` tool.
        b) If the search results have no geometry property => than you can try to determine the spatial context by the metadata body. 
    """
    
spatial_relevance_agent_factory = agent_factory(
    name="spatial-relevance-assistant",
    description="Assistant to evaluate the geospatial relevance of search results (retrieved metadata or datasets)",
    system_message="You can use the metadata of a search to evaluate if the results spatial coverage fits the query",
    tools=[geocode_query],
    model=llm,
    additional_authorized_imports=["shapely, json, requests"],
)

# Setup a scheduler

team = [metadata_retriever_agent_factory, spatial_relevance_agent_factory]

@tool
def distribute_sub_task(task_description:str, assistant:str)->str:
    """Prompt an assistant to solve a certain task. 
    
    Args:
        task_description: Detailed task description, to make sure to provide all necessary details. When handling text, hand over the complete original text, unmodified, containing all line-breaks, headlines, etc.
        assistant: name of the assistant that should take care of the task.
    """
    for t_factory in team:
        t = t_factory()
        if t.name == assistant:
            if verbose:
                print("".join(["-"]*80))
                short = task_description[:1000]
                num_chars = len(task_description)
                print(f"| I am asking {assistant} to take care of: {short}...[{num_chars} chars]")

            # execute the task
            result = t.run(task_description)
            
            if verbose:
                short = result[:1000]
                num_chars = len(result)
                print(f"| Response was: {short}...[{num_chars} chars]")
                print("".join(["-"]*80))

            return result

    return "Assistant unknown"

team_description = "\n".join([f"* {t().name}: {t().description}" for t in team])


scheduler = create_agent(
    name="scheduler",
    tools=[distribute_sub_task],
    description="Scheduler splits tasks into sub-tasks and distributes them.",
    system_message=f"""
You are an manager who has a team of assistants. Your task is to find relevant datasets together with your team.

# Important: Be as fast as possible! You need to perform very quick. 

# Team
Your assistants can either search for data and metadata and for you, 
or evaluate the relevance of the found datasets.
It can also give you a hint if you need to ask for more search criteria.  

Your team members are:
{team_description}

# Typical workflow
A typical workflow is like this:
* Retrieve datasets
* As soon as a first set of metadata is retrieved you stop. Never do a search or similar (only the retrievers in tools of team members).  
* Evaluate their usefulness for the given query
* Show the user a stuctured set of top candidates and a final answer describing what you found out
* Then immediatly stop 

# Hints
When distributing tasks, make sure to provide all necessary details to the assistants. 

# Your task
Distribute tasks to your team. Goal is to have a great set of datasets references that the user can download later for his/her use case.
""",
    model=llm
)

def call_agent(query: str) -> str:
    # return manager_agent.run(query)   
    preprocessed_query = f"""Task find relevant datasets for this query: {query}"""
    answer = scheduler.run(preprocessed_query)
    return answer