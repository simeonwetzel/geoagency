from smolagents import DuckDuckGoSearchTool, tools


@tool
def get_data_from_nfdi(query: str) -> dict:
    """
    Get data from NFDI4Earth the repository for Earth System Sciences related datasets.

    Args: 
        query (str): The query to search for.

    Returns:
        dict: The data from NFDI4Earth.
    """
    BASE_URL = "https://onestop4all.nfdi4earth.de/solr/metadata/"


    # Query parameters
    params = {
        "ident": "true",
        "q.op": "OR",
        "defType": "edismax",
        "bq": "isEdutrain:true^1000",
        "q": "climate adaptation",
        "qf": "title^1 keyword^50 collector",
        "fl": "*, [child author]",
        "fq": [
            "type:\"http://www.w3.org/ns/dcat#Dataset\"",
            "type:(\"http://www.w3.org/ns/dcat#Dataset\")"
        ],
        "rows": 20,
        "start": 0,
        "facet": "true",
        "facet.field": [
            "type",
            "subjectArea_str",
            "dataAccessType",
            "contentType_str",
            "dataUploadType",
            "supportsMetadataStandard_str",
            "software_license_str",
            "assignsIdentifierScheme"
        ],
        "facet.range": "datePublished",
        "facet.range.start": "2000-01-01T00:00:00Z",
        "facet.range.end": "2024-01-01T00:00:00Z",
        "facet.range.gap": "+1YEAR"
    }
    
    return {}