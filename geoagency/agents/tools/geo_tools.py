from smolagents import tool


@tool 
def geocode_query(query: str) -> dict:
    """
    Can generate a BoundingBox for a query. 
    Only use this tool if the search results you found include a geometry_geojson attribute.
    Only use this tool if the query includes a location name such as a city, country, etc.
    
    Args:
        query: The query used as input to the geocoding service.
    """ 
    import requests
    
    geocoding_endpoint = "https://photon.komoot.io/api/"
    
    params = {'q': query}

    call = requests.get(geocoding_endpoint,params=params)
    if call.status_code == 200:
        top_3_candidates = call.json().get('features')[:3]
        return f"""Here are some geocoding results for your query: {top_3_candidates}
    Use these bounding boxes to compare it with the BoundingBoxes you find in the search results `geometry_geojson` attribute."""
    else:
        return "No BoundingBox could be derived"
