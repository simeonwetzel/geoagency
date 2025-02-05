from smolagents import tool

@tool
def overpass_tool(query: str) -> dict:
  """
  Can retrieve OSM data using the Overpass API.

  Args:
    query: Needs to be a query in Overpass QL.

  Returns:
    A dict representation of the OSM data.
  """
  import requests
  overpass_url = "https://overpass-api.de/api/interpreter"

  response = requests.get(overpass_url, params={'data': query})
  if response.status_code == 200:
    return response.json()
  else:
    print(f"Request failed with status code {response.status_code}")
    
    
    
import pandas as pd
def select_by_name_from_OSM_data(df: pd.DataFrame, target_value: str) -> dict:
  """
  Select a row from a pandas DataFrame by name or alt_name stored in the tags dictionary.

  Args:
      df (pd.DataFrame): The DataFrame to search in.
      target_value (str): The name or alt_name to search for.

  Returns:
      dict: A dict representation of the selected row.
  """
  filtered_df = df[df['tags'].apply(lambda x: x.get('name') == target_value or x.get('alt_name') == target_value)]

  return filtered_df.to_dict()


def create_overpass_query_for_single_OSM_feature(osm_feature: dict) -> str:
  """
  Create an Overpass query for a single OSM feature.

  Args:
      osm_feature (dict): A dict representation of an OSM feature.

  Returns:
      str: An Overpass query string.
  """
  print(f"osm_feature: {osm_feature}")
  feature_type = next(iter(osm_feature.get('type').values()))
  _id = next(iter(osm_feature.get('id').values()))

  return f"[out:json];{feature_type}({_id});(._; >;);out body;"


def overpass_to_geojson(overpass_data, feature_type='polygon'):
    """
    Convert Overpass API JSON output to a GeoJSON feature.

    Args:
        overpass_data (dict): The JSON response from Overpass API.
        feature_type (str): The type of GeoJSON feature to generate ('point', 'line', 'polygon').

    Returns:
        dict: A GeoJSON dictionary.
    """
    if "elements" not in overpass_data:
        raise ValueError("Invalid Overpass data: 'elements' key not found")

    if feature_type == 'point':
        # Find the first node element
        node = next((element for element in overpass_data['elements'] if element['type'] == 'node'), None)

        if not node:
            raise ValueError("No 'node' element found for point feature")

        # Create GeoJSON Point feature
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [node['lon'], node['lat']]
                    },
                    "properties": node.get('tags', {})
                }
            ]
        }
        return geojson

    elif feature_type == 'line':
        # Find the way element and its node references
        way = next((element for element in overpass_data['elements'] if element['type'] == 'way'), None)

        if not way or "nodes" not in way:
            raise ValueError("No 'way' element with nodes found for line feature")

        # Create a lookup table for node coordinates by ID
        nodes = {element['id']: (element['lon'], element['lat'])
                 for element in overpass_data['elements']
                 if element['type'] == 'node'}

        # Construct the line coordinates from node references
        coordinates = [nodes[node_id] for node_id in way['nodes']]

        # Create GeoJSON LineString feature
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": way.get('tags', {})
                }
            ]
        }
        return geojson

    elif feature_type == 'polygon':
        # Find the way element and its node references
        way = next((element for element in overpass_data['elements'] if element['type'] == 'way'), None)

        if not way or "nodes" not in way:
            raise ValueError("No 'way' element with nodes found for polygon feature")

        # Create a lookup table for node coordinates by ID
        nodes = {element['id']: (element['lon'], element['lat'])
                 for element in overpass_data['elements']
                 if element['type'] == 'node'}

        # Construct the polygon coordinates from node references
        coordinates = [nodes[node_id] for node_id in way['nodes']]

        # Ensure the polygon is closed (first and last points must be the same)
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        # Create GeoJSON Polygon feature
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates]
                    },
                    "properties": way.get('tags', {})
                }
            ]
        }
        return geojson

    else:
        raise ValueError("Invalid feature_type. Supported types are 'point', 'line', and 'polygon'")

@tool
def get_osm_feature_as_geojson_by_name(osm_data: dict, target_name: str, feature_type: str) -> str:
  """
  Retrieves a specific OSM feature by name and returns its GeoJSON representation.

  SYSTEM NOTE: Only use this tool if a single feature is requested (never for a larger amount of features)!

  This function searches for a feature within the provided OSM data (typically
  from the Overpass API) matching the given target name. It then constructs
  an Overpass query to fetch detailed information for that feature, converts
  the result into a GeoJSON polygon, and returns the GeoJSON as a string.

  Args:
      osm_data: A dictionary containing OSM data, usually the JSON response
                from an Overpass API query.  The data should
                contain an 'elements' key.
      target_name: The name of the OSM feature to search for. The function
                will attempt to match this name against the 'name'
                or 'alt_name' keys within the 'tags' dictionary of
                each element in the 'elements' list.
      feature_type: The type of GeoJSON feature to generate ('point', 'line', 'polygon').


  Returns:
      str: A GeoJSON string representation of the selected OSM feature, or None
           if the feature is not found or an error occurs during processing.
           The GeoJSON will describe a polygon.

  Raises:
      ValueError: If the input 'osm_data' is invalid (e.g., missing 'elements' key),
                  or if no matching 'way' element with nodes is found in the
                  Overpass API response.
  """
  df = pd.DataFrame(osm_data['elements'])
  filtered_df = df[df['tags'].notna()]
  selected_row = select_by_name_from_OSM_data(filtered_df, target_name)

  if selected_row.get('id'):
    gen_query = create_overpass_query_for_single_OSM_feature(selected_row)

    overpass_data = overpass_tool(gen_query)

    return overpass_to_geojson(overpass_data, feature_type)

  else:
    print("No matching feature found")
    return