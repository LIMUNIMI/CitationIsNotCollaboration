from featgraph import jwebgraph
from featgraph.jwebgraph import start_jvm
from featgraph.pathutils import notisfile, notisglob, derived_paths
from featgraph import conversion
from functools import partial
import os
import jpype
import json

# start jvm
jvm_path = None
start_jvm(jvm_path=jvm_path)

import featgraph.jwebgraph.utils

# make conversion if not done yet
do_convert = False
adjacency_path = "Spotify_pickle/collaboration_network_edge_list.pickle"
metadata_path = "Spotify_pickle/artist_data.pickle"
dest_path = "graphs/spotify-2018"
if do_convert == True:
  spotipath = derived_paths(dest_path)
  os.makedirs(os.path.dirname(spotipath()), exist_ok=True)
  nnodes = conversion.make_ids_txt(spotipath("ids", "txt"), adjacency_path)
  conversion.make_metadata_txt(
      spotipath,
      metadata_path,
      spotipath("ids", "txt"),
  )
  conversion.make_asciigraph_txt(spotipath("graph-txt"), adjacency_path,
                                 spotipath("ids", "txt"))
  conversion.compress_to_bvgraph(spotipath())

# load graph and check if files exist
spotify_basename = "graphs/spotify-2018"
graph = jwebgraph.utils.BVGraph(spotify_basename)
graph.reconstruct_offsets()

for r in ("graph", "properties", "ids.txt"):
  if not os.path.isfile(graph.path(r)):
    raise FileNotFoundError(graph.path(r))

missing_value = -20


def popularity(missing_value=str(missing_value)):
  with open(graph.path("popularity", "txt"), "r") as f:
    return [float(r.rstrip("\n") or missing_value) for r in f]


def genre():
  with open(graph.path("genre", "txt"), "r") as f:
    return [json.loads(r.rstrip("\n")) for r in f]


def popularity_filter(graph, threshold):
  '''Filter the graph nodes using the check_func and setting the threshold'''
  pop_values = popularity(missing_value=-20)
  # popularity at index i is the same popularity at graph.artist(index=i).popularity - if last is None, pop is -20
  filtered_nodes = list(
      filter(lambda i: (pop_values[i] > threshold), range(len(pop_values))))
  return filtered_nodes


def genre_filter(graph, threshold, key):
  '''Filter the graph nodes using the check_func and setting the threshold'''
  genre_values = genre()
  if key == 'and':
    filtered_nodes = list(
        filter(lambda i: (all(x in genre_values[i] for x in threshold)),
               range(len(genre_values))))
  elif key == 'or':
    filtered_nodes = list(
        filter(lambda i: (any(x in genre_values[i] for x in threshold)),
               range(len(genre_values))))
  return filtered_nodes


def centrality(type):
  if type == "pagerank":
    c_values = graph.pagerank()
  elif type == "hc":
    #hc_rank = featgraph.load_as_doubles(graph.path("hc", "ranks"), "Float")
    c_values = graph.harmonicc()
  elif type == "closenessc":
    c_values = graph.closenessc()
  else:
    print("Unknown centrality")
  return c_values


def centrality_filter(graph, type, threshold):
  '''Filter the graph nodes using the check_func and setting the threshold'''
  c_values = centrality(type)
  filtered_nodes = list(
      filter(lambda i: (c_values[i] > threshold), range(len(c_values))))
  return filtered_nodes


def map_nodes(n, filtered_nodes):
  map = jpype.JInt[n]
  c = 0
  for i in range(n):
    if i not in filtered_nodes:
      map[i] = -1
    else:
      map[i] = c
      c += 1
  return map


def update_txt_metadata(src_path, key, type_filter, threshold, lines_to_keep):
  """Function for the creation of the metadata file for the filtered graph:

    Args:
        src_path: source path of the metadata file of the original graph
        key: the type of metadate you want to retrieve
        type_filter: the type of filter you used to filter the graph
        threshold: the threshold for type_filter you used to filter the graph
        lines_to_keep: the indices of the nodes you filtered and for which you want to retrieve the metadata
    """
  new_path = src_path + '.mapped.' + type_filter + '-' + str(
      threshold) + '.' + key
  print("Creating file: %s" % (new_path))
  src_path = src_path + '.' + key
  fn = open(src_path + '.txt', 'r')
  fn1 = open(new_path + '.txt', 'w')

  cont = fn.readlines()
  for i in range(0, len(cont)):
    if i in lines_to_keep:
      fn1.write(cont[i])
    else:
      pass

  fn.close()
  fn1.close()


# Filter the nodes for different attributes and thresholds
threshold_pop = 95
ids_pop = popularity_filter(graph, threshold_pop)
print("Ids of the nodes filtered by popularity: ", ids_pop)

threshold_genre = ["deep southern trap", "swedish indie rock"]
ids_genre = genre_filter(graph, threshold_genre, key='or')
# here, for threshold is important for now to have the double '' inside the str
print("Ids of the nodes filtered by genre: ", ids_genre)

threshold_centrality = 310000.000
type_centr = 'hc'
ids_c = centrality_filter(graph, type_centr, threshold_centrality)
print("Ids of the nodes filtered by centrality: ", ids_c)

# Generate the filtered graph - creating a map array to pass to the trasnform_map function
n = graph.numNodes()
map_pop = map_nodes(n, ids_pop)
print("Map generated")
subgraph_pop = graph.transform_map(map_pop)
print("Subgraph generated")

# Create the metadata file for the new filtered graph
type_filt = 'popularity'
update_txt_metadata(dest_path, 'ids', type_filt, threshold_pop, ids_pop)
update_txt_metadata(dest_path, 'type', type_filt, threshold_pop, ids_pop)
update_txt_metadata(dest_path, 'name', type_filt, threshold_pop, ids_pop)
update_txt_metadata(dest_path, 'popularity', type_filt, threshold_pop, ids_pop)
update_txt_metadata(dest_path, 'genre', type_filt, threshold_pop, ids_pop)
update_txt_metadata(dest_path, 'followers', type_filt, threshold_pop, ids_pop)

key = "popularity"
dest_path = "graphs/spotify-2018"
subgraph_path = dest_path + "-map-" + key
jwebgraph.utils.store_subgraph(subgraph_pop, subgraph_path)

#subgraph = jwebgraph.utils.BVGraph(subgraph_path)
#subgraph.__init__(subgraph_path, ".")
#print(subgraph.artist(id=0).genre)
