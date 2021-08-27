from featgraph import jwebgraph
from featgraph.jwebgraph import start_jvm
from featgraph.pathutils import notisfile, notisglob, derived_paths
from featgraph import conversion
from functools import partial
import os
import jpype
import json
import featgraph.metadata


# start jvm
jvm_path = None
start_jvm(jvm_path=jvm_path)

import featgraph.jwebgraph.utils

# make conversion if not done yet
do_convert = True
adjacency_path = "featgraph/Spotify_pickle/collaboration_network_edge_list.pickle"
metadata_path = "featgraph/Spotify_pickle/artist_data.pickle"
dest_path = "featgraph/graphs/spotify-2018"
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
spotify_basename = "featgraph/graphs/spotify-2018"
graph = jwebgraph.utils.BVGraph(spotify_basename)
graph.reconstruct_offsets()

for r in ("graph", "properties", "ids.txt"):
  if not os.path.isfile(graph.path(r)):
    raise FileNotFoundError(graph.path(r))

missing_value = -20

# Filter the nodes for different attributes and thresholds
threshold_pop = 95
ids_pop = graph.popularity_filter(threshold_pop)
print("Ids of the nodes filtered by popularity: ", ids_pop)

threshold_genre = ["deep southern trap", "swedish indie rock"]
ids_genre = graph.genre_filter(threshold_genre, key='or')
# here, for threshold is important for now to have the double '' inside the str
print("Ids of the nodes filtered by genre: ", ids_genre)

threshold_centrality = 310000.000
type_centr = 'hc'
ids_c = graph.centrality_filter(type_centr, threshold_centrality)
print("Ids of the nodes filtered by centrality: ", ids_c)

# Generate the filtered graph and store it
type_filt = 'popularity'
dest_path = "featgraph/graphs/spotify-2018"
subgraph_path = dest_path + '.mapped-' + type_filt + '-' + str(threshold_pop)
#map_pop = list(filter(lambda p: p > 95, graph.popularity(missing_value)))
#map_pop = [True if graph.popularity(missing_value)[i] > threshold_pop else False for i in range(len(graph.popularity(missing_value)))]
subgraph_pop = graph.transform_map(subgraph_path, ids_pop)
print("Subgraph generated")
print(subgraph_pop.artist(index=0).genre)


#jwebgraph.utils.store_subgraph(subgraph_pop, subgraph_path)

# Create the metadata file for the new filtered graph
#update_txt_metadata(dest_path, 'ids', type_filt, threshold_pop, ids_pop)
#update_txt_metadata(dest_path, 'type', type_filt, threshold_pop, ids_pop)
#update_txt_metadata(dest_path, 'name', type_filt, threshold_pop, ids_pop)
#update_txt_metadata(dest_path, 'popularity', type_filt, threshold_pop, ids_pop)
#update_txt_metadata(dest_path, 'genre', type_filt, threshold_pop, ids_pop)
#update_txt_metadata(dest_path, 'followers', type_filt, threshold_pop, ids_pop)



#subgraph = jwebgraph.utils.BVGraph(subgraph_path)
#print(subgraph.artist(id=0).genre)

