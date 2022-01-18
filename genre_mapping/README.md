# Genre Mapping
In the `spotify-2018` dataset metadata, there are music genres.
In total, there are 1533 different subgenres, so,
we need fewer generes in order to make a meaningful analysis.

The script [`create_map.py`](create_map.py) allows to compile a json dictionary
that associates any subgenre to a subset of supergenres. You can make direct
associations (as in [`source_direct.json`](source_direct.json))
from a subgenre to its supergenres or associate all genres that have a token
to the same supergenres (as in [`source_tokenized.json`](source_tokenized.json)).
String substitutions to be done before tokenization,  they can be
specified in a third file (as in [`preprocess.json`](preprocess.json)).

Interactive mode allows the interactive compilation of direct and tokenized
associations until all subgenres have been covered by the map.

The resulting map can be saved to a JSON file
(as in [`genre_map.json`](../featgraph/genre_map.json)).
