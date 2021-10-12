# Statistical Tests Visualization
This folder contains a client-side web-app that can be used to visualize the results of the Bayesian comparisons of centralities between genres.

## Setup
The `index.json` file stores the configuration metadata for the visualizer. Every key of the JSON object is a centrality measure. For each one, we specify
 - `"title"`: the title of the page for this centrality
 - `"subtitle"`: the subtitle (e.g. if and how the centrality is scaled)
 - `"folder"`: the directory in which the global plots (`rope_probs.svg` and `violin_plots.svg`) are saved
 - `"subfolder"`: the subfolder of the previous folder in which the comparison plots are saved
 - `"pad"`: the padding of the ROPE matrix plot as a percentage of the whole dimension (for each of the four margins)
 - `"genres"`: the list of genres in the same order as they appear in the ROPE matrix plot

If you've been given a zip file with test plots, unzip it here and it should work.

## HTML Server
To host a simple HTML server on your machine to serve this website, you can run
```
python -m http.server
```
