---
# Configurations
# black, white, league, beige, sky, simple, serif, night, moon, solarized
theme: league
# cube, page, concave, zoom, linear, fade, none, default 
transition: slide
#  Syntax highlighting style https://highlightjs.org/static/demo/
highlight: solarized-dark
backgroundTransition: zoom
progress: true
controls: true
hideAddressBar: true

# Editor settings
editor:
    fontSize: 16
---

# FeatGraph
### Analyzing musician collaborations using WebGraph and HyperBall on Spotify data

---
<!-- .slide: data-auto-animate -->
## The Network and the Data

---
<!-- .slide: data-auto-animate -->
## The Network and the Data
![Spotify logo](https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg)

<p> ~70 million tracks  <!-- .element: class="fragment " data-fragment-index="1" --> </p>
<p> REST Web API  <!-- .element: class="fragment " data-fragment-index="2" --> </p>

---
<!-- .slide: data-auto-animate -->
## The Network and the Data

Data collected by Tobin South _et al._ for their work

> Popularity and centrality in Spotify networks: critical transitions
in eigenvector centrality, Journal of Complex Networks (2021)

<table style="width:100%;" ><tr>
  <td style="text-align:left;width:50%">Nodes: artists</td>
  <td style="text-align:right;width:50%">Edges: collaborations</td>
</tr></table><!-- .element: class="fragment " data-fragment-index="1" -->

<p> there is an edge from node A to node B if artist B is credited for any song in artist A's discography </p><!-- .element: class="fragment " data-fragment-index="2" -->

---
<!-- .slide: data-auto-animate -->
## The Network and the Data

Data collected by Tobin South _et al._ for their work

> Popularity and centrality in Spotify networks: critical transitions
in eigenvector centrality, Journal of Complex Networks (2021)

between December 2017 and January 2018

<table style="width:100%;" ><tr>
  <td style="width:50%;text-align:left;">Nodes: 1 250 114</td>
  <td style="width:50%;text-align:right;">Edges: 7 435 330</td>
</tr></table>

