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
> in eigenvector centrality, Journal of Complex Networks (2021)

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
> in eigenvector centrality, Journal of Complex Networks (2021)

between December 2017 and January 2018

<table style="width:100%;" ><tr>
  <td style="width:50%;text-align:left;">Nodes: 1 250 114</td>
  <td style="width:50%;text-align:right;">Edges: 7 435 330</td>
</tr></table>

---

<!-- .slide: data-auto-animate -->

## The Network and the Data

<table><tr>
<td><img alt="Degree Distribution" src="https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/degrees.png"
style="height:50vh" /></td>
<td><img alt="Distances Distribution" src="https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/distances.png"
style="height:50vh" /></td>
</tr></table>

---

<!-- .slide: data-auto-animate -->

## Music Genre and Artist Centrality

<p> Does music genre affect the centrality of artists? </p><!-- .element: class="fragment " data-fragment-index="1" -->
<p> Which genres have the most central artists? </p><!-- .element: class="fragment " data-fragment-index="2" -->

---

<!-- .slide: data-auto-animate -->

## Music Genre and Artist Centrality

<!-- This links to the local website for plot navigation -->

<a href="http://0.0.0.0:8000/" >
<img alt="Distances Distribution" src="https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/indegrees.svg"
style="height:55vh" />
</a>

---

<!-- .slide: data-auto-animate -->

## Transitions in Centrality

classical artists are the most central in the full graph<!-- .element: class="fragment " data-fragment-index="1" -->

<ol style="column-count: 2; column-gap: 5vw; width: 55vw">
<!--li>Traditional</li-->
<!--li>Various Artists</li-->
<li>Johann Sebastian Bach</li><!-- .element: class="fragment " data-fragment-index="2" -->
<li>Wolfgang Amadeus Mozart</li><!-- .element: class="fragment " data-fragment-index="3" -->
<!--li>Anonymous</li-->
<li>Ludwig van Beethoven</li><!-- .element: class="fragment " data-fragment-index="4" -->
<li>Franz Schubert</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>George Frideric Handel</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Johannes Brahms</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Pyotr Ilyich Tchaikovsky</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Claude Debussy</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Felix Mendelssohn</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Giuseppe Verdi</li><!-- .element: class="fragment " data-fragment-index="5" -->
</ol>

<div style="height: 5vh"></div>
<small>
results shown for harmonic centrality, but they are
similar for other centrality measures
</small><!-- .element: class="fragment " data-fragment-index="5" -->

---

<!-- .slide: data-auto-animate -->

## Transitions in Centrality

what happens if we prune low-popularity nodes?

---

<!-- .slide: data-auto-animate -->

## Transitions in Centrality

hip-hop artists emerge

<ol style="column-count: 2; column-gap: 5vw; width: 55vw">
<li>Snoop Dogg</li>
<li>Lil Wayne</li>
<li>Nicki Minaj</li>
<li>Ty Dolla $ign</li>
<li>Rick Ross</li>
<li>French Montana</li>
<li>Pharrell Williams</li>
<li>Akon</li>
<li>Chris Brown</li>
<li>2 Chainz</li>
</ol>

<div style="height: 5vh"></div>
<small>
top 10 artists for harmonic centrality in the graph
of artists with a popularity of more than 60%
</small>
