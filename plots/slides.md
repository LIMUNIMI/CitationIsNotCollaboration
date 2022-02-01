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

<small>data processed using WebGraph<br/>webgraph.di.unimi.it</small>

<table><tr>
<td><img alt="Degree Distribution" src="https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/degrees.png"
style="height:50vh" /></td>
<td><img alt="Distances Distribution" src="https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/distances.svg"
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

![Indegree distributions](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/indegrees.svg)

---

<!-- .slide: data-auto-animate -->

## Music Genre and Artist Centrality

![Indegree distributions](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/harmonicc.svg)

---

<!-- .slide: data-auto-animate -->

## Music Genre and Artist Centrality

![Indegree distributions](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/pagerank.svg)

---

<!-- .slide: data-auto-animate -->

## Music Genre and Artist Centrality

![Indegree distributions](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/closenessc.svg)

---

<!-- .slide: data-auto-animate -->

## Transitions in Centrality

classical artists are the most central in the full graph<!-- .element: class="fragment " data-fragment-index="1" -->

<ol style="column-count: 2; column-gap: 5vw; width: 65vw">
<!--li>Traditional</li-->
<!--li>Various Artists</li-->
<li>Johann S. Bach</li><!-- .element: class="fragment " data-fragment-index="2" -->
<li>Wolfgang A. Mozart</li><!-- .element: class="fragment " data-fragment-index="3" -->
<!--li>Anonymous</li-->
<li>Ludwig van Beethoven</li><!-- .element: class="fragment " data-fragment-index="4" -->
<li>Franz Schubert</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>George F. Handel</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Johannes Brahms</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Pyotr I. Tchaikovsky</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Claude Debussy</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Felix Mendelssohn</li><!-- .element: class="fragment " data-fragment-index="5" -->
<li>Giuseppe Verdi</li><!-- .element: class="fragment " data-fragment-index="5" -->
</ol>

<div style="height: 5vh"></div>
<small>
results shown for harmonic centrality, but they are
similar for other centrality measures
</small><!-- .element: class="fragment " data-fragment-index="5" -->

Note:
Let the audience guess the first 3 most important classical artists

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
top 10 artists for harmonic centrality in the graph of artists<br/>
with a popularity of more than 60%
</small>

---

<!-- .slide: data-auto-animate -->

## SGC Model

Tobin South _et al._ proposed a _Social Group Centrality_ model to explain the transitions in centrality

---

<!-- .slide: data-auto-animate -->

## SGC Model

![SGC Model](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/SGC.svg)

Tobin South _et al._ proposed a _Social Group Centrality_ model to explain the transitions in centrality

---

<!-- .slide: data-auto-animate -->

## Compare Spotify graph and SGC model

Tobin South _et al._ observed a critical transition in the _eigenvector centrality_

- Does the same happen for other centralities? <!-- .element: class="fragment " data-fragment-index="1" -->
- Does the same happen for the directed graph? <!-- .element: class="fragment " data-fragment-index="2" -->
- Does a graph sampled from the SGC model behave the same as the Spotify graph? <!-- .element: class="fragment " data-fragment-index="3" -->

---

## Indegree

$c_i^{(indegree)} = \sum_{j=1}^{N} A_{ij}$

![Indegree Centrality](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/transition-indegrees.svg)

Note:
normalized by the number of nodes

---

## Harmonic Centrality

$c_{i}^{(Harmonic)} = \sum_{i \neq j}^{N} \frac{1}{d(i, j)}$

![Harmonic Centrality](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/transition-harmonicc.svg)

Note:
normalized by the number of nodes

---

## Pagerank

$c_{i}^{(Pagerank)} = \frac{1 - d}{N} + d \sum_{j \in M(i)} \frac{c_{j}^{(Pagerank)}}{\sum_{k}A_{jk}}$

![Pagerank](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/transition-pagerank.svg)

Note:
normalized by the number of nodes

---

## Closeness Centrality

$c_i^{(closeness)} = \frac{N}{ \sum_{j=1,j \neq i}^{N} d_{ij} }$

![Closeness Centrality](https://gist.githubusercontent.com/ChromaticIsobar/ce60492f849668e1d64d370ea7440e93/raw/transition-closenessc.svg)

Note:
Normalized by the number of nodes

We remind you that this graph is directed and the pruning can lead to unconnected nodes.
The behaviour for SGC is justified by the principle proposed by Marchiori and Latora (2000), which states that in graphs with infinite distances the harmonic mean behaves better than the arithmetic mean.

---

<!-- .slide: data-auto-animate -->

## Conclusion

We found that the transition in centrality

- occurs for indegree, harmonic centrality, and pagerank <!-- .element: class="fragment " data-fragment-index="1" -->
- occurs at aroud the same popularity threshold as for the eigenvector centrality (40 ~ 50) <!-- .element: class="fragment " data-fragment-index="2" -->
- is not as critical in other centralities as for the eigenvector centrality <!-- .element: class="fragment " data-fragment-index="3" -->

---

<!-- .slide: data-auto-animate -->

## Conclusion

This suggests that the transition in centrality is due to some property of the graph

and that the SGC model captures and extremizes that property <!-- .element: class="fragment " data-fragment-index="1" -->

---

## Thank you for your attention
