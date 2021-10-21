# Slides
This folder contains the slideshow source for the project presentation. To upload the source for presentation, from this directory run
```
curl -F file=@slides.md https://mark.show
```
or
```
cat slides.md | curl -F 'data=<-' https://mark.show
```
You will find the presentation at the URL returned in the HTTP response

## Make plots
You can make the same plots that are in the slides by running
```
python -m featgraph.plots \
  <graph path> \
  <dest path> \
  -a 2wOqMjp9TyABvtHdOSOTUS \
  -a 3fWtSlVVBl6uvTZdNrefU2 \
  -a 6eUKZXaKkcviH0Ku9w2n3V \
  -a 1sBkRIssrMs1AbVkOJbc7a \
  -a 7pXu47GoqSYRajmBCjxdD6
```
