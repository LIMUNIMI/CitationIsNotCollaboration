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
python slides.py \
  <graph path> \
  <dest path> \
  -a 4NJhFmfw43RLBLjQvxDuRS \
  -a 4RcqZYDDLikC5uAIUD8Ptx \
  -a 6eUKZXaKkcviH0Ku9w2n3V \
  -a 1sBkRIssrMs1AbVkOJbc7a \
  -a 7pXu47GoqSYRajmBCjxdD6
```
