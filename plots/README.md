# Plots
You can make the same plots that are in the slides or in the paper
with the script [`main.py`](main.py)

## Paper
To make the plots for the paper, run
```
python plots/main.py \
  <graph path> \
  <dest path> \
  -a 4NJhFmfw43RLBLjQvxDuRS \
  -a 4NJhFmfw43RLBLjQvxDuRS \
  -a 4RcqZYDDLikC5uAIUD8Ptx \
  -a 6eUKZXaKkcviH0Ku9w2n3V \
  -a 1sBkRIssrMs1AbVkOJbc7a
```

## Slides
To make the plots for the slides, run
```
python plots/main.py \
  <graph path> \
  <dest path> \
  --mpl-style ./plots/slides.mplstyle \
  --suffix .svg \
  -a 4NJhFmfw43RLBLjQvxDuRS \
  -a 4NJhFmfw43RLBLjQvxDuRS \
  -a 4RcqZYDDLikC5uAIUD8Ptx \
  -a 6eUKZXaKkcviH0Ku9w2n3V \
  -a 1sBkRIssrMs1AbVkOJbc7a
```

### Slideshow
This folder contains the slideshow source for the paper presentation.
To upload the source for presentation run
```
curl -F file=@./plots/slides.md https://mark.show
```
or
```
cat ./plots/slides.md | curl -F 'data=<-' https://mark.show
```
You will find the presentation at the URL returned in the HTTP response
