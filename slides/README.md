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
