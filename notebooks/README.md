# FeatGraph Notebooks
Example notebooks

## Configuration
### Python/Conda
To run these notebooks, please install the `featgraph` package and its dependencies (including the [notebooks requirements](../notebooks-requirements.txt)).
For detailed setup instructions, see the main [README.md](../README.md) file

### Java
To use WebGraph you need a Java runtime of version 9 or higher.
To use these notebooks, you can change this line 
```
jvm_path = None
```
and set `jvm_path` to your JVM full fath. Leaving it to `None`
will result in starting your system's default JVM

#### Dependency Jars
Binary jar files for WebGraph and its dependencies
will be downloaded upon call to the function
```
featgraph.jwebgraph.start_jvm
```
This happens at the end of the `Configure` section of any notebook.
To not download these files, you can call
```
start_jvm(
  download=False,
  root=<jar_root>
)
```
The jar files will be looked for in the `<jar_root>` directory
