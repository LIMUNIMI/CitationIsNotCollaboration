# Contributing to FeatGraph
The following is a set of guidelines for contributing to FeatGraph on GitHub.
These are mostly guidelines, not rules. So, use your best judgment.

## Styleguides
### Python
Install the Python packages required to automate style check and formatting with
```
pip install -Ur style-requirements.txt
```

#### Code style
Check your code style with `pylint`.
We are using the `pylintrc` from
[Google Python Style Guide](https://google.github.io/styleguide) 

To check your code style, run this command from the repository root directory
```
pylint featgraph tests -r y --exit-zero --jobs 1 --persistent=n --rcfile=pylintrc
```

To update the `pylintrc` file, download it from Google's repository
```
wget https://google.github.io/styleguide/pylintrc -O pylintrc
```

#### Format code
Format your code with the yapf auto-formatter.
Format rules are in the [`.style.yapf`](.style.yapf) file.
As for [code style](#code-style), we are using Google's style.

To reformat the codebase, run this command from the repository root directory
```
yapf -ir feagraph tests
```
