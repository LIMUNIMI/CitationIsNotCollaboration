# Contributing to FeatGraph
The following is a set of guidelines for contributing to FeatGraph on GitHub.
These are mostly guidelines, not rules. So, use your best judgment.

#### Table Of Contents

[Git](#git)
 * [Commit frequency vs size](#commit-frequency-vs-size)
 * [Feature branches](#feature-branches)
 * [Binary files](#binary-files)

[Tests](#tests)
 * [Unit Tests](#unit-tests)
 * [Coverage](#coverage)

[Styleguides](#styleguides)
 * [Code style](#code-style)
 * [Code format](#code-format)

## Git
We are using
[`git-flow`](https://nvie.com/posts/a-successful-git-branching-model/)
branching model. Consider using automated tools to manage your branches

### Commit frequency vs size
When possible, prefer a big number of small (self-contained) commits over a
small number of big commits. This usually helps the other contributors in
understanding what you are have done.

### Feature branches
Merged feature branches should not overlap.
To avoid this, consider rebasing your feature branch with respect to `develop`
before merging, or use `git flow feature rebase` if you are using `git-flow`
automated tools.

This is ok, because branches have not been merged yet
```
    o---o---o  develop
        |\
        | o---o---o---o---o  feature/a
         \
          o---o---o---o  feature/b
```

This is not ok, because branches overlap
```
    o---o---o---------------o--o  develop
        |\                 /  /
        | o---o---o---o---o  / feature/a
         \                  /
          o-----o---o------o  feature/b
```

This is ok, because branches have been correctly rebased
```
    o---o---o-------------------o---------------o  develop
             \                 / \             /
              o---o---o---o---o   o---o---o---o
                feature/a           feature/b
```

### Binary files
As a general rule, we should avoid versioning binary files directly using git.
Where possible, provide download links instead.
E.g. the `featgraph` package uses the
[`dependencies.json`](featgraph/jwebgraph/dependencies.json) file to download
the jar files for Java dependencies, instead of storing the jar files directly.

## Tests
Your code should pass all unit tests before it is merged onto `develop`.
Install the Python packages required to run unit tests with
```
pip install -Ur test-requirements.txt
```
### Unit Tests
To perform unit tests, run this command from the repository root directory
```
python -m unittest -v
```

### Coverage
Coverage settings are in the [`.coveragerc`](.coveragerc) file.
To evaluate test coverage, run this commands from the repository root directory
```
coverage run
coverage combine
```

You can output human-readable reports with
```
coverage html
coverage xml
coverage json
```

## Styleguides
Install the Python packages required to automate style check and formatting with
```
pip install -Ur style-requirements.txt
```

### Code style
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

### Code format
Format your code with the yapf auto-formatter.
Format rules are in the [`.style.yapf`](.style.yapf) file.
As for [code style](#code-style), we are using Google's style.
To reformat the codebase, run this command from the repository root directory
```
yapf -ir feagraph tests
```
