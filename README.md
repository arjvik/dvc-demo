# DVC Demo Project

Demo project to test out DVC and DAGsHub

### See this repository on [DAGsHub](https://dagshub.com/arjvik/dvc-demo) and [GitHub](https://github.com/arjvik/dvc-demo)

[Data Version Control (DVC)](https://dvc.org/) is a version control system built around the machine learning workflow. It allows you to build and run pipelines, represented as a Directed Acyclic (dependency) Graphs, with data and code, tracking large outputs using Git-controlled metafiles. [DAGsHub](https://dagshub.com/) is a fully-featured Git and DVC remote, i.e. DAGsHub is to DVC as GitHub is to Git.

This repository implements a binary classifier on questions from CrossValidated Stack Exchange to determine if they are about machine learning or not. The machine learning portion of this repository is unremarkable and uses standard techniques. The python file `main.py` contains code for all steps of the ML pipeline.

```
Usage: python3 main.py [split|featurize|tfidf|train|test]
```

## DAGsHub Features

### Experiment Tracker

![](.github/experiments.png)

### Pipeline DAG

![](.github/dag.png)

### DVC-tracked folder view (`outputs/`)

![](.github/outputs.png)

## Credits

- [DAGsHub Experiments Tutorial](https://dagshub.com/docs/experiment-tutorial/overview/)
- [DVC Getting Started - Data Pipelines](https://dvc.org/doc/start/data-pipelines)
- [DVC Command Reference](https://dvc.org/doc/command-reference)