Team philo-of-alexandria for ValueEval'24

## Setup
```bash
# 1. download valueeval24.zip from https://zenodo.org/records/10974214
# 2. then execute
unzip valueeval24.zip
mkdir -p models
```

## Prediction

```bash
# build
docker build -f Dockerfile_predict -t valueeval24-philo-of-alexandria-deberta-cascading:1.0.0 .

# run
docker run --rm \
  -v $PWD/valueeval24:/dataset -v $PWD/output:/output \
  valueeval24-philo-of-alexandria-deberta-cascading:1.0.0

# view results
cat output/task-2/predictions.tsv
```


## Training

```bash
# build
docker build -f Dockerfile_train -t valueeval24-philo-of-alexandria-train-deberta-cascading:1.0.0 .

# run
docker run --rm \
  -v $PWD/valueeval24:/dataset -v $PWD/models:/models \
  valueeval24-philo-of-alexandria-train-deberta-cascading:1.0.0
```

To use these trained models (and not the oned on Huggingface Hub), uncomment the `# load from directory`-lines in `deberta_task_1.py` and `deberta_task_2.py`. To also use models in the Docker image, uncomment the `# use for locally trained models, also remove from .dockerignore`-line in the `Dockerfile_predict` and comment out the following two lines, then remove `models/` from `.dockerignore` and build the image again.

