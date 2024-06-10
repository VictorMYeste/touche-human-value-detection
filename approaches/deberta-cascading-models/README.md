# Team philo-of-alexandria for ValueEval'24

## Training

```bash
# build
docker build -f Dockerfile_train -t valueeval24-philo-of-alexandria-train-deberta-cascading:1.0.0 .

# run
docker run --rm \
  -v $PWD/../../../data/valueeval24:/dataset -v $PWD/model:/model \
  valueeval24-philo-of-alexandria-train-deberta-cascading:1.0.0
```

## Prediction

```bash
# build
docker build -f Dockerfile_predict -t valueeval24-philo-of-alexandria-deberta-cascading:1.0.0 .

# run
docker run --rm \
  -v $PWD/../../../data/valueeval24:/dataset -v $PWD/output:/output \
  valueeval24-philo-of-alexandria-deberta-cascading:1.0.0

# view results
cat output/task-2/predictions.tsv
```
