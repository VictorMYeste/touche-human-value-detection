# GitHub

https://github.com/VictorMYeste/touche-human-value-detection

# Probando modelos

## bert-baseline (principios de abril)

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.16
Self-direction: action constrained:     0.00
Stimulation attained:                   0.25
Stimulation constrained:                0.00
Hedonism attained:                      0.32
Hedonism constrained:                   0.00
Achievement attained:                   0.34
Achievement constrained:                0.21
Power: dominance attained:              0.22
Power: dominance constrained:           0.04
Power: resources attained:              0.21
Power: resources constrained:           0.22
Face attained:                          0.00
Face constrained:                       0.10
Security: personal attained:            0.08
Security: personal constrained:         0.24
Security: societal attained:            0.31
Security: societal constrained:         0.36
Tradition attained:                     0.41
Tradition constrained:                  0.00
Conformity: rules attained:             0.33
Conformity: rules constrained:          0.32
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.02
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.18
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.20
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.34
Universalism: concern constrained:      0.17
Universalism: nature attained:          0.48
Universalism: nature constrained:       0.39
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.06

Macro average:                          0.16

## bert-baseline (17/04)

batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.12
Self-direction: action constrained:     0.00
Stimulation attained:                   0.26
Stimulation constrained:                0.00
Hedonism attained:                      0.24
Hedonism constrained:                   0.00
Achievement attained:                   0.34
Achievement constrained:                0.19
Power: dominance attained:              0.30
Power: dominance constrained:           0.04
Power: resources attained:              0.24
Power: resources constrained:           0.20
Face attained:                          0.01
Face constrained:                       0.02
Security: personal attained:            0.02
Security: personal constrained:         0.23
Security: societal attained:            0.30
Security: societal constrained:         0.41
Tradition attained:                     0.43
Tradition constrained:                  0.00
Conformity: rules attained:             0.37
Conformity: rules constrained:          0.27
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.18
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.22
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.34
Universalism: concern constrained:      0.12
Universalism: nature attained:          0.45
Universalism: nature constrained:       0.34
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.03

Macro average:                          0.15

## bert-baseline-optuna

Best trial:
  Value: 0.12
  Params:
    learning_rate: 2.9052405145906253e-05
    batch_size: 8
    num_train_epochs: 3
    weight_decay: 0.04094509981752778

Best trial is worse than bert-baseline (17/04), so optune is discarded for now (it is too slow to do a new batch of different trials with no cuda)

## bert-baseline-scheduler (Linear)

batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01

num_warmup_steps=0

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.15
Self-direction: action constrained:     0.00
Stimulation attained:                   0.27
Stimulation constrained:                0.00
Hedonism attained:                      0.26
Hedonism constrained:                   0.00
Achievement attained:                   0.33
Achievement constrained:                0.17
Power: dominance attained:              0.28
Power: dominance constrained:           0.03
Power: resources attained:              0.23
Power: resources constrained:           0.18
Face attained:                          0.01
Face constrained:                       0.08
Security: personal attained:            0.06
Security: personal constrained:         0.28
Security: societal attained:            0.30
Security: societal constrained:         0.39
Tradition attained:                     0.43
Tradition constrained:                  0.00
Conformity: rules attained:             0.36
Conformity: rules constrained:          0.28
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.18
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.22
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.34
Universalism: concern constrained:      0.13
Universalism: nature attained:          0.40
Universalism: nature constrained:       0.28
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.00

Macro average:                          0.15

## bert-baseline-scheduler (StepLR)

batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01

step_size=10, gamma=0.5

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.00
Self-direction: action constrained:     0.00
Stimulation attained:                   0.00
Stimulation constrained:                0.00
Hedonism attained:                      0.00
Hedonism constrained:                   0.00
Achievement attained:                   0.00
Achievement constrained:                0.00
Power: dominance attained:              0.07
Power: dominance constrained:           0.00
Power: resources attained:              0.00
Power: resources constrained:           0.00
Face attained:                          0.00
Face constrained:                       0.00
Security: personal attained:            0.00
Security: personal constrained:         0.00
Security: societal attained:            0.00
Security: societal constrained:         0.00
Tradition attained:                     0.00
Tradition constrained:                  0.00
Conformity: rules attained:             0.00
Conformity: rules constrained:          0.00
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.00
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.00
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.00
Universalism: concern constrained:      0.00
Universalism: nature attained:          0.00
Universalism: nature constrained:       0.00
Universalism: tolerance attained:       0.01
Universalism: tolerance constrained:    0.00

Macro average:                          0.00

## bert-baseline-scheduler (ExponentialLR)

batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01

gamma=0.9

Self-direction: thought attained:       0.01
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.01
Self-direction: action constrained:     0.00
Stimulation attained:                   0.05
Stimulation constrained:                0.01
Hedonism attained:                      0.01
Hedonism constrained:                   0.00
Achievement attained:                   0.01
Achievement constrained:                0.01
Power: dominance attained:              0.00
Power: dominance constrained:           0.03
Power: resources attained:              0.05
Power: resources constrained:           0.00
Face attained:                          0.02
Face constrained:                       0.00
Security: personal attained:            0.01
Security: personal constrained:         0.03
Security: societal attained:            0.02
Security: societal constrained:         0.00
Tradition attained:                     0.03
Tradition constrained:                  0.00
Conformity: rules attained:             0.00
Conformity: rules constrained:          0.03
Conformity: interpersonal attained:     0.01
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.01
Humility constrained:                   0.00
Benevolence: caring attained:           0.00
Benevolence: caring constrained:        0.01
Benevolence: dependability attained:    0.00
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.06
Universalism: concern constrained:      0.03
Universalism: nature attained:          0.04
Universalism: nature constrained:       0.02
Universalism: tolerance attained:       0.01
Universalism: tolerance constrained:    0.01

Macro average:                          0.01

## bert-baseline-scheduler (CosineAnnealingLR)

batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01

T_max=50

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.04
Self-direction: action constrained:     0.00
Stimulation attained:                   0.26
Stimulation constrained:                0.00
Hedonism attained:                      0.20
Hedonism constrained:                   0.00
Achievement attained:                   0.33
Achievement constrained:                0.17
Power: dominance attained:              0.24
Power: dominance constrained:           0.01
Power: resources attained:              0.25
Power: resources constrained:           0.18
Face attained:                          0.00
Face constrained:                       0.05
Security: personal attained:            0.04
Security: personal constrained:         0.26
Security: societal attained:            0.31
Security: societal constrained:         0.38
Tradition attained:                     0.42
Tradition constrained:                  0.00
Conformity: rules attained:             0.37
Conformity: rules constrained:          0.27
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.19
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.20
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.32
Universalism: concern constrained:      0.17
Universalism: nature attained:          0.47
Universalism: nature constrained:       0.24
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.00

Macro average:                          0.14

## roberta-base con scheduler (Linear)

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.00
Self-direction: action constrained:     0.00
Stimulation attained:                   0.00
Stimulation constrained:                0.00
Hedonism attained:                      0.00
Hedonism constrained:                   0.00
Achievement attained:                   0.00
Achievement constrained:                0.00
Power: dominance attained:              0.00
Power: dominance constrained:           0.00
Power: resources attained:              0.00
Power: resources constrained:           0.00
Face attained:                          0.00
Face constrained:                       0.00
Security: personal attained:            0.00
Security: personal constrained:         0.00
Security: societal attained:            0.00
Security: societal constrained:         0.00
Tradition attained:                     0.00
Tradition constrained:                  0.00
Conformity: rules attained:             0.00
Conformity: rules constrained:          0.00
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.00
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.00
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.00
Universalism: concern constrained:      0.00
Universalism: nature attained:          0.00
Universalism: nature constrained:       0.00
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.00

Macro average:                          0.00

## roberta-base

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.20
Self-direction: action constrained:     0.00
Stimulation attained:                   0.30
Stimulation constrained:                0.00
Hedonism attained:                      0.39
Hedonism constrained:                   0.00
Achievement attained:                   0.38
Achievement constrained:                0.23
Power: dominance attained:              0.31
Power: dominance constrained:           0.01
Power: resources attained:              0.22
Power: resources constrained:           0.20
Face attained:                          0.04
Face constrained:                       0.05
Security: personal attained:            0.03
Security: personal constrained:         0.29
Security: societal attained:            0.30
Security: societal constrained:         0.44
Tradition attained:                     0.42
Tradition constrained:                  0.00
Conformity: rules attained:             0.37
Conformity: rules constrained:          0.33
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.17
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.21
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.35
Universalism: concern constrained:      0.14
Universalism: nature attained:          0.52
Universalism: nature constrained:       0.42
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.00

Macro average:                          0.17

## deberta

Self-direction: thought attained:       0.04
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.24
Self-direction: action constrained:     0.03
Stimulation attained:                   0.31
Stimulation constrained:                0.00
Hedonism attained:                      0.39
Hedonism constrained:                   0.06
Achievement attained:                   0.35
Achievement constrained:                0.25
Power: dominance attained:              0.30
Power: dominance constrained:           0.08
Power: resources attained:              0.23
Power: resources constrained:           0.19
Face attained:                          0.08
Face constrained:                       0.23
Security: personal attained:            0.13
Security: personal constrained:         0.34
Security: societal attained:            0.28
Security: societal constrained:         0.42
Tradition attained:                     0.42
Tradition constrained:                  0.13
Conformity: rules attained:             0.38
Conformity: rules constrained:          0.31
Conformity: interpersonal attained:     0.02
Conformity: interpersonal constrained:  0.09
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.27
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.27
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.31
Universalism: concern constrained:      0.18
Universalism: nature attained:          0.43
Universalism: nature constrained:       0.47
Universalism: tolerance attained:       0.15
Universalism: tolerance constrained:    0.07

Macro average:                          0.20

## electra

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.15
Self-direction: action constrained:     0.00
Stimulation attained:                   0.29
Stimulation constrained:                0.00
Hedonism attained:                      0.12
Hedonism constrained:                   0.00
Achievement attained:                   0.35
Achievement constrained:                0.21
Power: dominance attained:              0.32
Power: dominance constrained:           0.02
Power: resources attained:              0.24
Power: resources constrained:           0.25
Face attained:                          0.00
Face constrained:                       0.04
Security: personal attained:            0.00
Security: personal constrained:         0.27
Security: societal attained:            0.31
Security: societal constrained:         0.42
Tradition attained:                     0.38
Tradition constrained:                  0.00
Conformity: rules attained:             0.38
Conformity: rules constrained:          0.31
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.19
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.21
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.35
Universalism: concern constrained:      0.19
Universalism: nature attained:          0.47
Universalism: nature constrained:       0.39
Universalism: tolerance attained:       0.00
Universalism: tolerance constrained:    0.00

Macro average:                          0.15

## xlnet

Self-direction: thought attained:       0.00
Self-direction: thought constrained:    0.00
Self-direction: action attained:        0.19
Self-direction: action constrained:     0.00
Stimulation attained:                   0.26
Stimulation constrained:                0.00
Hedonism attained:                      0.35
Hedonism constrained:                   0.00
Achievement attained:                   0.35
Achievement constrained:                0.23
Power: dominance attained:              0.28
Power: dominance constrained:           0.07
Power: resources attained:              0.28
Power: resources constrained:           0.25
Face attained:                          0.00
Face constrained:                       0.11
Security: personal attained:            0.10
Security: personal constrained:         0.31
Security: societal attained:            0.31
Security: societal constrained:         0.41
Tradition attained:                     0.37
Tradition constrained:                  0.00
Conformity: rules attained:             0.35
Conformity: rules constrained:          0.35
Conformity: interpersonal attained:     0.00
Conformity: interpersonal constrained:  0.00
Humility attained:                      0.00
Humility constrained:                   0.00
Benevolence: caring attained:           0.25
Benevolence: caring constrained:        0.00
Benevolence: dependability attained:    0.28
Benevolence: dependability constrained: 0.00
Universalism: concern attained:         0.35
Universalism: concern constrained:      0.19
Universalism: nature attained:          0.46
Universalism: nature constrained:       0.41
Universalism: tolerance attained:       0.11
Universalism: tolerance constrained:    0.03

Macro average:                          0.18

# data augmentation

synonyms:

falta stop words y no hace sinónimos buenos

Invert sentences no es recomendable, y evitar spacy.load dentro de la función

parafrasis podría ser interesante, pero evitar cargar el modelo de nuevo

MEJOR DEJARLO POR AHORA

# lingüística

dos modelos: primera tarea de ver si el valor humano está presente y luego si está a favor o en contra (stance detection), así solo se trabajará con la mitad de las clases

podrían esta cambiando la distribución de los valores humanos según la cultura (si hay alguna clasificación de algún autor), atacando el prefijo. Aplicar el entrenamiento por separado por cada cultura en la primera capa (primera tarea)

features lingüísticos quizá mejor hablarlo en una semana, igual que cómo ejecutarlo

# Subtask 1

Con deberta.

Self-direction: thought:                0.07
Self-direction: action:                 0.24
Stimulation:                            0.29
Hedonism:                               0.29
Achievement:                            0.33
Power: dominance:                       0.32
Power: resources:                       0.31
Face:                                   0.25
Security: personal:                     0.28
Security: societal:                     0.42
Tradition:                              0.45
Conformity: rules:                      0.44
Conformity: interpersonal:              0.09
Humility:                               0.00
Benevolence: caring:                    0.24
Benevolence: dependability:             0.24
Universalism: concern:                  0.33
Universalism: nature:                   0.52
Universalism: tolerance:                0.23

Macro average:                          0.28