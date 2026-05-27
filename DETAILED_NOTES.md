# Open ML Engineer Detailed Notes

These notes are the first professional draft for the full curriculum. Each
topic is written to become a standalone lesson page later. The goal is not to
memorize definitions, but to understand how the concept works, why it exists,
how to implement it, and how engineers use it in real systems.

## Level 0 - Prerequisites

### Python For ML

#### Big Picture

Python is the working language of modern machine learning because it gives a
rare combination of readability, scientific libraries, tooling, and ecosystem
support. A professional ML engineer does not only write Python syntax; they
write reproducible experiments, reusable modules, clean data-processing code,
and training scripts that other engineers can run.

In ML work, Python usually appears in four forms: notebooks for exploration,
scripts for repeatable jobs, packages for reusable code, and services for
deployment. A beginner often keeps everything in one notebook. A professional
starts in a notebook when exploring, then moves stable logic into modules with
clear inputs, outputs, tests, and configuration.

#### Core Intuition

Think of Python as the control layer around optimized numerical libraries.
When you call NumPy, pandas, PyTorch, or scikit-learn, the expensive work is
often performed by compiled code underneath. Your job is to organize data,
express transformations clearly, and avoid unnecessary slow Python loops when
array operations can do the work.

The most important Python skill for ML is not exotic language knowledge. It is
knowing how data flows through a project: from raw files, to cleaned arrays or
tables, to training code, to evaluation, to saved artifacts.

#### Mathematical Foundation

Python itself is not mathematical, but its data model controls how mathematical
objects are represented. Lists represent ordered collections, dictionaries map
names to values, tuples represent fixed records, iterators stream values, and
classes can bundle data with behavior. In ML code, these structures often hold
datasets, hyperparameters, metrics, and model outputs.

You should understand mutability. Lists, dictionaries, arrays, and model
objects can be changed in place. Integers, floats, strings, and tuples are
immutable. This matters because hidden mutation can make experiments
irreproducible.

#### From Scratch Implementation

Important exercises:

- Implement `train_test_split(X, y, test_size, seed)` without scikit-learn.
- Implement a mini-batch iterator that yields batches from arrays.
- Implement a small `ExperimentConfig` class or dataclass containing seed,
  learning rate, batch size, and output directory.

Minimal example:

```python
from dataclasses import dataclass
import random

@dataclass
class ExperimentConfig:
    seed: int = 42
    learning_rate: float = 0.01
    batch_size: int = 32

def train_test_split(X, y, test_size=0.2, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx
```

#### Framework Implementation

Python ML projects commonly use:

- NumPy for arrays and vectorized numerical computation.
- pandas for tabular data.
- scikit-learn for classical ML pipelines.
- PyTorch or TensorFlow for deep learning.
- FastAPI for serving models.

The professional pattern is to keep framework-specific code isolated. For
example, data cleaning should not depend on PyTorch unless it genuinely needs
to. This keeps the project easier to test and reuse.

#### Engineering Perspective

Reproducibility requires more than setting a random seed. You need pinned
dependencies, saved configs, stable data splits, tracked code versions, and
clear output artifacts. Avoid global mutable state, hidden notebook cells, and
manual steps that are not documented.

A strong project layout usually separates:

- `data/` for local raw or processed data, often ignored by Git.
- `src/` for reusable code.
- `notebooks/` for exploration.
- `scripts/` for repeatable commands.
- `configs/` for experiment settings.
- `tests/` for correctness checks.

#### Interview Questions

- Why is vectorized NumPy often faster than a Python loop?
- What is the difference between a module and a package?
- Why can notebooks become dangerous for production ML work?
- How would you make a training script reproducible?
- What should and should not be committed to Git in an ML project?

#### Further Reading

- Python official tutorial.
- Python Packaging User Guide.
- NumPy user guide.
- scikit-learn developer and user guides.

### Math For ML: Linear Algebra

#### Big Picture

Linear algebra is the language of machine learning models. Features become
vectors, datasets become matrices, images become tensors, and neural networks
become compositions of matrix operations. If you understand shapes, dot
products, projections, matrix multiplication, and eigendecomposition, many ML
algorithms stop feeling mysterious.

#### Core Intuition

A vector is a point or direction. A matrix is a transformation that moves,
rotates, scales, combines, or projects vectors. A tensor is a higher-dimensional
array used when data has more structure, such as batches of images with height,
width, and channels.

The dot product measures alignment. If two vectors point in similar directions,
their dot product is large and positive. If they are perpendicular, it is zero.
This idea appears everywhere: linear regression, logistic regression, attention,
embeddings, and similarity search.

#### Mathematical Foundation

For vectors `a` and `b`, the dot product is:

```text
a · b = sum_i a_i b_i
```

Matrix multiplication composes many dot products. If `X` is a matrix of shape
`(n_samples, n_features)` and `w` is a vector of shape `(n_features,)`, then
`Xw` produces one score per sample.

Norms measure size. The L2 norm is:

```text
||x||_2 = sqrt(sum_i x_i^2)
```

Cosine similarity divides the dot product by vector lengths:

```text
cos(a, b) = (a · b) / (||a|| ||b||)
```

Rank tells how many independent directions a matrix contains. Eigenvectors are
directions that a matrix scales without rotating. PCA uses eigenvectors or SVD
to find directions of maximum variance.

#### From Scratch Implementation

```python
import numpy as np

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def l2_norm(x):
    return sum(v * v for v in x) ** 0.5

def cosine_similarity(a, b):
    return dot(a, b) / (l2_norm(a) * l2_norm(b))

def pca_numpy(X, k):
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    components = vectors[:, order[:k]]
    return X_centered @ components
```

#### Framework Implementation

In NumPy, use `@`, `np.linalg.norm`, `np.linalg.eigh`, and `np.linalg.svd`. In
PyTorch, use tensors and operations such as `torch.matmul`, `torch.linalg.norm`,
and `torch.linalg.svd`. Prefer library functions for numerical stability.

#### Engineering Perspective

Most ML bugs are shape bugs. Always know whether your data is shaped as
`(batch, features)`, `(batch, channels, height, width)`, or `(batch, sequence,
embedding)`. Keep the batch dimension explicit. Avoid manually computing matrix
inverses for regression or optimization when a stable solver exists.

#### Interview Questions

- What does a dot product measure?
- Why is matrix multiplication central to neural networks?
- What is cosine similarity and where is it used?
- Why is directly inverting a matrix often a bad idea?
- What does PCA optimize?

#### Further Reading

- Gilbert Strang, linear algebra lectures.
- Mathematics for Machine Learning.
- NumPy linear algebra documentation.

### Math For ML: Probability

#### Big Picture

Probability is how ML represents uncertainty. Classification outputs are often
probabilities, generative models define distributions, evaluation metrics
estimate expected behavior, and Bayesian reasoning helps update beliefs from
evidence.

#### Core Intuition

A model does not know the truth. It makes claims under uncertainty. A predicted
probability of `0.8` should mean that among many similar cases, roughly 80%
will be positive if the model is calibrated.

Probability also explains learning. Maximum likelihood training asks: which
parameters make the observed data most probable under the model?

#### Mathematical Foundation

Conditional probability:

```text
P(A | B) = P(A and B) / P(B)
```

Bayes rule:

```text
P(A | B) = P(B | A) P(A) / P(B)
```

Expectation:

```text
E[X] = sum_x x P(X = x)
```

Variance:

```text
Var(X) = E[(X - E[X])^2]
```

Maximum likelihood chooses parameters `theta` that maximize:

```text
P(data | theta)
```

In practice we maximize log likelihood because products of probabilities can
underflow.

#### From Scratch Implementation

```python
import numpy as np

def bernoulli_mle(samples):
    return np.mean(samples)

def gaussian_log_likelihood(x, mu, sigma):
    return np.sum(
        -0.5 * np.log(2 * np.pi * sigma ** 2)
        - ((x - mu) ** 2) / (2 * sigma ** 2)
    )
```

#### Framework Implementation

scikit-learn exposes probabilistic classifiers with `predict_proba`. PyTorch
uses distributions in `torch.distributions` and losses such as cross entropy,
which corresponds to negative log likelihood for classification.

#### Engineering Perspective

Probabilities are only useful if they are meaningful for the decision. A fraud
system, medical triage system, or loan model needs calibrated probabilities
because thresholds affect real outcomes. Always distinguish ranking quality
from calibration quality.

#### Interview Questions

- Explain Bayes rule intuitively.
- What is maximum likelihood estimation?
- Why do we use log probabilities?
- What is calibration?
- What independence assumption does Naive Bayes make?

#### Further Reading

- PRML probability review.
- scikit-learn calibration documentation.
- Deep Learning book probability chapter.

### Math For ML: Calculus Intuition

#### Big Picture

Calculus tells models how to improve. A loss function measures how wrong a
model is. Gradients tell us which direction changes parameters to reduce that
loss. Deep learning is possible because the chain rule lets us compute
gradients through large compositions of functions.

#### Core Intuition

Imagine standing on a landscape where height means loss. The gradient points in
the direction of steepest increase. Gradient descent moves in the opposite
direction. The learning rate controls step size.

#### Mathematical Foundation

For a one-dimensional function `f(x)`, the derivative measures local slope:

```text
f'(x) = limit_{h -> 0} (f(x + h) - f(x)) / h
```

For many parameters, the gradient is a vector of partial derivatives:

```text
grad f = [df/dw1, df/dw2, ..., df/dwk]
```

Gradient descent update:

```text
w_new = w_old - learning_rate * grad_w loss
```

The chain rule says:

```text
d f(g(x)) / dx = f'(g(x)) g'(x)
```

Backpropagation is the chain rule applied efficiently through a computation
graph.

#### From Scratch Implementation

```python
def finite_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent_quadratic(lr=0.1, steps=50):
    x = 10.0
    history = []
    for _ in range(steps):
        grad = 2 * x
        x -= lr * grad
        history.append(x)
    return history
```

#### Framework Implementation

PyTorch records operations on tensors with `requires_grad=True`. Calling
`loss.backward()` computes gradients using reverse-mode automatic
differentiation.

#### Engineering Perspective

Gradient checks are valuable when writing custom losses or layers. If a model
does not train, inspect learning rate, gradient norms, initialization, data
scale, and loss implementation. Non-convex models do not promise global
optima; engineers care about stable training and useful validation performance.

#### Interview Questions

- What does a gradient represent?
- Why does gradient descent subtract the gradient?
- What is the chain rule?
- How does backpropagation use the chain rule?
- Why can a learning rate be too high or too low?

#### Further Reading

- Deep Learning book optimization chapters.
- PyTorch autograd tutorials.
- Mathematics for Machine Learning calculus chapters.

### Data Handling: NumPy

#### Big Picture

NumPy is the foundation of numerical Python. It provides efficient
multi-dimensional arrays and vectorized operations. Many ML libraries either
accept NumPy arrays directly or follow NumPy's shape and broadcasting ideas.

#### Core Intuition

NumPy lets you express operations over whole arrays instead of individual
Python objects. This matters because loops in Python are slow, while NumPy
delegates numerical loops to optimized compiled code.

#### Mathematical Foundation

An `ndarray` has:

- shape: number of elements along each axis.
- dtype: type of each element.
- strides: how memory is stepped through.
- axes: dimensions over which operations happen.

Broadcasting allows arrays with compatible shapes to operate together. For
example, subtracting a feature mean of shape `(features,)` from a matrix of
shape `(samples, features)` subtracts the mean from every row.

#### From Scratch Implementation

```python
import numpy as np

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def predict_linear(X, w, b):
    return X @ w + b

def batches(X, y, batch_size, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]
```

#### Framework Implementation

Use NumPy arrays for classical ML and convert to tensors for deep learning:

```python
import torch

X_tensor = torch.from_numpy(X).float()
```

#### Engineering Perspective

Be explicit about dtype. `float64` is precise but larger; `float32` is common in
deep learning. Understand when slicing creates a view versus a copy. Avoid
broadcasting that makes code clever but unclear.

#### Interview Questions

- What is broadcasting?
- Why is vectorization faster than Python loops?
- What is the difference between shape `(n,)` and `(n, 1)`?
- Why does dtype matter?
- What is a view in NumPy?

#### Further Reading

- NumPy user guide.
- NumPy broadcasting documentation.
- NumPy random generation documentation.

### Data Handling: Pandas

#### Big Picture

pandas is the standard tool for tabular data exploration and preparation. It is
used to inspect datasets, clean missing values, join tables, create aggregate
features, and prepare arrays for modeling.

#### Core Intuition

A DataFrame is a table with named columns and an index. The names matter:
columns represent variables, rows represent observations, and indexes determine
alignment. pandas is powerful because it combines spreadsheet-like operations
with programmable pipelines.

#### Mathematical Foundation

pandas is not a mathematical library in the same way NumPy is, but it controls
the statistical dataset. Groupby operations compute conditional summaries:
means, counts, rates, and distributions per group. Joins combine observations
from different tables and can easily introduce leakage if done incorrectly.

#### From Scratch Implementation

```python
import pandas as pd

df = pd.read_csv("data.csv")

df["age"] = df["age"].fillna(df["age"].median())
df["signup_month"] = pd.to_datetime(df["signup_date"]).dt.month

customer_features = (
    df.groupby("customer_id")
      .agg(total_spend=("amount", "sum"), orders=("order_id", "count"))
      .reset_index()
)

X = df.drop(columns=["target"])
y = df["target"]
```

#### Framework Implementation

scikit-learn pipelines can consume pandas DataFrames, especially with
`ColumnTransformer` for numeric and categorical processing. Convert to NumPy
only when the downstream library requires it.

#### Engineering Perspective

Be careful with chained assignment and hidden copies. Track schema: expected
columns, types, allowed categories, and missing-value rules. Avoid leakage when
creating aggregate features: never use future information to predict the past.

#### Interview Questions

- What is the difference between a Series and a DataFrame?
- How does groupby work?
- What is data leakage in feature aggregation?
- Why can joins be dangerous in ML pipelines?
- How do you handle missing values?

#### Further Reading

- pandas user guide.
- scikit-learn preprocessing documentation.
- Feature engineering references from production ML systems.

### Git And Linux Basics

#### Big Picture

ML engineers work in teams, on servers, with code that must be reviewed,
reproduced, deployed, and debugged. Git tracks source code history. Linux shell
skills let you inspect files, run jobs, automate workflows, and work on remote
machines.

#### Core Intuition

Git is a history graph of snapshots. Your working tree contains current files.
The staging area chooses what will go into the next commit. A commit records a
named, reviewable change. Branches let work happen independently before merge.

Linux commands compose small tools. `ls` lists, `cd` moves, `cat` prints,
`grep` or `rg` searches, pipes connect commands, and environment variables
configure processes.

#### Mathematical Foundation

There is no ML math here, but there is an engineering model: reproducibility is
a graph of code version, data version, configuration, environment, and output
artifacts. Git covers the code part of that graph.

#### From Scratch Implementation

```bash
git init
git status
git add src/train.py
git commit -m "Add training script"
git checkout -b add-logistic-regression
git diff
git log --oneline
```

Useful shell commands:

```bash
pwd
ls -lah
head data.csv
wc -l data.csv
rg "TODO" .
python scripts/train.py --config configs/baseline.yaml
```

#### Framework Implementation

Use GitHub pull requests for review. Use `.gitignore` to exclude generated
artifacts, local datasets, virtual environments, and secrets.

#### Engineering Perspective

Commit small logical changes. Never commit API keys, credentials, private data,
or large generated files unless the repository is designed for them. A good Git
history explains how the project evolved.

#### Interview Questions

- What is the difference between working tree, staging area, and commit?
- Why use branches?
- What should go into `.gitignore`?
- How would you reproduce an experiment from a Git commit?
- How do pipes help in shell workflows?

#### Further Reading

- Pro Git book.
- Git official documentation.
- Linux command-line documentation.

## Level 1 - Core Machine Learning

### Linear Regression

#### Big Picture

Linear regression predicts a continuous target from one or more input features.
It is used in forecasting, pricing, estimation, and as a baseline for more
complex models. It is simple, interpretable, fast, and mathematically rich
enough to introduce the full supervised learning workflow.

#### Core Intuition

The model learns a weighted sum of features:

```text
y_hat = w1 x1 + w2 x2 + ... + wd xd + b
```

Each weight describes how much the prediction changes when that feature
changes, assuming other features are fixed.

#### Mathematical Foundation

For matrix `X`, target `y`, weights `w`, and bias `b`:

```text
y_hat = Xw + b
L = mean((y - y_hat)^2)
```

The normal equation gives a closed-form solution when conditions are favorable:

```text
w = (X^T X)^(-1) X^T y
```

In practice, gradient-based solvers or stable linear algebra routines are often
preferred over manual inversion.

#### From Scratch Implementation

Implement prediction, MSE, gradients, and gradient descent. Extend to multiple
features and regularization.

#### Framework Implementation

Use scikit-learn `LinearRegression`, `Ridge`, and `Lasso`. Use PyTorch for a
single linear layer when teaching gradient-based learning.

#### Engineering Perspective

Check residuals, outliers, feature scaling, leakage, and multicollinearity.
Linear regression is often the first honest baseline. If a complex model cannot
beat it meaningfully, the problem may be data quality, not model capacity.

#### Interview Questions

- What assumptions does linear regression make?
- Why can outliers strongly affect MSE?
- What is multicollinearity?
- What is the difference between Ridge and Lasso?
- Why should residuals be inspected?

#### Further Reading

- scikit-learn linear models.
- Elements of Statistical Learning.
- PRML linear regression chapter.

### Logistic Regression

#### Big Picture

Logistic regression is a linear classifier that predicts class probabilities.
It is widely used in risk scoring, fraud detection, churn prediction, medical
triage, ad ranking baselines, and interpretable classification systems.

#### Core Intuition

The model computes a linear score called a logit:

```text
z = w · x + b
```

The sigmoid function maps that score to a probability:

```text
p = 1 / (1 + exp(-z))
```

A threshold then converts probability into a class. The default threshold is
often `0.5`, but production systems choose thresholds based on cost,
precision-recall tradeoffs, and risk tolerance.

#### Mathematical Foundation

For binary labels `y in {0, 1}`, binary cross entropy is:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

This is the negative log likelihood of a Bernoulli model. Training logistic
regression means finding weights that make the observed labels likely.

#### From Scratch Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, p, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        error = p - y
        w -= lr * (X.T @ error) / len(X)
        b -= lr * error.mean()
    return w, b
```

#### Framework Implementation

Use `sklearn.linear_model.LogisticRegression` for production-style classical
ML. Use `torch.nn.Linear` with `BCEWithLogitsLoss` for deep-learning style
training. Prefer `BCEWithLogitsLoss` over manually applying sigmoid then BCE
because it is more numerically stable.

#### Engineering Perspective

Logistic regression is linear in the features, not in the output probability.
Feature engineering can make it powerful. Class imbalance affects threshold
choice and metrics. Regularization controls coefficient size and can improve
generalization.

#### Interview Questions

- Why does logistic regression use sigmoid?
- What is a logit?
- Why is cross entropy used instead of MSE for classification?
- How do you handle class imbalance?
- Why might `0.5` be the wrong threshold?

#### Further Reading

- scikit-learn logistic regression documentation.
- PRML classification chapters.
- PyTorch `BCEWithLogitsLoss` documentation.

### Optimization: GD, SGD, Adam

#### Big Picture

Optimization is how models learn. A model has parameters, a loss measures
error, and an optimizer updates parameters to reduce that loss. Linear models,
neural networks, embeddings, and transformers all depend on optimization.

#### Core Intuition

Batch gradient descent computes the gradient using the full dataset. SGD uses
one example at a time. Mini-batch SGD uses a small batch and is the practical
standard because it balances noisy learning signals with efficient hardware
use.

Momentum smooths updates by accumulating velocity. Adam adapts learning rates
per parameter using estimates of first and second moments of gradients.

#### Mathematical Foundation

Gradient descent:

```text
theta = theta - lr * grad_theta L
```

Momentum:

```text
v = beta v + grad
theta = theta - lr * v
```

Adam tracks:

```text
m_t = beta1 m_{t-1} + (1 - beta1) g_t
v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
```

Then applies bias correction and updates parameters with `m_hat / sqrt(v_hat)`.

#### From Scratch Implementation

Implement GD, SGD, momentum, and Adam on a quadratic function and on logistic
regression. Plot convergence, loss curves, and sensitivity to learning rate.

#### Framework Implementation

Use `torch.optim.SGD`, `torch.optim.Adam`, schedulers, gradient clipping, and
mixed precision tools when appropriate.

#### Engineering Perspective

Adam is a strong default for deep learning, but not always the best final
choice. SGD with momentum can generalize well in some vision systems. Batch
size affects noise, throughput, memory, and generalization. Always monitor
loss, validation metrics, gradient norms, and learning rate.

#### Interview Questions

- What is the difference between batch GD, SGD, and mini-batch SGD?
- Why does momentum help?
- What problem does Adam solve?
- Why can a large batch size change training behavior?
- What does gradient clipping do?

#### Further Reading

- Adam paper.
- PyTorch optimizer documentation.
- Deep Learning book optimization chapters.

### Loss Functions

#### Big Picture

A loss function defines what the model is punished for during training. It is
the objective the optimizer actually sees. Metrics evaluate what humans care
about; losses are differentiable training targets that usually approximate
those goals.

#### Core Intuition

Different losses express different assumptions. MSE punishes large errors
strongly. MAE treats errors linearly. Cross entropy rewards assigning high
probability to the correct class. Contrastive losses pull related examples
together and push unrelated examples apart.

#### Mathematical Foundation

Regression:

```text
MSE = mean((y - y_hat)^2)
MAE = mean(|y - y_hat|)
```

Classification:

```text
CrossEntropy = -sum_i y_i log(p_i)
```

Hinge loss for margins:

```text
max(0, 1 - y f(x))
```

#### From Scratch Implementation

Implement MSE, MAE, Huber, binary cross entropy, multiclass cross entropy, and
their basic gradients. Compare how each reacts to an outlier.

#### Framework Implementation

Use PyTorch losses such as `MSELoss`, `L1Loss`, `HuberLoss`,
`BCEWithLogitsLoss`, and `CrossEntropyLoss`.

#### Engineering Perspective

Loss reduction matters. Averaging over tokens, examples, devices, or batches
can change gradient scale. For imbalanced classification, class weighting or
focal loss may help. For ranking and retrieval, classification losses may be
the wrong objective.

#### Interview Questions

- What is the difference between loss and metric?
- Why is cross entropy common for classification?
- When would you use MAE instead of MSE?
- What is Huber loss?
- How can loss reduction affect distributed training?

#### Further Reading

- PyTorch loss documentation.
- scikit-learn metrics documentation.
- Deep Learning book objective functions sections.

### Bias-Variance

#### Big Picture

Bias-variance explains why models fail. A high-bias model is too simple and
underfits. A high-variance model is too sensitive to the training data and
overfits. Good ML engineering is often the art of finding the right capacity,
data, regularization, and validation strategy.

#### Core Intuition

If both train and validation error are high, the model likely has high bias. If
train error is low but validation error is high, the model likely has high
variance.

#### Mathematical Foundation

Expected prediction error can be decomposed conceptually into:

```text
error = bias^2 + variance + irreducible noise
```

This decomposition is clearest under squared error, but the intuition applies
widely.

#### From Scratch Implementation

Generate noisy data from a polynomial. Fit models of degree 1, 3, 10, and 20.
Plot train and validation error. Observe underfitting and overfitting.

#### Framework Implementation

Use scikit-learn polynomial features, train/validation splits, and
cross-validation.

#### Engineering Perspective

High variance is not fixed only by simpler models. More data, stronger
regularization, augmentation, ensembling, or better validation can help. High
bias may require better features, architecture, loss design, or problem
framing.

#### Interview Questions

- What is underfitting?
- What is overfitting?
- How do train and validation curves diagnose bias and variance?
- How does regularization affect variance?
- What is irreducible error?

#### Further Reading

- Elements of Statistical Learning.
- scikit-learn model selection documentation.

### Evaluation Metrics

#### Big Picture

Metrics decide whether a model is useful. A technically impressive model can be
bad if measured with the wrong metric. Evaluation must reflect the cost of
errors, the distribution of real data, and the decision the model supports.

#### Core Intuition

Accuracy asks, "How often are we right?" Precision asks, "When we predict
positive, how often are we right?" Recall asks, "Of all true positives, how
many did we catch?" These are different questions and lead to different
systems.

#### Mathematical Foundation

Confusion matrix terms:

- TP: predicted positive and actually positive.
- FP: predicted positive but actually negative.
- TN: predicted negative and actually negative.
- FN: predicted negative but actually positive.

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Regression metrics include MAE, MSE, RMSE, R2, and MAPE. Retrieval metrics
include recall@k, MRR, MAP, and NDCG.

#### From Scratch Implementation

Implement confusion matrix, precision, recall, F1, ROC curve, PR curve, MAE,
MSE, RMSE, and recall@k.

#### Framework Implementation

Use scikit-learn metrics. For deep learning, accumulate metrics outside the
training graph unless differentiability is required.

#### Engineering Perspective

Accuracy fails on imbalanced datasets. ROC-AUC can look good when PR-AUC is
poor under heavy imbalance. Offline metrics can disagree with online impact
because production data shifts, user behavior changes, or business constraints
matter.

#### Interview Questions

- When is accuracy misleading?
- What is the difference between ROC-AUC and PR-AUC?
- How do you choose a classification threshold?
- What metric would you use for search ranking?
- Why can offline evaluation fail?

#### Further Reading

- scikit-learn model evaluation documentation.
- Information retrieval metric references.

### Feature Engineering

#### Big Picture

Feature engineering turns raw data into signals a model can use. For tabular
ML, feature quality often matters more than model complexity. Even in deep
learning, preprocessing, tokenization, augmentation, and schema design are
forms of feature engineering.

#### Core Intuition

A model cannot use information that is not represented. Feature engineering
asks: what representation makes the prediction relationship easier to learn?

#### Mathematical Foundation

Scaling changes feature distributions. Encoding maps categorical values into
numeric spaces. Interactions add terms such as `x1 * x2`. Aggregations estimate
historical behavior. Feature selection reduces noise or redundancy.

#### From Scratch Implementation

Build a preprocessing pipeline with:

- numeric imputation and scaling.
- categorical one-hot encoding.
- date/time extraction.
- safe train-only fitting of preprocessing statistics.

#### Framework Implementation

Use scikit-learn `Pipeline`, `ColumnTransformer`, `StandardScaler`,
`OneHotEncoder`, and feature selection utilities.

#### Engineering Perspective

Feature engineering is production code. Training-serving skew happens when
features are computed differently during training and inference. Leakage
happens when features use information unavailable at prediction time. Track
feature schemas and validation rules.

#### Interview Questions

- What is data leakage?
- Why scale numeric features?
- What is one-hot encoding?
- Why can target encoding leak?
- What is training-serving skew?

#### Further Reading

- scikit-learn preprocessing documentation.
- Feature store and production ML design docs.

### Classical ML Algorithms

#### Big Picture

Classical ML algorithms remain essential. Tree ensembles dominate many tabular
problems. Logistic regression remains a strong interpretable baseline. kNN,
Naive Bayes, SVMs, k-means, and PCA teach important modeling ideas and still
solve real problems.

#### Core Intuition

Each algorithm has an inductive bias:

- kNN assumes nearby points have similar labels.
- Naive Bayes assumes conditional independence.
- Decision trees split feature space into rules.
- Random forests average many decorrelated trees.
- Gradient boosting builds trees sequentially to fix errors.
- SVMs seek large-margin decision boundaries.
- k-means groups points by distance to centroids.
- PCA finds directions of maximum variance.

#### Mathematical Foundation

Understand distance metrics, impurity measures, margins, ensemble averaging,
boosting residuals, cluster objectives, and variance maximization.

#### From Scratch Implementation

Implement:

- kNN classifier.
- small decision tree using Gini impurity.
- k-means clustering.
- PCA with eigendecomposition.

#### Framework Implementation

Use scikit-learn estimators with consistent `fit`, `predict`, `transform`, and
`score` APIs.

#### Engineering Perspective

Tree ensembles are powerful for structured data and require less scaling than
linear models or SVMs. kNN is simple but can be expensive at inference. SVMs can
work well on medium-sized datasets but require scaling and kernel choices.

#### Interview Questions

- Why are random forests less prone to overfitting than a single tree?
- How does gradient boosting differ from bagging?
- Why does kNN have slow inference?
- What is the kernel trick?
- What objective does k-means optimize?

#### Further Reading

- scikit-learn supervised and unsupervised learning guides.
- Elements of Statistical Learning.

## Level 2 - Deep Learning

### Neural Networks From Scratch

#### Big Picture

Neural networks learn layered representations. Each layer transforms data into
a new representation, and deeper networks can compose simple patterns into
complex ones. They are central to vision, language, speech, recommendation, and
modern generative AI.

#### Core Intuition

A neuron computes a weighted sum, adds a bias, and applies a non-linear
activation. Without non-linear activations, many layers collapse into one
linear transformation. Non-linearity gives networks expressive power.

#### Mathematical Foundation

Dense layer:

```text
Z = XW + b
A = activation(Z)
```

Common activations include ReLU, sigmoid, tanh, and GELU. Softmax converts
class scores into probabilities:

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

#### From Scratch Implementation

Implement dense layers, ReLU, softmax, cross entropy, and a two-layer MLP in
NumPy. Train on a toy classification dataset.

#### Framework Implementation

Use PyTorch `nn.Module`, `nn.Linear`, activation layers, optimizers, and a
training loop.

#### Engineering Perspective

Start with tiny networks and overfit a small batch to debug. Bad
initialization, wrong labels, unnormalized inputs, or incorrect loss usage can
make a network appear broken. Track shapes at every layer.

#### Interview Questions

- Why do neural networks need non-linear activations?
- What is a hidden layer?
- What does softmax do?
- How do you debug a neural network that does not train?
- Why does initialization matter?

#### Further Reading

- Deep Learning book.
- PyTorch neural network tutorials.
- Dive into Deep Learning.

### Backpropagation

#### Big Picture

Backpropagation computes gradients efficiently through a neural network. It is
the reason deep networks can be trained with millions or billions of
parameters.

#### Core Intuition

During the forward pass, the network computes outputs and stores intermediate
values. During the backward pass, gradients flow backward from the loss through
each operation. Each operation uses the chain rule to compute how much it
contributed to the final loss.

#### Mathematical Foundation

If:

```text
z = Wx + b
a = relu(z)
L = loss(a, y)
```

then backpropagation computes `dL/da`, `dL/dz`, `dL/dW`, `dL/db`, and `dL/dx`.
Reverse-mode autodiff is efficient when there are many parameters and one
scalar loss.

#### From Scratch Implementation

Implement manual backprop for a two-layer MLP. Then implement a tiny scalar
autodiff engine where each value stores its parents and local backward
function.

#### Framework Implementation

PyTorch builds a dynamic computation graph. `loss.backward()` fills `.grad`
fields on parameters. `optimizer.step()` uses those gradients to update
weights.

#### Engineering Perspective

Autograd hides gradient computation, but it does not remove the need to
understand it. In-place operations, detached tensors, incorrect loss shapes,
and missing `zero_grad()` calls are common bugs.

#### Interview Questions

- What is backpropagation?
- Why is reverse-mode autodiff efficient for neural networks?
- What does `zero_grad()` do?
- What is a computational graph?
- How would you gradient-check a custom layer?

#### Further Reading

- PyTorch autograd tutorials.
- Deep Learning book backpropagation chapter.

### CNNs

#### Big Picture

Convolutional neural networks are designed for grid-like data such as images.
They exploit local structure and weight sharing, making them efficient and
effective for vision tasks.

#### Core Intuition

A convolutional filter slides across an image and detects local patterns such
as edges, textures, or shapes. Early layers detect simple patterns. Later
layers combine them into higher-level concepts.

#### Mathematical Foundation

A 2D convolution computes weighted sums over local neighborhoods. Key
hyperparameters:

- kernel size.
- stride.
- padding.
- number of channels.
- number of filters.

The receptive field is the region of input that influences a unit in a later
layer.

#### From Scratch Implementation

Implement single-channel 2D convolution with nested loops. Add padding and
stride. Then compare with PyTorch output.

#### Framework Implementation

Use `torch.nn.Conv2d`, `MaxPool2d`, `BatchNorm2d`, and pretrained models from
torchvision.

#### Engineering Perspective

Data augmentation often matters more than small architecture changes. Transfer
learning is practical when data is limited. Watch channel order, image
normalization, input resolution, and train/eval mode.

#### Interview Questions

- Why are CNNs better than dense layers for images?
- What is weight sharing?
- What is a receptive field?
- What do stride and padding do?
- Why use data augmentation?

#### Further Reading

- PyTorch vision tutorials.
- Foundational CNN papers such as LeNet, AlexNet, VGG, ResNet.

### RNNs And LSTMs

#### Big Picture

RNNs model sequences by maintaining hidden state over time. They are important
for understanding sequence modeling and still useful in streaming or
resource-constrained settings, even though transformers dominate many modern
sequence tasks.

#### Core Intuition

An RNN reads one timestep at a time. The hidden state acts as memory. LSTMs add
gates that control what to remember, forget, and output, reducing vanishing
gradient problems.

#### Mathematical Foundation

Simple RNN:

```text
h_t = tanh(W_x x_t + W_h h_{t-1} + b)
```

LSTM uses input, forget, output, and candidate gates to update cell state.
Backpropagation through time unrolls the network across timesteps.

#### From Scratch Implementation

Implement a simple RNN cell and an LSTM cell. Train a character-level model on
a tiny text corpus.

#### Framework Implementation

Use PyTorch `nn.RNN`, `nn.GRU`, and `nn.LSTM`. Pay attention to input shape,
hidden state shape, packed sequences, and batch-first settings.

#### Engineering Perspective

RNNs can suffer from vanishing or exploding gradients. Gradient clipping is
common. For long-range dependencies, transformers are often easier to train,
but RNNs can be efficient for streaming inference.

#### Interview Questions

- What is hidden state?
- Why do vanilla RNNs struggle with long sequences?
- How does an LSTM help?
- What is backpropagation through time?
- When might you still use an RNN?

#### Further Reading

- PyTorch sequence model tutorials.
- Original LSTM paper.
- GRU references.

### Transformers Deep Dive

#### Big Picture

Transformers are the architecture behind most modern LLMs and many multimodal
models. They replaced recurrence with attention, allowing tokens to directly
interact with other tokens and enabling highly parallel training.

#### Core Intuition

Self-attention lets each token ask: which other tokens should I use to update
my representation? Multi-head attention lets the model learn several relation
types at once. Feed-forward layers process each token independently after
information has been mixed by attention.

#### Mathematical Foundation

Scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Transformer block:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Architectures differ:

- encoder-only: understanding tasks.
- decoder-only: autoregressive generation.
- encoder-decoder: sequence-to-sequence tasks.

#### From Scratch Implementation

Implement attention, causal masking, multi-head splitting, positional
embeddings, MLP block, residual connections, and autoregressive sampling.

#### Framework Implementation

Use PyTorch modules or Hugging Face Transformers for production models. Use
small from-scratch versions for education before using high-level APIs.

#### Engineering Perspective

Attention is powerful but quadratic in sequence length. Masking bugs can leak
future tokens. Layer norm placement, initialization, optimizer settings, and
learning-rate schedules affect stability.

#### Interview Questions

- What problem did transformers solve compared with RNNs?
- What are Q, K, and V?
- Why divide by `sqrt(d_k)`?
- What is causal masking?
- What is the difference between encoder-only and decoder-only models?

#### Further Reading

- Attention Is All You Need.
- Hugging Face Transformers documentation.
- PyTorch transformer tutorials.

### Regularization

#### Big Picture

Regularization improves generalization by reducing overfitting. It includes
explicit penalties, stochastic training techniques, data augmentation, early
stopping, and architectural choices.

#### Core Intuition

A model can memorize training data without learning robust patterns.
Regularization makes memorization harder or less attractive.

#### Mathematical Foundation

L2 regularization adds:

```text
lambda ||w||_2^2
```

L1 adds:

```text
lambda ||w||_1
```

Dropout randomly zeroes activations during training. Label smoothing prevents
the model from becoming overconfident.

#### From Scratch Implementation

Add L2 penalty to loss. Implement dropout mask during training. Compare
validation curves with and without regularization.

#### Framework Implementation

Use optimizer weight decay, dropout layers, data augmentation libraries, early
stopping callbacks, and label smoothing options.

#### Engineering Perspective

Regularization is not free. Too much causes underfitting. The right technique
depends on data size, noise, architecture, and task. Data quality and
augmentation often beat clever penalties.

#### Interview Questions

- What is overfitting?
- How does L2 regularization work?
- What does dropout do?
- Why use early stopping?
- What is label smoothing?

#### Further Reading

- Deep Learning book regularization chapter.
- PyTorch dropout and optimizer docs.

### Training At Scale

#### Big Picture

Training at scale means making models train efficiently, reliably, and
reproducibly on larger datasets and hardware. It involves data loading,
distributed computation, memory management, checkpointing, and monitoring.

#### Core Intuition

Large training jobs fail when any part of the pipeline is slow or unstable. GPU
utilization depends on feeding data fast enough. Memory limits require mixed
precision, gradient accumulation, checkpointing, or model parallelism.

#### Mathematical Foundation

Distributed data parallel averages gradients across workers. Gradient
accumulation simulates a larger batch by summing gradients across several
smaller forward/backward passes before stepping.

#### From Scratch Implementation

Profile a training loop. Measure data loading time versus compute time. Add
gradient accumulation and mixed precision to a small PyTorch model.

#### Framework Implementation

Use PyTorch `DataLoader` settings, AMP, `DistributedDataParallel`, checkpoint
saving, and experiment tracking.

#### Engineering Perspective

Save full training state: model, optimizer, scheduler, scaler, config, code
version, and data version. Many bottlenecks are data-loading bottlenecks, not
model bottlenecks. Monitor throughput, memory, loss, and validation metrics.

#### Interview Questions

- What is mixed precision training?
- Why use gradient accumulation?
- How does distributed data parallel work?
- What should a checkpoint contain?
- How do you diagnose low GPU utilization?

#### Further Reading

- PyTorch performance tuning guide.
- PyTorch distributed documentation.

## Level 3 - Modern AI And LLMs

### Attention Mechanism

#### Big Picture

Attention is a mechanism for routing information. It allows a model to decide
which parts of an input are relevant when computing a representation.

#### Core Intuition

Queries ask questions, keys describe what each item offers, and values contain
the information to retrieve. A query compares with keys to produce attention
weights, then uses those weights to combine values.

#### Mathematical Foundation

```text
scores = QK^T / sqrt(d_k)
weights = softmax(scores)
output = weights V
```

Masks can block padding tokens or future tokens.

#### From Scratch Implementation

Implement attention over a toy sequence and visualize the attention matrix as a
heatmap.

#### Framework Implementation

Use PyTorch tensor operations first, then compare with high-level attention
modules.

#### Engineering Perspective

Attention weights are diagnostic signals, not guaranteed explanations. Long
context increases memory and latency. Masking, dtype, and numerical stability
matter.

#### Interview Questions

- What are queries, keys, and values?
- Why is attention useful?
- Why scale by `sqrt(d_k)`?
- What is an attention mask?
- Are attention weights explanations?

#### Further Reading

- Attention Is All You Need.
- PyTorch attention modules.

### Transformers From Scratch

#### Big Picture

Building a transformer from scratch teaches the mechanics behind LLMs:
tokenization, embeddings, attention, residual streams, normalization, and
autoregressive generation.

#### Core Intuition

A decoder-only transformer repeatedly updates token representations while
preventing each token from seeing the future. It learns to predict the next
token.

#### Mathematical Foundation

Training objective:

```text
maximize P(x_t | x_1, ..., x_{t-1})
```

The model outputs logits over vocabulary for each position. Cross entropy is
computed against the next token.

#### From Scratch Implementation

Build character tokenizer, embeddings, causal self-attention, feed-forward
block, layer norm, residual connections, and sampling.

#### Framework Implementation

Use PyTorch for the educational implementation. Later use Hugging Face
Transformers for pretrained models.

#### Engineering Perspective

Shape conventions are critical. Causal masks must be correct. Generation
quality depends on data, context length, temperature, top-k/top-p sampling, and
training quality.

#### Interview Questions

- What does a decoder-only transformer predict?
- What is a residual stream?
- Why use positional embeddings?
- What is autoregressive generation?
- How does temperature affect sampling?

#### Further Reading

- Attention Is All You Need.
- GPT-style model references.
- Hugging Face generation documentation.

### Tokenization

#### Big Picture

Tokenization converts raw text into model-readable units. It controls context
length, cost, multilingual behavior, handling of rare words, and domain-specific
performance.

#### Core Intuition

Characters are flexible but long. Words are meaningful but break on rare words.
Subwords balance both by splitting text into reusable pieces.

#### Mathematical Foundation

BPE starts with small units and repeatedly merges frequent adjacent pairs.
WordPiece and unigram tokenizers use different scoring strategies but share the
goal of representing text with a fixed vocabulary.

#### From Scratch Implementation

Implement toy BPE merges over a small corpus. Compare token counts for common
English, code, math, and domain-specific text.

#### Framework Implementation

Use Hugging Face tokenizers. Inspect `input_ids`, `attention_mask`, special
tokens, padding, and truncation.

#### Engineering Perspective

Always inspect tokenization for your domain. Tokenization affects latency and
cost because models process tokens, not characters. Bad tokenization can hurt
code, medical text, legal text, low-resource languages, and structured data.

#### Interview Questions

- Why not tokenize only by words?
- What is BPE?
- What are special tokens?
- What is an attention mask?
- How does tokenization affect cost?

#### Further Reading

- Hugging Face tokenizer documentation.
- BPE and WordPiece references.

### Embeddings

#### Big Picture

Embeddings represent items as dense vectors. Words, documents, users, products,
images, and code snippets can all be embedded. Similar items should have nearby
vectors.

#### Core Intuition

Embeddings turn semantic similarity into geometry. Retrieval systems search for
vectors close to a query vector. Recommendation systems use embeddings to
represent users and items.

#### Mathematical Foundation

Similarity is often measured with cosine similarity or dot product. Contrastive
learning trains embeddings by pulling positive pairs together and pushing
negative pairs apart.

#### From Scratch Implementation

Create document vectors, compute cosine similarity, and return top-k nearest
documents. Evaluate recall@k on toy queries.

#### Framework Implementation

Use sentence-transformers or Hugging Face models for text embeddings. Use a
vector index for larger collections.

#### Engineering Perspective

Embedding quality is domain dependent. Chunking can dominate retrieval quality.
Normalize embeddings when using cosine similarity. Monitor retrieval failures,
not just generation failures.

#### Interview Questions

- What is an embedding?
- Why use cosine similarity?
- What is contrastive learning?
- How do embeddings support RAG?
- Why does chunking matter?

#### Further Reading

- Sentence embedding model documentation.
- Vector search documentation.
- Contrastive learning papers.

### Fine-Tuning

#### Big Picture

Fine-tuning adapts a pretrained model to a task, domain, style, or instruction
format. It is cheaper than pretraining because the model already contains
general representations.

#### Core Intuition

Pretraining teaches broad patterns. Fine-tuning nudges behavior toward a
specific dataset. It does not reliably insert a large changing knowledge base;
RAG is usually better for facts that change.

#### Mathematical Foundation

Full fine-tuning updates all model weights. Parameter-efficient fine-tuning
updates a small number of added or decomposed parameters. LoRA represents an
update as a low-rank product:

```text
Delta W = A B
```

where `A` and `B` are smaller matrices.

#### From Scratch Implementation

Fine-tune a small classifier. Conceptually demonstrate low-rank adaptation by
freezing a base matrix and learning a low-rank update.

#### Framework Implementation

Use Hugging Face `Trainer`, PyTorch training loops, and PEFT libraries for LoRA
when appropriate.

#### Engineering Perspective

Fine-tuning can overfit, regress safety behavior, or damage general ability.
Use held-out evaluation, regression tests, and clear dataset formatting.
Measure task quality, not only training loss.

#### Interview Questions

- When should you fine-tune instead of use RAG?
- What is LoRA?
- What is supervised fine-tuning?
- Why can fine-tuning overfit?
- What should be in an evaluation set?

#### Further Reading

- Hugging Face fine-tuning docs.
- PEFT/LoRA papers and documentation.

### RAG Systems

#### Big Picture

Retrieval-augmented generation combines an LLM with external knowledge. Instead
of relying only on model parameters, the system retrieves relevant documents and
conditions the answer on them.

#### Core Intuition

RAG is an open-book exam. The retriever finds useful pages. The generator reads
the selected pages and answers. If retrieval fails, generation often fails even
if the model is strong.

#### Mathematical Foundation

A simple RAG system has:

```text
query -> query embedding -> vector search -> top-k chunks -> prompt -> answer
```

Evaluation separates retrieval quality from answer quality. Retrieval can be
measured with recall@k. Answer quality needs faithfulness, relevance, and
citation checks.

#### From Scratch Implementation

Build a local RAG pipeline over Markdown files:

- chunk documents.
- embed chunks.
- search with cosine similarity.
- construct a prompt.
- log retrieved context.

#### Framework Implementation

Use Hugging Face models, vector databases, rerankers, and structured evaluation
tools as the system grows.

#### Engineering Perspective

Most RAG failures are ingestion, chunking, retrieval, reranking, context
assembly, or evaluation failures. Updating an index is cheaper than retraining.
Always log the retrieved context for debugging.

#### Interview Questions

- What is RAG?
- Why use RAG instead of fine-tuning?
- What is chunking?
- How do you evaluate retrieval?
- What are common RAG failure modes?

#### Further Reading

- RAG paper by Lewis et al.
- Hugging Face RAG documentation.
- Retrieval evaluation references.

### Agents

#### Big Picture

AI agents use models to choose actions, call tools, maintain state, and pursue
goals over multiple steps. They are useful when the task requires interaction
with systems, not just one response.

#### Core Intuition

An agent is not magic autonomy. It is a loop: observe state, decide next action,
call a tool, update state, evaluate progress, and stop. The hard parts are
permissions, reliability, memory, tool contracts, and evaluation.

#### Mathematical Foundation

Agents can be viewed as policy systems: given state `s`, choose action `a`.
Unlike classic RL, many LLM agents are prompt/program hybrids rather than
learned policies.

#### From Scratch Implementation

Build a small agent loop with:

- explicit state object.
- tool registry.
- step budget.
- approval boundary.
- action log.

#### Framework Implementation

Use framework abstractions only after building the loop manually. Tools should
have narrow inputs, validated outputs, and clear failure behavior.

#### Engineering Perspective

Agents are distributed systems with an LLM inside. They need observability,
retries, permission boundaries, audit logs, and safe defaults. Avoid giving
broad tools when narrow tools will do.

#### Interview Questions

- What makes a system an agent?
- Why are tool permissions important?
- What is a step budget?
- How do you evaluate an agent?
- What are common agent failure modes?

#### Further Reading

- Tool-use and ReAct-style agent papers.
- Production agent safety and observability docs.

### Multimodal Models

#### Big Picture

Multimodal models process more than one data type: text, images, audio, video,
tables, or documents. They power visual question answering, OCR reasoning,
image captioning, document extraction, speech systems, and robotics interfaces.

#### Core Intuition

Different modalities need different encoders. A vision encoder represents
images, a text model represents language, and a fusion mechanism aligns them so
the model can reason across both.

#### Mathematical Foundation

Contrastive multimodal training often pulls matching image-text pairs together
and pushes mismatched pairs apart. Transformer-based systems may turn image
patches or audio frames into token-like representations.

#### From Scratch Implementation

Compare text and image embeddings conceptually. Build a toy retrieval system
where images and captions are matched by embedding similarity.

#### Framework Implementation

Use pretrained vision-language models for demos. Later add OCR, document AI,
and VQA examples.

#### Engineering Perspective

Multimodal systems fail through poor alignment, OCR errors, visual grounding
mistakes, data rights issues, and weak evaluation. Privacy is harder because
images and audio can contain sensitive information.

#### Interview Questions

- What is a multimodal model?
- How do image-text models align modalities?
- What is visual question answering?
- Why is multimodal evaluation difficult?
- What privacy issues arise with image/audio data?

#### Further Reading

- CLIP-style contrastive learning papers.
- Vision Transformer papers.
- Hugging Face multimodal docs.

## Level 4 - MLOps And Engineering

### Data Pipelines

#### Big Picture

Data pipelines move data from raw sources to validated, transformed, model-ready
datasets. Since ML systems learn from data, pipeline quality often determines
model quality.

#### Core Intuition

An ML pipeline is a factory. Raw data enters, checks and transformations happen,
and reliable features come out. If the factory silently changes, the model may
fail even if the training code is correct.

#### Mathematical Foundation

Pipeline math includes distributions, missingness rates, schema constraints,
and aggregate statistics. Data validation checks whether incoming data matches
expected assumptions.

#### From Scratch Implementation

Build an ETL script:

- read raw CSV.
- validate required columns and types.
- handle missing values.
- write processed data.
- produce a data quality report.

#### Framework Implementation

Use orchestrators and validation tools as systems grow. Batch pipelines and
streaming pipelines have different latency and correctness tradeoffs.

#### Engineering Perspective

Make pipelines idempotent when possible. Track data contracts. Handle backfills
carefully. Silent schema drift is one of the most common production ML failure
modes.

#### Interview Questions

- What is a data pipeline?
- What is schema drift?
- What is a backfill?
- Why is idempotency useful?
- How do data pipelines affect model quality?

#### Further Reading

- Production ML system design references.
- Data validation and orchestration documentation.

### Model Versioning

#### Big Picture

Model versioning tracks which model artifact was trained from which code,
data, configuration, and environment. Without lineage, a model cannot be
trusted, reproduced, rolled back, or audited.

#### Core Intuition

A model file alone is not a model version. A real version includes weights,
training code, data snapshot, config, metrics, dependencies, and ownership.

#### Mathematical Foundation

The key relationship is lineage:

```text
model = f(code_version, data_version, config, environment, random_seed)
```

#### From Scratch Implementation

Save a model artifact with metadata:

- model file.
- training config.
- metrics.
- Git commit.
- dataset hash or version.
- timestamp.

#### Framework Implementation

Use model registries from tools such as MLflow or platform-specific systems.

#### Engineering Perspective

Production needs promotion stages, rollback paths, approvals, and clear owners.
Version the full training recipe, not just weights.

#### Interview Questions

- Why is a model artifact alone insufficient?
- What is lineage?
- What should model metadata contain?
- How do you roll back a model?
- What is a model registry?

#### Further Reading

- MLflow model registry documentation.
- DVC documentation.

### Experiment Tracking

#### Big Picture

Experiment tracking records parameters, metrics, artifacts, and code context
for training runs. It lets teams compare ideas and reproduce promising results.

#### Core Intuition

Without tracking, experiments become memory and filenames. With tracking, each
run becomes a searchable record of what changed and what happened.

#### Mathematical Foundation

Experiment tracking is about controlled comparisons. Change one or a few
variables, measure relevant metrics, and preserve enough context to reproduce
the result.

#### From Scratch Implementation

Log run metadata to JSON or CSV:

- hyperparameters.
- metrics per epoch.
- final metrics.
- artifact paths.
- seed.
- Git commit.

#### Framework Implementation

Use MLflow, Weights and Biases alternatives, DVC experiments, or simple local
tracking depending on project constraints.

#### Engineering Perspective

Log meaningful metrics before logging everything. Use consistent names. Save
artifacts that explain behavior: confusion matrices, residual plots, examples,
and checkpoints.

#### Interview Questions

- What should an experiment tracker record?
- Why is tracking important for collaboration?
- How do you compare experiments fairly?
- What artifacts are useful?
- What is the difference between metric and artifact?

#### Further Reading

- MLflow tracking documentation.
- DVC experiment tracking documentation.

### Deployment: Docker And APIs

#### Big Picture

Deployment turns a trained model into a usable system. A common pattern is a
model served behind an API, packaged in a Docker container, and monitored in
production.

#### Core Intuition

Training code optimizes learning. Serving code optimizes reliability, latency,
clear schemas, and operational safety. They should share model logic when
possible but have different dependencies and constraints.

#### Mathematical Foundation

Serving is less about new math and more about preserving the same feature
transformation used during training:

```text
raw request -> validation -> features -> model -> response
```

#### From Scratch Implementation

Build a FastAPI service:

- load model at startup.
- define request schema.
- validate input.
- return prediction and metadata.
- add `/health` endpoint.

#### Framework Implementation

Containerize with Docker. Keep images small, pin dependencies, and separate
training-only dependencies from serving dependencies.

#### Engineering Perspective

Validate request and response schemas. Watch cold start, memory, latency, and
throughput. Add health checks and graceful error handling. Never expose raw
tracebacks to users.

#### Interview Questions

- How do you serve a model with an API?
- Why use Docker?
- What is a health check?
- Why separate training and serving dependencies?
- What is training-serving skew?

#### Further Reading

- FastAPI documentation.
- Docker documentation.
- Model serving platform docs.

### Monitoring

#### Big Picture

Monitoring tells you whether the deployed ML system is healthy. It includes
service health, data quality, model behavior, drift, and business impact.

#### Core Intuition

You usually do not get labels immediately in production. So you monitor proxies
such as input distributions, prediction distributions, confidence, latency,
errors, and delayed outcome labels when they arrive.

#### Mathematical Foundation

Drift detection compares distributions over time. Examples include summary
statistics, histograms, KL divergence, population stability index, and
statistical tests.

#### From Scratch Implementation

Log:

- request timestamp.
- feature summaries.
- prediction.
- confidence.
- latency.
- model version.

Compare recent feature distributions with training distributions.

#### Framework Implementation

Use observability tools, dashboards, alerts, and model-monitoring systems as
the platform matures.

#### Engineering Perspective

Monitor both software and model behavior. Alert fatigue is real, so alerts
should be actionable. Logging must respect privacy and retention policies.

#### Interview Questions

- What should you monitor for an ML model?
- Why are labels often delayed?
- What is data drift?
- What is prediction drift?
- How do you avoid alert fatigue?

#### Further Reading

- ML monitoring platform docs.
- Production ML reliability references.

### Scaling

#### Big Picture

Scaling ML systems means serving more traffic, larger models, bigger datasets,
or lower latency under cost constraints. Scaling is not just adding hardware;
it is choosing the right architecture.

#### Core Intuition

Online inference optimizes latency. Batch inference optimizes throughput.
Caching avoids repeated work. Queues smooth spikes. Quantization and
distillation reduce model cost.

#### Mathematical Foundation

Throughput, latency, memory, and cost are system metrics. Batch processing
amortizes overhead across many examples. Quantization reduces numeric precision
to improve speed and memory at possible accuracy cost.

#### From Scratch Implementation

Benchmark:

- single request latency.
- batch latency.
- throughput.
- memory usage.

Add simple request batching conceptually.

#### Framework Implementation

Use autoscaling, queues, caching layers, optimized runtimes, and hardware-aware
deployment strategies.

#### Engineering Perspective

Measure before optimizing. The bottleneck might be tokenization, network IO,
feature lookup, model compute, database calls, or serialization. Cost is a
first-class engineering metric.

#### Interview Questions

- What is the difference between batch and online inference?
- How does caching help?
- What is quantization?
- What is autoscaling?
- How do you identify a bottleneck?

#### Further Reading

- Model serving and distributed systems references.
- PyTorch and ONNX deployment docs.

### Failure Cases

#### Big Picture

Production ML fails in ways that pure software and pure modeling do not.
Failures come from data leakage, drift, feedback loops, bad metrics,
misaligned incentives, pipeline bugs, unsafe automation, and unclear ownership.

#### Core Intuition

A model is part of a system. Even a high-quality model can fail if the data
changes, features are computed differently in production, the metric rewards
the wrong behavior, or users adapt to the model.

#### Mathematical Foundation

Failure analysis often compares distributions, error slices, calibration,
counterfactual behavior, and time-based performance. The key question is not
only "Did the metric drop?" but "Where, when, for whom, and why?"

#### From Scratch Implementation

Write incident-style case studies:

- what happened.
- impact.
- detection.
- root cause.
- mitigation.
- prevention.

Add pre-deployment checklists for leakage, schema, metrics, monitoring, and
rollback.

#### Framework Implementation

Use model cards, data cards, incident reviews, evaluation dashboards, and
release gates.

#### Engineering Perspective

Mature ML teams design for failure. Every production model needs monitoring,
rollback, ownership, and a clear definition of acceptable behavior. Many ML
failures are socio-technical: the model, product, users, and organization
interact.

#### Interview Questions

- What is data leakage?
- What is training-serving skew?
- What is concept drift?
- How can feedback loops harm models?
- What should a model incident review include?

#### Further Reading

- Production ML case studies.
- Model cards and data cards literature.
- MLOps reliability references.

