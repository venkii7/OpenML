# Open ML Engineer Content Plan

This file is the production checklist for turning the curriculum into a full
open-source textbook. Each topic should become a lesson page with the standard
template:

1. Big Picture
2. Core Intuition
3. Mathematical Foundation
4. From Scratch Implementation
5. Framework Implementation
6. Engineering Perspective
7. Interview Questions
8. Further Reading

## Source Policy

Use primary and durable sources first:

- Official docs: Python, NumPy, pandas, scikit-learn, PyTorch, TensorFlow/Keras,
  Hugging Face, Docker, Git.
- Original papers: Adam, Attention Is All You Need, RAG, foundational CNN/RNN
  and regularization papers.
- High-quality books: ESL, PRML, Deep Learning, Dive into Deep Learning.
- Engineering docs and postmortems for MLOps topics.

Avoid using low-quality blog posts as canonical references. Blogs can be useful
for intuition, but they should not be the source of truth.

## Level 0 - Prerequisites

### Python for ML

Goal: Make learners productive in scientific Python, not just basic syntax.

Must cover:

- Runtime model: interpreter, scripts, notebooks, modules, packages.
- Data model: numbers, strings, lists, tuples, dicts, sets, iterators.
- Functions: arguments, defaults, closures, typing, docstrings.
- OOP only where useful: classes for datasets, configs, experiment wrappers.
- Exceptions and debugging.
- Virtual environments, dependency pinning, project layout.
- Performance mindset: vectorization first, Python loops when acceptable,
  profiling before optimizing.

From scratch tasks:

- Implement train/test split without scikit-learn.
- Implement mini-batch iteration over arrays.
- Implement a simple experiment config object.

Engineering notes:

- Reproducibility requires pinned dependencies, seeds, and saved configs.
- Avoid global mutable state in experiments.
- Keep notebooks for exploration; move reusable logic into modules.

### Math for ML: Linear Algebra

Goal: Teach vectors, matrices, and tensor thinking as the language of models.

Must cover:

- Scalars, vectors, matrices, tensors.
- Dot product as similarity and projection.
- Matrix multiplication as composition of linear maps.
- Norms, distance, cosine similarity.
- Rank, span, basis, independence.
- Eigenvectors, eigenvalues, PCA intuition.
- Broadcasting and shapes in NumPy/PyTorch.

From scratch tasks:

- Implement dot product, matrix multiplication, vector norm.
- Implement cosine similarity.
- Implement PCA with centering, covariance, eigendecomposition.

Engineering notes:

- Most ML bugs are shape bugs.
- Keep batch dimension explicit.
- Numerical stability matters: prefer library routines over manual inverses.

### Math for ML: Probability

Goal: Build the probabilistic language behind uncertainty, data generation,
classification, and evaluation.

Must cover:

- Random variables, events, probability mass/density.
- Conditional probability and Bayes rule.
- Expectation, variance, covariance.
- Common distributions: Bernoulli, categorical, Gaussian, binomial.
- Maximum likelihood estimation.
- Independence assumptions and why they fail.
- Calibration and uncertainty.

From scratch tasks:

- Estimate a Bernoulli parameter by maximum likelihood.
- Implement Gaussian log-likelihood.
- Simulate sampling variance with repeated experiments.

Engineering notes:

- Probabilities are model claims, not facts.
- Calibration matters when predictions drive decisions.
- Log probabilities avoid underflow.

### Math for ML: Calculus Intuition

Goal: Make gradients feel like operational tools for optimization.

Must cover:

- Functions as landscapes.
- Derivative as local rate of change.
- Partial derivatives and gradients.
- Chain rule as the backbone of backpropagation.
- Jacobians only where needed.
- Taylor approximation intuition.
- Convex vs non-convex objectives.

From scratch tasks:

- Numerically approximate derivatives.
- Compare finite differences with analytic gradients.
- Implement gradient descent on a simple quadratic.

Engineering notes:

- Gradient checks catch implementation bugs.
- Learning rate controls update size, not intelligence.
- Non-convex optimization is about useful minima, not guaranteed global minima.

### Data Handling: NumPy

Goal: Teach array programming as the foundation for implementing ML algorithms.

Must cover:

- ndarray, dtype, shape, strides.
- Indexing, slicing, boolean masks.
- Broadcasting.
- Vectorization.
- Reductions and aggregations.
- Random number generation.
- Saving/loading arrays.

From scratch tasks:

- Normalize a feature matrix.
- Implement vectorized linear regression predictions.
- Implement a mini-batch sampler.

Engineering notes:

- Avoid accidental copies with large arrays.
- Be explicit about dtype for memory and precision.
- Broadcasting should clarify code, not hide shape confusion.

### Data Handling: Pandas

Goal: Teach tabular data inspection, cleaning, joining, and feature preparation.

Must cover:

- Series and DataFrame.
- Indexes and column selection.
- Missing values.
- Groupby and aggregation.
- Joins and merges.
- Time series basics.
- CSV/parquet IO.
- Converting to NumPy arrays for modeling.

From scratch tasks:

- Clean a messy tabular dataset.
- Build aggregate features with groupby.
- Split features and target safely.

Engineering notes:

- Avoid data leakage while aggregating.
- Be careful with chained assignment.
- Track schema expectations before training.

### Git and Linux Basics

Goal: Make learners comfortable working like engineers.

Must cover:

- Shell navigation, paths, pipes, redirects.
- Files, permissions, environment variables.
- Git working tree, staging area, commits.
- Branches, merges, pull requests.
- `.gitignore`, diffs, logs, blame.
- Reproducible command-line workflows.

From scratch tasks:

- Create a repo, commit a change, create a branch, merge it.
- Use shell commands to inspect dataset files.
- Write a simple training command with arguments.

Engineering notes:

- Commit small logical changes.
- Never commit secrets, datasets, or generated artifacts unless intended.
- Use branches for reviewable work.

## Level 1 - Core Machine Learning

### Linear Regression

Status: first lesson exists. Expand later with normal equation, multivariate
regression, regularized regression, residual diagnostics, and real dataset
case study.

### Logistic Regression

Must cover:

- Binary classification and decision boundaries.
- Sigmoid function as mapping from logit to probability.
- Log loss / binary cross entropy.
- Maximum likelihood view.
- Gradient descent training.
- Thresholds, calibration, precision/recall tradeoff.

From scratch tasks:

- Implement sigmoid, binary cross entropy, gradients.
- Train logistic regression in NumPy.
- Plot decision boundary on 2D data.

Framework tasks:

- scikit-learn `LogisticRegression`.
- PyTorch single-layer classifier.

Engineering notes:

- Logistic regression is linear in features, not in probabilities.
- Class imbalance changes threshold choice and metric selection.
- Regularization changes coefficients and generalization.

### Optimization: GD, SGD, Adam

Must cover:

- Objective functions and parameter updates.
- Batch gradient descent.
- Stochastic and mini-batch gradient descent.
- Momentum.
- Adaptive learning rates.
- Adam: first and second moment estimates, bias correction.
- Learning rate schedules.

From scratch tasks:

- Implement GD, SGD, momentum, Adam.
- Compare convergence on convex and noisy objectives.

Engineering notes:

- Adam is often a strong default for deep learning, not a universal best.
- Batch size affects noise, throughput, and generalization.
- Always monitor loss curves and gradient norms.

### Loss Functions

Must cover:

- MSE, MAE, Huber.
- Binary cross entropy, multiclass cross entropy.
- Hinge loss.
- Negative log likelihood.
- Ranking/contrastive losses.
- Loss vs metric distinction.

From scratch tasks:

- Implement common losses and gradients.
- Compare sensitivity to outliers.

Engineering notes:

- Optimize a differentiable proxy, evaluate with task metrics.
- Loss choice encodes assumptions about errors.
- Reduction mode matters for distributed training.

### Bias-Variance

Must cover:

- Underfitting and overfitting.
- Approximation error, estimation error, irreducible noise.
- Train/validation curves.
- Regularization and data size.
- Cross-validation.

From scratch tasks:

- Fit polynomials of increasing degree.
- Plot train vs validation error.

Engineering notes:

- High variance is not fixed only by simpler models; more data and better
  regularization can help.
- High bias often needs better features, architecture, or objective.

### Evaluation Metrics

Must cover:

- Regression: MAE, MSE, RMSE, R2, MAPE.
- Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- Ranking/retrieval: MRR, MAP, NDCG, recall@k.
- Calibration: Brier score, reliability curves.
- Offline vs online metrics.

From scratch tasks:

- Implement confusion matrix and derived metrics.
- Plot ROC and precision-recall curves.

Engineering notes:

- Pick metrics based on business cost, not habit.
- Accuracy fails on imbalanced datasets.
- Offline metrics can disagree with production impact.

### Feature Engineering

Must cover:

- Numeric scaling and transformations.
- Categorical encoding.
- Text features.
- Date/time features.
- Aggregations.
- Interactions.
- Leakage prevention.
- Feature selection.

From scratch tasks:

- Build a preprocessing pipeline.
- Compare one-hot, ordinal, and target encoding conceptually.

Engineering notes:

- Feature engineering is production code.
- Training-serving skew is a common failure mode.
- Validate feature schemas continuously.

### Classical ML Algorithms

Must cover:

- k-nearest neighbors.
- Naive Bayes.
- Decision trees.
- Random forests.
- Gradient boosted trees.
- SVMs.
- k-means.
- PCA.

From scratch tasks:

- Implement kNN and a small decision tree.
- Implement k-means.

Engineering notes:

- Tree ensembles are extremely strong for tabular data.
- kNN is simple but expensive at inference.
- SVMs require careful kernels and scaling.

## Level 2 - Deep Learning

### Neural Networks From Scratch

Must cover:

- Neurons, layers, activations.
- Forward pass.
- Parameters and computation graphs.
- Initialization.
- Batch processing.

From scratch tasks:

- Implement dense layer, ReLU, softmax.
- Train a small MLP with NumPy.

Engineering notes:

- Initialization and normalization strongly affect trainability.
- Debug tiny networks before large ones.

### Backpropagation

Must cover:

- Chain rule.
- Computational graphs.
- Reverse-mode automatic differentiation.
- Gradients for affine layer, activation, loss.
- Gradient checking.

From scratch tasks:

- Manual backprop for MLP.
- Tiny autodiff engine for scalar values.

Engineering notes:

- Backprop is bookkeeping plus chain rule.
- Autograd hides mechanics, not math.
- In-place ops can break gradient computation.

### CNNs

Must cover:

- Convolution, filters, channels.
- Padding, stride, receptive fields.
- Pooling.
- CNN blocks.
- Image augmentation.
- Transfer learning.

From scratch tasks:

- Implement 2D convolution for one channel.
- Train a small CNN in PyTorch.

Engineering notes:

- CNNs exploit spatial locality and weight sharing.
- Data augmentation often matters more than architecture tweaks.

### RNNs and LSTMs

Must cover:

- Sequence modeling.
- Hidden state.
- Backpropagation through time.
- Vanishing/exploding gradients.
- LSTM gates.
- GRU comparison.

From scratch tasks:

- Implement a simple RNN cell.
- Implement LSTM equations.

Engineering notes:

- RNNs are useful for streaming and small sequence systems.
- Transformers dominate many batch sequence tasks, but not every deployment.

### Transformers Deep Dive

Must cover:

- Self-attention.
- Multi-head attention.
- Positional encodings.
- Encoder-decoder vs decoder-only.
- Residual connections, layer norm, MLP blocks.
- Masking.
- Complexity and context length.

From scratch tasks:

- Implement scaled dot-product attention.
- Implement a small transformer block.

Engineering notes:

- Attention is flexible but quadratic in sequence length.
- Masking bugs can silently leak future tokens.
- Normalization placement affects stability.

### Regularization

Must cover:

- L1/L2 weight penalties.
- Dropout.
- Early stopping.
- Data augmentation.
- Label smoothing.
- Batch norm and layer norm as related stabilizers.

From scratch tasks:

- Add L2 penalty and dropout to an MLP.
- Compare train/validation curves.

Engineering notes:

- Regularization is task-dependent.
- Too much regularization underfits.
- Data quality often beats clever penalties.

### Training at Scale

Must cover:

- GPUs and tensor parallelism basics.
- DataLoader throughput.
- Mixed precision.
- Gradient accumulation.
- Distributed data parallel.
- Checkpointing.
- Reproducibility.

From scratch tasks:

- Profile a training loop.
- Add mixed precision and gradient accumulation.

Engineering notes:

- Bottlenecks are often data loading, not model compute.
- Checkpoints need model, optimizer, scheduler, scaler, and config state.

## Level 3 - Modern AI and LLMs

### Attention Mechanism

Must cover:

- Query, key, value.
- Dot-product attention.
- Scaling by sqrt(d_k).
- Masks.
- Attention weights as routing, not explanation by default.

From scratch tasks:

- Implement attention over a toy sequence.
- Visualize attention matrix.

Engineering notes:

- Attention weights are useful diagnostics but not complete interpretability.
- Long context creates memory and latency pressure.

### Transformers From Scratch

Must cover:

- Token embeddings.
- Position information.
- Attention block.
- Feed-forward block.
- Residual stream.
- Layer norm.
- Autoregressive generation.

From scratch tasks:

- Build a tiny decoder-only transformer.
- Train on a character-level dataset.

Engineering notes:

- The forward pass must match masking and shape conventions exactly.
- Generation quality depends on decoding parameters and training data.

### Tokenization

Must cover:

- Characters, words, subwords.
- BPE, WordPiece, unigram tokenization.
- Vocabulary size tradeoffs.
- Special tokens.
- Padding, truncation, attention masks.

From scratch tasks:

- Implement toy BPE merges.
- Compare token counts across examples.

Engineering notes:

- Tokenization affects cost, latency, context use, and multilingual behavior.
- Always inspect tokenization for domain-specific text.

### Embeddings

Must cover:

- Dense vector representations.
- Similarity metrics.
- Word/document embeddings.
- Contrastive learning intuition.
- Retrieval embeddings vs generative model embeddings.
- Vector indexes.

From scratch tasks:

- Build cosine similarity search over small documents.
- Evaluate retrieval with recall@k.

Engineering notes:

- Embedding quality is domain dependent.
- Chunking strategy can dominate RAG quality.

### Fine-Tuning

Must cover:

- Full fine-tuning vs parameter-efficient fine-tuning.
- Supervised fine-tuning.
- LoRA intuition.
- Dataset formatting.
- Evaluation and overfitting.
- Safety and regression tests.

From scratch tasks:

- Fine-tune a small classifier.
- Demonstrate adapter-style low-rank update conceptually.

Engineering notes:

- Fine-tuning changes behavior; it is not a knowledge database.
- Use RAG for frequently changing facts.
- Keep a held-out eval set that reflects production use.

### RAG Systems

Must cover:

- Retrieval-augmented generation architecture.
- Chunking, embeddings, indexing.
- Query rewriting.
- Reranking.
- Context assembly.
- Citation and grounding.
- Evaluation: retrieval and answer quality.

From scratch tasks:

- Build a tiny local RAG pipeline over Markdown files.
- Measure recall@k and answer faithfulness manually.

Engineering notes:

- Most RAG failures are ingestion, chunking, retrieval, or evaluation failures.
- Updating an index is cheaper than retraining a model.
- Log retrieved context for debugging.

### Agents

Must cover:

- Tool use.
- Planning vs reactive loops.
- State, memory, and constraints.
- Human approval boundaries.
- Evaluation and observability.
- Failure modes: loops, unsafe actions, tool misuse.

From scratch tasks:

- Build a small tool-using loop with explicit state.
- Add guardrails and step budget.

Engineering notes:

- Agents are distributed systems with an LLM in the loop.
- Tool permissions and audit logs matter.
- Prefer narrow tools and clear contracts.

### Multimodal Models

Must cover:

- Text, image, audio, video modalities.
- Encoders and fusion.
- Vision-language models.
- Contrastive pretraining.
- OCR and document understanding.
- Evaluation challenges.

From scratch tasks:

- Compare image embeddings with text embeddings conceptually.
- Build a small image captioning or VQA demo using a pretrained model later.

Engineering notes:

- Multimodal systems fail at modality alignment and grounding.
- Data rights, privacy, and evaluation are harder than in text-only systems.

## Level 4 - MLOps and Engineering

### Data Pipelines

Must cover:

- Batch vs streaming.
- Ingestion, validation, transformation.
- Feature pipelines.
- Orchestration.
- Data contracts.
- Backfills.

From scratch tasks:

- Build a simple reproducible ETL script.
- Validate schema and missingness.

Engineering notes:

- ML systems inherit data pipeline failures.
- Silent schema drift is dangerous.
- Make pipelines idempotent when possible.

### Model Versioning

Must cover:

- Dataset, code, config, and model artifact lineage.
- Model registry.
- Promotion stages.
- Reproducibility.
- Rollbacks.

From scratch tasks:

- Save a model artifact with metadata.
- Track model version and training config.

Engineering notes:

- A model file without lineage is not production-ready.
- Version the full training recipe, not just weights.

### Experiment Tracking

Must cover:

- Parameters, metrics, artifacts.
- Runs and experiments.
- Comparing runs.
- Checkpoints.
- Reproducibility and collaboration.

From scratch tasks:

- Log metrics to JSON/CSV.
- Compare several model runs.

Engineering notes:

- Track fewer high-quality signals before logging everything.
- Name experiments and runs intentionally.

### Deployment: Docker and APIs

Must cover:

- Model serialization.
- REST APIs with FastAPI.
- Docker images.
- Dependency isolation.
- CPU/GPU serving.
- Health checks.

From scratch tasks:

- Serve a model behind a `/predict` endpoint.
- Containerize the app.

Engineering notes:

- Separate training dependencies from serving dependencies.
- Validate request/response schemas.
- Cold start and latency matter.

### Monitoring

Must cover:

- Service metrics: latency, errors, throughput.
- Model metrics: prediction distribution, confidence, drift.
- Data quality monitoring.
- Feedback loops.
- Alerting.

From scratch tasks:

- Log predictions and input summary statistics.
- Detect feature drift with simple distribution checks.

Engineering notes:

- You rarely have labels immediately in production.
- Monitor both system health and model behavior.
- Alert fatigue is real.

### Scaling

Must cover:

- Batch inference vs online inference.
- Caching.
- Queues.
- Autoscaling.
- Quantization and distillation.
- Hardware tradeoffs.

From scratch tasks:

- Benchmark batch vs single prediction latency.
- Add simple request batching conceptually.

Engineering notes:

- Scaling starts with measurement.
- Optimize the bottleneck, not the most interesting component.
- Cost is an engineering metric.

### Failure Cases

Must cover:

- Data leakage.
- Training-serving skew.
- Distribution shift.
- Concept drift.
- Feedback loops.
- Silent pipeline failures.
- Bad metrics.
- Unsafe automation.

From scratch tasks:

- Write incident-style case studies.
- Add pre-deployment checklists.

Engineering notes:

- Mature ML teams design for failure.
- Every production model needs rollback, monitoring, and ownership.
- Most failures are socio-technical, not only mathematical.

