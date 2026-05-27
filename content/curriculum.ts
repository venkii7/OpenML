export type CurriculumLevel = {
  id: string;
  slug: string;
  title: string;
  description: string;
  modules: string[];
};

export const curriculum: CurriculumLevel[] = [
  {
    id: "Level 0",
    slug: "prerequisites",
    title: "Prerequisites",
    description: "The minimum toolbelt for doing ML seriously without drowning in notation or tooling.",
    modules: [
      "Python for ML",
      "Linear algebra for ML",
      "Probability and statistics",
      "Calculus intuition",
      "NumPy and Pandas",
      "Git, Linux, and reproducible environments"
    ]
  },
  {
    id: "Level 1",
    slug: "core-machine-learning",
    title: "Core Machine Learning",
    description: "Classical supervised learning, optimization, evaluation, and feature thinking.",
    modules: [
      "Linear Regression",
      "Logistic Regression",
      "Gradient Descent, SGD, and Adam",
      "Loss Functions",
      "Bias-Variance",
      "Evaluation Metrics",
      "Feature Engineering",
      "Trees, Ensembles, SVMs, kNN, and Naive Bayes"
    ]
  },
  {
    id: "Level 2",
    slug: "deep-learning",
    title: "Deep Learning",
    description: "Neural networks from scratch, backpropagation, architectures, and scalable training.",
    modules: [
      "Neural Networks from Scratch",
      "Backpropagation",
      "CNNs",
      "RNNs and LSTMs",
      "Transformers",
      "Regularization",
      "Training at Scale"
    ]
  },
  {
    id: "Level 3",
    slug: "modern-ai-llms",
    title: "Modern AI and LLMs",
    description: "Tokenization, embeddings, transformers, fine-tuning, retrieval, agents, and multimodality.",
    modules: [
      "Attention Mechanism",
      "Transformers from Scratch",
      "Tokenization",
      "Embeddings",
      "Fine-tuning",
      "RAG Systems",
      "Agents",
      "Multimodal Models"
    ]
  },
  {
    id: "Level 4",
    slug: "mlops-engineering",
    title: "MLOps and Engineering",
    description: "The production discipline around data, experiments, serving, monitoring, scaling, and incidents.",
    modules: [
      "Data Pipelines",
      "Model Versioning",
      "Experiment Tracking",
      "Deployment with Docker and APIs",
      "Monitoring",
      "Scaling",
      "Failure Cases"
    ]
  },
  {
    id: "Level 5",
    slug: "research-advanced",
    title: "Research and Advanced Topics",
    description: "Paper reading, architecture tradeoffs, open problems, and community study practices.",
    modules: [
      "Papers Explained",
      "New Architectures",
      "Tradeoffs",
      "Open Problems",
      "Reading Groups"
    ]
  }
];
