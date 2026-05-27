import { LinearRegressionPlayground } from "@/components/interactive/LinearRegressionPlayground";

export default function LinearRegressionLesson() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-10 lesson-prose">
      <p className="text-sm font-bold uppercase tracking-[0.08em] text-accent">Level 1 - Core Machine Learning</p>
      <h1 className="mt-3 text-4xl font-black">Linear Regression</h1>

      <h2>Big Picture</h2>
      <p>
        Linear regression learns a numeric relationship between input features
        and a continuous target. In real systems it appears in forecasting,
        pricing, demand estimation, anomaly baselines, A/B test analysis, and
        as the simplest model you should beat before using something heavier.
      </p>

      <h2>Core Intuition</h2>
      <p>
        A linear model draws a line through data. Each point has an actual
        value and a predicted value. The vertical distance between them is the
        residual. A good model chooses slope and intercept values that make the
        residuals small across the dataset, not just for one point.
      </p>
      <LinearRegressionPlayground />

      <h2>Mathematical Foundation</h2>
      <p>
        For one feature, the model is y_hat = wx + b. The training objective is
        usually mean squared error: average((y - y_hat)^2). Squaring makes large
        errors expensive and gives a smooth objective for optimization.
      </p>
      <p>
        The gradient tells us how changing w or b changes the loss. For a batch
        of n examples, dL/dw = (-2/n) sum x_i(y_i - y_hat_i), and dL/db =
        (-2/n) sum(y_i - y_hat_i). Gradient descent updates parameters in the
        opposite direction of these gradients.
      </p>

      <h2>From Scratch Implementation</h2>
      <pre>{`import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1.2, 1.9, 3.1, 3.8, 5.2], dtype=float)

w = 0.0
b = 0.0
lr = 0.01

for step in range(1000):
    y_hat = w * X + b
    error = y - y_hat
    loss = np.mean(error ** 2)

    dw = (-2 / len(X)) * np.sum(X * error)
    db = (-2 / len(X)) * np.sum(error)

    w -= lr * dw
    b -= lr * db

print(w, b, loss)`}</pre>

      <h2>Framework Implementation</h2>
      <pre>{`import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[1.2], [1.9], [3.1], [3.8], [5.2]])

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for step in range(1000):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

print(model.weight.item(), model.bias.item())`}</pre>

      <h2>Engineering Perspective</h2>
      <ul>
        <li>Use it as a baseline before gradient boosted trees or neural networks.</li>
        <li>Standardize features when using gradient-based optimization.</li>
        <li>Watch for leakage, outliers, correlated features, and non-linear residual patterns.</li>
        <li>Prefer MAE or Huber loss when large outliers should not dominate training.</li>
        <li>Inspect residual plots; metrics alone can hide systematic failure.</li>
      </ul>

      <h2>Interview Questions</h2>
      <ul>
        <li>Why does mean squared error penalize outliers more than mean absolute error?</li>
        <li>Derive the gradient of MSE for simple linear regression.</li>
        <li>When can linear regression perform better than a deep model?</li>
        <li>How would you detect feature leakage in a regression problem?</li>
        <li>What changes when you move from one feature to many features?</li>
      </ul>

      <h2>Further Reading</h2>
      <ul>
        <li>Elements of Statistical Learning, chapters on linear methods.</li>
        <li>Pattern Recognition and Machine Learning, linear models for regression.</li>
        <li>scikit-learn linear model documentation.</li>
        <li>PyTorch autograd tutorials.</li>
      </ul>
    </main>
  );
}
