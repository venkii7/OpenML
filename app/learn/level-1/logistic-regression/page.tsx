import type { ReactNode } from "react";

const pythonImplementation = `import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, lam=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.lam = lam

    def _sigma(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        X_b = np.c_[np.ones(n), X]
        self.theta = np.zeros(d + 1)

        for _ in range(self.n_iter):
            p = self._sigma(X_b @ self.theta)
            grad = X_b.T @ (p - y) / n
            reg = self.lam * self.theta / n
            reg[0] = 0.0
            self.theta -= self.lr * (grad + reg)

    def predict_proba(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return self._sigma(X_b @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def log_loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))`;

function Eq({ children, label }: { children: string; label?: string }) {
  return (
    <div className="eq-block">
      <code>{children}</code>
      {label ? <span className="eq-label">{label}</span> : null}
    </div>
  );
}

function Callout({ title, tone, children }: { title: string; tone: string; children: ReactNode }) {
  return (
    <div className={`callout ${tone}`}>
      <div className="callout-title">{title}</div>
      {children}
    </div>
  );
}

function SigmoidVisual() {
  return (
    <div className="sigmoid-visual">
      <div className="sigmoid-curve">
        <svg viewBox="0 0 280 180" xmlns="http://www.w3.org/2000/svg" style={{ width: "100%", display: "block" }}>
          <defs>
            <marker id="sigmoid-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <polygon points="0 0, 7 3.5, 0 7" fill="#5c5852" />
            </marker>
          </defs>
          <line x1="20" y1="45" x2="265" y2="45" stroke="#d8d3c8" strokeWidth="0.5" strokeDasharray="3,3" />
          <line x1="20" y1="90" x2="265" y2="90" stroke="#d8d3c8" strokeWidth="0.5" strokeDasharray="3,3" />
          <line x1="20" y1="135" x2="265" y2="135" stroke="#d8d3c8" strokeWidth="0.5" strokeDasharray="3,3" />
          <line x1="20" y1="160" x2="265" y2="160" stroke="#5c5852" strokeWidth="1" markerEnd="url(#sigmoid-arrow)" />
          <line x1="142" y1="160" x2="142" y2="12" stroke="#5c5852" strokeWidth="1" markerEnd="url(#sigmoid-arrow)" />
          <text x="268" y="163" fontSize="11" fill="#9a9690">z</text>
          <text x="145" y="10" fontSize="11" fill="#9a9690">sigma(z)</text>
          <text x="6" y="93" fontSize="10" fill="#9a9690">0.5</text>
          <text x="12" y="48" fontSize="10" fill="#9a9690">1</text>
          <text x="12" y="138" fontSize="10" fill="#9a9690">0</text>
          <line x1="20" y1="90" x2="142" y2="90" stroke="#2a7a6e" strokeWidth="0.8" strokeDasharray="4,3" />
          <path d="M20,153 C35,151 50,148 60,143 C75,135 88,120 105,105 C118,94 127,88 142,83 C158,78 165,72 178,65 C192,57 207,48 222,43 C235,39 250,37 260,36" stroke="#2a7a6e" strokeWidth="2.2" fill="none" />
          <circle cx="142" cy="90" r="3.5" fill="#2a7a6e" />
          <text x="148" y="87" fontSize="9" fill="#2a7a6e">inflection</text>
          <text x="22" y="41" fontSize="9" fill="#9a9690">y -&gt; 1</text>
          <text x="22" y="155" fontSize="9" fill="#9a9690">y -&gt; 0</text>
        </svg>
      </div>
      <div className="sigmoid-desc">
        <h3 style={{ marginTop: 0 }}>Key Properties</h3>
        <ul>
          <li>Range strictly (0, 1), so the output is a valid probability.</li>
          <li>sigma(0) = 0.5 at the symmetric inflection point.</li>
          <li>Monotonically increasing and smooth everywhere.</li>
          <li>Asymptotes approach 1 for large positive z and 0 for large negative z.</li>
          <li>Identity: sigma(-z) = 1 - sigma(z).</li>
        </ul>
      </div>
    </div>
  );
}

export default function LogisticRegressionLesson() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-10 lesson-prose">
      <header className="doc-header">
        <p className="doc-label">Technical Reference - Machine Learning</p>
        <h1 className="mt-3 text-4xl font-black">Logistic Regression</h1>
        <p className="doc-subtitle">A complete treatment from first principles to production diagnostics.</p>
        <div className="doc-meta">
          <div>Depth <span>Advanced</span></div>
          <div>Prerequisites <span>Linear Algebra, Probability, Calculus</span></div>
          <div>Coverage <span>Theory, Derivation, Implementation, Diagnostics</span></div>
        </div>
      </header>

      <nav className="toc" aria-label="Lesson contents">
        <p className="toc-title">Contents</p>
        <ol>
          <li><a href="#s1">Motivation and Problem Setting</a></li>
          <li><a href="#s2">The Sigmoid Function</a></li>
          <li><a href="#s3">Model Formulation</a></li>
          <li><a href="#s4">Probabilistic Interpretation</a></li>
          <li><a href="#s5">Log-Loss and MLE</a></li>
          <li><a href="#s6">Gradient Descent Optimization</a></li>
          <li><a href="#s7">Decision Boundary</a></li>
          <li><a href="#s8">Regularization</a></li>
          <li><a href="#s9">Multiclass Extension</a></li>
          <li><a href="#s10">Evaluation and Diagnostics</a></li>
        </ol>
      </nav>

      <p className="section-number">01 -</p>
      <h2 id="s1">Motivation and Problem Setting</h2>
      <p>
        Logistic regression is a discriminative probabilistic classifier used for binary, and by extension
        multiclass, classification tasks. The word regression refers to fitting a linear function to the log-odds
        of the outcome.
      </p>
      <Eq label="(1.1)">{`D = {(x^(i), y^(i))}_{i=1}^n,   x^(i) in R^d,   y^(i) in {0, 1}`}</Eq>
      <p>
        The goal is to learn h_theta(x), a probability estimate for P(y = 1 | x; theta). This is useful when the
        decision depends on calibrated risk, not only a hard class label.
      </p>

      <h3>Why not Linear Regression for Classification?</h3>
      <Eq>{`y_hat = theta^T x in (-infinity, +infinity)`}</Eq>
      <Callout title="Critical Flaw" tone="warning">
        Linear predictions are unbounded, so they can exceed 1 or fall below 0. Squared error is also mismatched to
        Bernoulli labels and does not give the likelihood-based objective logistic regression needs.
      </Callout>

      <p className="section-number">02 -</p>
      <h2 id="s2">The Sigmoid Function</h2>
      <p>The sigmoid, or logistic function, maps a real-valued score into the probability interval.</p>
      <Eq label="(2.1)">{`sigma(z) = 1 / (1 + e^(-z)) = e^z / (1 + e^z)`}</Eq>
      <SigmoidVisual />
      <h3>Derivative of the Sigmoid</h3>
      <Eq label="(2.2)">{`d sigma / dz = sigma(z) (1 - sigma(z))`}</Eq>
      <Callout title="Derivation" tone="insight">
        Starting from sigma(z) = (1 + e^(-z))^-1, the chain rule gives e^(-z) / (1 + e^(-z))^2,
        which factors into sigma(z)(1 - sigma(z)).
      </Callout>

      <p className="section-number">03 -</p>
      <h2 id="s3">Model Formulation</h2>
      <p>
        With a bias-augmented feature vector x and parameter vector theta, the linear score is z = theta^T x.
        The model passes that score through the sigmoid.
      </p>
      <Eq label="(3.1)">{`h_theta(x) = sigma(theta^T x) = 1 / (1 + exp(-theta^T x))`}</Eq>
      <Callout title="Interpretation" tone="definition">
        h_theta(x) estimates P(y = 1 | x; theta). The complementary probability is P(y = 0 | x; theta) =
        1 - h_theta(x).
      </Callout>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>Shape</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>x_i</td><td>R^d</td><td>Feature vector, usually augmented with x_0 = 1 for the bias.</td></tr>
            <tr><td>y_i</td><td>{"{0, 1}"}</td><td>Binary class label.</td></tr>
            <tr><td>theta</td><td>R^(d+1)</td><td>Weight vector including bias theta_0.</td></tr>
            <tr><td>X</td><td>n x (d+1)</td><td>Design matrix of all training examples.</td></tr>
          </tbody>
        </table>
      </div>

      <p className="section-number">04 -</p>
      <h2 id="s4">Probabilistic Interpretation</h2>
      <p>The odds of an event are p / (1 - p). Logistic regression assumes log-odds are linear in the features.</p>
      <Eq label="(4.1)">{`Odds = P(y = 1 | x) / P(y = 0 | x) = h_theta(x) / (1 - h_theta(x))`}</Eq>
      <Eq label="(4.2)">{`logit(p) = log(p / (1 - p)) = theta^T x`}</Eq>
      <Callout title="Key Insight" tone="insight">
        Logistic regression fits a linear model to the log-odds. The sigmoid is the inverse-logit link function.
      </Callout>
      <h3>Bernoulli Likelihood</h3>
      <Eq label="(4.3)">{`P(y | x; theta) = h_theta(x)^y (1 - h_theta(x))^(1-y)`}</Eq>

      <p className="section-number">05 -</p>
      <h2 id="s5">The Log-Loss Cost Function</h2>
      <p>Binary cross-entropy, also called log-loss, is the negative average log-likelihood.</p>
      <Eq label="(5.1)">{`L(theta) = -(1/n) sum_i [ y_i log(h_i) + (1 - y_i) log(1 - h_i) ]`}</Eq>
      <Callout title="Non-Convexity Problem" tone="warning">
        Using MSE with sigmoid outputs creates a non-convex landscape. Log-loss is derived from maximum likelihood
        and gives the convex objective used for logistic regression.
      </Callout>
      <h3>Convexity Proof Sketch</h3>
      <Eq label="(5.2)">{`H = (1/n) X^T S X,   S = diag(y_hat_i (1 - y_hat_i))`}</Eq>

      <h2>Maximum Likelihood Estimation</h2>
      <Eq label="(6.1)">{`Likelihood(theta; D) = product_i h_i^(y_i) (1 - h_i)^(1-y_i)`}</Eq>
      <Eq label="(6.2)">{`log Likelihood(theta) = sum_i [y_i log(h_i) + (1-y_i) log(1-h_i)]`}</Eq>
      <Callout title="No Closed-Form Solution" tone="definition">
        Unlike ordinary least squares, logistic regression has no algebraic closed-form solution for theta. It
        requires iterative optimization such as gradient descent or Newton's method.
      </Callout>

      <p className="section-number">06 -</p>
      <h2 id="s6">Gradient Descent Optimization</h2>
      <p>The sigmoid derivative cancels cleanly inside the log-loss derivative.</p>
      <Eq label="(7.1)">{`dL/d theta_j = (1/n) sum_i (h_theta(x_i) - y_i) x_ij`}</Eq>
      <Eq label="(7.2)">{`grad_theta L = (1/n) X^T (y_hat - y)`}</Eq>
      <Eq label="(7.3)">{`theta <- theta - alpha * grad_theta L = theta - (alpha/n) X^T (y_hat - y)`}</Eq>
      <div className="algo-box">
        <div className="algo-header">Algorithm - Stochastic Gradient Descent</div>
        <div className="algo-body">
          <strong>Input:</strong> training set D, learning rate alpha, epochs T<br />
          <strong>Initialize:</strong> theta = 0 or small random values<br /><br />
          <strong>For</strong> each epoch:<br />
          <div className="indent">
            Shuffle D. For each sample, compute y_hat = sigma(theta^T x), error e = y_hat - y, and update
            theta = theta - alpha e x.
          </div>
        </div>
      </div>
      <h3>Newton-Raphson Method</h3>
      <Eq label="(7.4)">{`theta <- theta - H^(-1) grad L = theta - (X^T S X)^(-1) X^T (y_hat - y)`}</Eq>
      <p>Newton-Raphson is also known as iteratively reweighted least squares. It is fast on small datasets but expensive in high dimensions.</p>

      <p className="section-number">07 -</p>
      <h2 id="s7">Decision Boundary</h2>
      <Eq label="(8.1)">{`y_hat_class = 1 if h_theta(x) >= 0.5, else 0`}</Eq>
      <Eq label="(8.2)">{`theta^T x = 0  <=>  theta_0 + theta_1 x_1 + ... + theta_d x_d = 0`}</Eq>
      <Callout title="Non-Linear Boundaries via Feature Expansion" tone="note">
        Polynomial and interaction features create non-linear boundaries in the original input space while keeping
        the classifier linear in the augmented feature space.
      </Callout>

      <p className="section-number">08 -</p>
      <h2 id="s8">Regularization</h2>
      <p>Regularization discourages large coefficients and reduces overfitting when d is large relative to n.</p>
      <Eq label="(9.1)">{`L_ridge(theta) = L(theta) + (lambda / 2n) ||theta_-0||_2^2`}</Eq>
      <Eq label="(9.2)">{`L_lasso(theta) = L(theta) + (lambda / n) ||theta_-0||_1`}</Eq>
      <Eq label="(9.3)">{`L_EN(theta) = L(theta) + (lambda/n)[((1-rho)/2)||theta_-0||_2^2 + rho||theta_-0||_1]`}</Eq>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Penalty</th><th>Effect</th><th>Best For</th></tr></thead>
          <tbody>
            <tr><td>L2</td><td>Shrinks all coefficients smoothly.</td><td>Many small predictors.</td></tr>
            <tr><td>L1</td><td>Can shrink coefficients exactly to zero.</td><td>High-dimensional sparse features.</td></tr>
            <tr><td>Elastic Net</td><td>Combines shrinkage and sparsity.</td><td>Correlated features with sparse signal.</td></tr>
          </tbody>
        </table>
      </div>
      <Callout title="Bayesian Equivalence" tone="insight">
        L2 regularization corresponds to a Gaussian prior. L1 regularization corresponds to a Laplace prior.
      </Callout>

      <p className="section-number">09 -</p>
      <h2 id="s9">Multiclass Extension</h2>
      <h3>One-vs-Rest</h3>
      <Eq>{`y_hat = argmax_k h_{theta_k}(x)`}</Eq>
      <h3>Multinomial Softmax Regression</h3>
      <Eq label="(10.1)">{`P(y = k | x; Theta) = exp(theta_k^T x) / sum_j exp(theta_j^T x)`}</Eq>
      <Eq label="(10.2)">{`L_CE(Theta) = -(1/n) sum_i sum_k 1[y_i = k] log P(y = k | x_i; Theta)`}</Eq>
      <Callout title="Identifiability" tone="definition">
        Softmax has redundant parameters because shifting all class scores by the same constant leaves
        probabilities unchanged. One class is often chosen as a reference.
      </Callout>

      <p className="section-number">10 -</p>
      <h2 id="s10">Evaluation, Calibration, and Diagnostics</h2>
      <p>
        Evaluate with a confusion matrix, accuracy, precision, recall, F1, specificity, log-loss, ROC-AUC, and
        calibration. ROC curves plot true positive rate against false positive rate as the threshold varies.
      </p>
      <Eq>{`TPR = TP / (TP + FN),   FPR = FP / (FP + TN)`}</Eq>
      <Eq label="(11.1)">{`Brier Score = (1/n) sum_i (p_i - y_i)^2`}</Eq>
      <div className="props-grid">
        <div className="prop-card"><div className="prop-label">Assumption 1</div><strong>Binary outcome</strong><p>Use proper extensions for multiclass tasks.</p></div>
        <div className="prop-card"><div className="prop-label">Assumption 2</div><strong>Independence</strong><p>Repeated or clustered observations need correction.</p></div>
        <div className="prop-card"><div className="prop-label">Assumption 3</div><strong>No perfect collinearity</strong><p>Check feature redundancy and VIF.</p></div>
        <div className="prop-card"><div className="prop-label">Assumption 4</div><strong>Log-odds linearity</strong><p>The logit, not the probability, is linear.</p></div>
        <div className="prop-card"><div className="prop-label">Assumption 5</div><strong>Large sample</strong><p>MLE is more reliable with 10-20 events per predictor.</p></div>
        <div className="prop-card"><div className="prop-label">Assumption 6</div><strong>No perfect separation</strong><p>Separating features can make coefficients diverge.</p></div>
      </div>
      <Callout title="Hauck-Donner Effect" tone="warning">
        Under perfect separation, coefficients can move toward infinity and the model degenerates into assigning
        probabilities near 0 or 1. L2 regularization, Firth likelihood, or feature changes are common remedies.
      </Callout>
      <Eq label="(12.1)">{`chi^2_HL = sum_g (O_g - E_g)^2 / [E_g (1 - E_g/n_g)]`}</Eq>

      <h2>Implementation Reference</h2>
      <pre>{pythonImplementation}</pre>

      <h2>Engineering Perspective</h2>
      <ul>
        <li>Use logistic regression as a strong interpretable baseline for tabular classification.</li>
        <li>Standardize features when optimization speed or coefficient comparison matters.</li>
        <li>Choose thresholds based on decision cost, not by blindly using 0.5.</li>
        <li>Use log-loss and calibration checks when probabilities drive decisions.</li>
        <li>Watch class imbalance; accuracy can be misleading when positives are rare.</li>
      </ul>

      <h2>Interview Questions</h2>
      <ul>
        <li>Why is logistic regression a classifier despite the name?</li>
        <li>Derive the sigmoid derivative and binary cross-entropy gradient.</li>
        <li>Why use log-loss instead of mean squared error?</li>
        <li>What happens under perfect separation?</li>
      </ul>
    </main>
  );
}
