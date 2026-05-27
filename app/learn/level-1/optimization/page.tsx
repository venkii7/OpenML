import type { ReactNode } from "react";

const optimizerImplementation = `import numpy as np

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd = wd
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1
        grad = grad + self.wd * theta
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr = lr
        self.mu = momentum
        self.nesterov = nesterov
        self.v = None

    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)

        self.v = self.mu * self.v + (1 - self.mu) * grad
        update = self.mu * self.v + (1 - self.mu) * grad if self.nesterov else self.v
        return theta - self.lr * update`;

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

function LandscapeVisuals() {
  return (
    <div className="landscape-grid">
      <div className="landscape-cell">
        <div className="landscape-title">Saddle Point</div>
        <svg viewBox="0 0 120 90" width="120" height="90" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="opt-a1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <polygon points="0 0, 6 3, 0 6" fill="#9b2335" />
            </marker>
          </defs>
          <path d="M10,50 C30,10 90,10 110,50" stroke="#d8d3c8" strokeWidth="1" fill="none" />
          <path d="M10,50 C30,80 90,80 110,50" stroke="#d8d3c8" strokeWidth="1" fill="none" />
          <circle cx="60" cy="50" r="3" fill="#2a7a6e" />
          <path d="M45,50 L57,50" stroke="#9b2335" strokeWidth="1.5" markerEnd="url(#opt-a1)" />
          <path d="M75,50 L63,50" stroke="#9b2335" strokeWidth="1.5" markerEnd="url(#opt-a1)" />
        </svg>
        <p>Gradient is near zero, but the point is not a minimum. Optimizers can stall here.</p>
      </div>
      <div className="landscape-cell">
        <div className="landscape-title">Ravine</div>
        <svg viewBox="0 0 120 90" width="120" height="90" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="opt-a2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <polygon points="0 0, 6 3, 0 6" fill="#9b2335" />
            </marker>
          </defs>
          <ellipse cx="60" cy="45" rx="50" ry="10" fill="none" stroke="#d8d3c8" strokeWidth="1" />
          <ellipse cx="60" cy="45" rx="35" ry="7" fill="none" stroke="#d8d3c8" strokeWidth="1" />
          <ellipse cx="60" cy="45" rx="20" ry="4" fill="none" stroke="#d8d3c8" strokeWidth="1" />
          <ellipse cx="60" cy="45" rx="6" ry="1.5" fill="#2a7a6e" opacity="0.4" />
          <path d="M40,25 L55,43" stroke="#9b2335" strokeWidth="1.5" markerEnd="url(#opt-a2)" />
          <path d="M80,65 L65,47" stroke="#9b2335" strokeWidth="1.5" markerEnd="url(#opt-a2)" />
        </svg>
        <p>Curvature varies across dimensions, so vanilla gradients zig-zag through narrow valleys.</p>
      </div>
      <div className="landscape-cell">
        <div className="landscape-title">Local Minima</div>
        <svg viewBox="0 0 120 90" width="120" height="90" xmlns="http://www.w3.org/2000/svg">
          <path d="M10,20 C20,20 28,65 40,65 C52,65 55,30 65,30 C75,30 78,70 90,70 C102,70 108,40 110,40" stroke="#2a7a6e" strokeWidth="2" fill="none" />
          <circle cx="40" cy="65" r="3" fill="#b8860b" />
          <circle cx="90" cy="70" r="3" fill="#2a7a6e" />
          <text x="25" y="82" fontSize="8" fill="#b8860b">local min</text>
          <text x="78" y="85" fontSize="8" fill="#2a7a6e">global</text>
        </svg>
        <p>Non-convex losses can have local minima that trap deterministic optimizers.</p>
      </div>
    </div>
  );
}

function AdamFlow() {
  return (
    <div className="adam-flow">
      <div className="adam-row">
        <div className="adam-step-label">Step 1</div>
        <div className="adam-desc">Compute the mini-batch gradient: g_t = grad_theta L_t(theta_(t-1)).</div>
      </div>
      <div className="adam-row">
        <div className="adam-step-label">Step 2</div>
        <div className="adam-desc">Update the first moment m_t, an exponential moving average of gradients.</div>
      </div>
      <div className="adam-row">
        <div className="adam-step-label">Step 3</div>
        <div className="adam-desc">Update the second moment v_t, an exponential moving average of squared gradients.</div>
      </div>
      <div className="adam-row">
        <div className="adam-step-label">Step 4</div>
        <div className="adam-desc">Apply bias correction to m_t and v_t because both were initialized at zero.</div>
      </div>
      <div className="adam-row">
        <div className="adam-step-label">Step 5</div>
        <div className="adam-desc">Take a scale-normalized parameter step with m_hat_t / (sqrt(v_hat_t) + epsilon).</div>
      </div>
    </div>
  );
}

export default function OptimizationLesson() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-10 lesson-prose">
      <header className="doc-header">
        <p className="doc-label">Technical Reference - Deep Learning Foundations</p>
        <h1 className="mt-3 text-4xl font-black">Optimization in Machine Learning</h1>
        <p className="doc-subtitle">Gradient Descent, Stochastic Gradient Descent, Momentum, RMSProp, and Adam.</p>
        <div className="doc-meta">
          <div>Depth <span>Advanced</span></div>
          <div>Prerequisites <span>Calculus, Linear Algebra, ML Basics</span></div>
          <div>Coverage <span>Theory, Derivation, Algorithms, Intuition, Code</span></div>
        </div>
      </header>

      <nav className="toc" aria-label="Lesson contents">
        <p className="toc-title">Contents</p>
        <ol>
          <li><a href="#s1">The Optimization Problem</a></li>
          <li><a href="#s2">Gradient Descent</a></li>
          <li><a href="#s3">Batch, SGD, Mini-batch</a></li>
          <li><a href="#s4">Learning Rate and Convergence</a></li>
          <li><a href="#s5">Challenges in Optimization</a></li>
          <li><a href="#s6">Momentum</a></li>
          <li><a href="#s7">RMSProp</a></li>
          <li><a href="#s8">Adam Optimizer</a></li>
          <li><a href="#s9">Bias Correction</a></li>
          <li><a href="#s10">AdaGrad and AdaDelta</a></li>
          <li><a href="#s11">Comparative Analysis</a></li>
          <li><a href="#s12">Practical Guidelines</a></li>
        </ol>
      </nav>

      <p className="section-number">01 -</p>
      <h2 id="s1">The Optimization Problem</h2>
      <p>Training a model means finding parameter values that minimize a scalar loss function over a dataset.</p>
      <Eq label="(1.1)">{`theta* = argmin_{theta in R^d} L(theta)`}</Eq>
      <Eq label="(1.2)">{`L(theta) = (1/n) sum_i loss(f(x_i; theta), y_i)`}</Eq>
      <Callout title="Why Analytic Solutions Fail" tone="definition">
        Modern neural network losses are high-dimensional, non-convex surfaces. Setting grad L = 0 usually gives a
        system with no closed-form solution, so iterative gradient-based methods are the practical path.
      </Callout>
      <h3>The Gradient as a Direction</h3>
      <p>The gradient points in the direction of steepest ascent. Moving opposite to it descends the loss surface.</p>
      <Eq label="(1.3)">{`grad_theta L = [dL/dtheta_1, dL/dtheta_2, ..., dL/dtheta_d]^T`}</Eq>
      <Eq label="(1.4)">{`D_v L = grad_theta L^T v = ||grad_theta L|| ||v|| cos(angle)`}</Eq>

      <p className="section-number">02 -</p>
      <h2 id="s2">Gradient Descent (Batch GD)</h2>
      <p>Batch gradient descent computes the gradient over the full dataset before every parameter update.</p>
      <Eq label="(2.1)">{`theta_(t+1) = theta_t - alpha grad_theta L(theta_t)`}</Eq>
      <div className="algo-box">
        <div className="algo-header"><span>Algorithm 1 - Batch Gradient Descent</span><span className="algo-tag">Full Dataset Per Update</span></div>
        <div className="algo-body">
          <strong>Input:</strong> data D, learning rate alpha, iterations T<br />
          <strong>Initialize:</strong> theta_0<br /><br />
          <strong>For</strong> t = 0, 1, ..., T - 1:<br />
          <div className="indent">Compute g_t = (1/n) sum_i grad_theta loss(f(x_i; theta_t), y_i).<br />Update theta_(t+1) = theta_t - alpha g_t.</div>
        </div>
      </div>
      <h3>Convergence for Convex Functions</h3>
      <Eq label="(2.2)">{`L(theta_T) - L(theta*) <= ||theta_0 - theta*||^2 / (2 alpha T)`}</Eq>
      <Callout title="Smoothness Condition" tone="proof">
        For an L-smooth objective, the gradient is Lipschitz-continuous. This gives a quadratic upper bound on
        the loss and explains why alpha &lt;= 1/L makes consistent progress.
      </Callout>
      <h3>Taylor Expansion Justification</h3>
      <Eq label="(2.3)">{`L(theta_t + delta) ~= L(theta_t) + grad L(theta_t)^T delta`}</Eq>

      <p className="section-number">03 -</p>
      <h2 id="s3">Gradient Descent Variants</h2>
      <p>The variants differ by how many samples are used per gradient estimate.</p>
      <div className="variant-grid">
        <div className="variant-card v-gd"><div className="variant-head">Batch GD</div><div className="variant-body">Uses all n samples. Accurate gradient, high cost, deterministic path.</div></div>
        <div className="variant-card v-sgd"><div className="variant-head">SGD</div><div className="variant-body">Uses one random sample. Noisy but cheap, often improves generalization.</div></div>
        <div className="variant-card v-mini"><div className="variant-head">Mini-batch</div><div className="variant-body">Uses B samples. Standard hardware-friendly training loop.</div></div>
      </div>
      <h3>Stochastic Gradient Descent</h3>
      <Eq label="(3.1)">{`g_t^SGD = grad_theta loss(f(x^(i_t); theta_t), y^(i_t))`}</Eq>
      <Eq label="(3.2)">{`theta_(t+1) = theta_t - alpha_t g_t^SGD`}</Eq>
      <Eq label="(3.3)">{`E[g_t^SGD] = grad_theta L(theta_t)`}</Eq>
      <Callout title="The Noise Advantage" tone="insight">
        SGD noise acts as implicit regularization. It can help escape shallow local minima and saddle points, and
        often pushes training toward flatter basins.
      </Callout>
      <h3>Mini-batch SGD</h3>
      <Eq label="(3.4)">{`g_t^mini = (1/B) sum_{i in B_t} grad_theta loss(f(x_i; theta_t), y_i)`}</Eq>
      <Eq label="(3.5)">{`Var[g_t^mini] = (1/B) Var[g_t^SGD]`}</Eq>

      <p className="section-number">04 -</p>
      <h2 id="s4">Learning Rate and Convergence Theory</h2>
      <p>The learning rate is the most sensitive hyperparameter: too small is slow, too large oscillates or diverges.</p>
      <Eq label="(4.1)">{`sum_t alpha_t = infinity,   sum_t alpha_t^2 < infinity`}</Eq>
      <Eq label="(4.2)">{`E[L((1/T) sum_t theta_t)] - L(theta*) <= O(1 / sqrt(T))`}</Eq>
      <p>Common schedules include step decay, exponential decay, cosine annealing, linear warmup, and one-cycle schedules.</p>

      <p className="section-number">05 -</p>
      <h2 id="s5">Challenges in Optimization</h2>
      <p>Vanilla GD and SGD struggle with ill-conditioning, ravines, saddle points, local minima, and noisy gradients.</p>
      <LandscapeVisuals />
      <h3>The Vanishing / Exploding Gradient Problem</h3>
      <Eq label="(5.1)">{`||dL/dtheta_1|| ~= product_{k=1}^L sigma_k -> 0 if sigma_k < 1`}</Eq>

      <p className="section-number">06 -</p>
      <h2 id="s6">Momentum</h2>
      <p>Momentum accumulates a velocity vector that damps oscillations and accelerates progress along consistent directions.</p>
      <Eq label="(6.1)">{`v_(t+1) = beta v_t + (1 - beta) g_t`}</Eq>
      <Eq label="(6.2)">{`theta_(t+1) = theta_t - alpha v_(t+1)`}</Eq>
      <Callout title="Exponential Moving Average Interpretation" tone="insight">
        The velocity is a geometrically weighted sum of past gradients. For beta = 0.9, the effective window spans
        roughly 10 gradients.
      </Callout>
      <h3>Nesterov Accelerated Gradient</h3>
      <Eq label="(6.3)">{`v_(t+1) = beta v_t + alpha grad L(theta_t - beta v_t)`}</Eq>
      <Eq label="(6.4)">{`theta_(t+1) = theta_t - v_(t+1)`}</Eq>
      <Callout title="Convergence Acceleration" tone="definition">
        For convex L-smooth objectives, Nesterov acceleration reaches O(1/T^2), improving on GD's O(1/T).
      </Callout>

      <p className="section-number">07 -</p>
      <h2 id="s7">RMSProp</h2>
      <p>RMSProp tracks an exponential moving average of squared gradients and normalizes each coordinate.</p>
      <Eq label="(7.1)">{`s_(t+1) = rho s_t + (1 - rho) g_t * g_t`}</Eq>
      <Eq label="(7.2)">{`theta_(t+1) = theta_t - alpha g_t / (sqrt(s_(t+1)) + epsilon)`}</Eq>
      <Callout title="Geometric Intuition" tone="insight">
        Parameters with consistently large gradients get smaller effective learning rates; flat or sparse
        dimensions get larger effective learning rates.
      </Callout>

      <p className="section-number">08 -</p>
      <h2 id="s8">Adam: Adaptive Moment Estimation</h2>
      <p>Adam combines momentum with RMSProp by tracking first and second moments of the gradient.</p>
      <AdamFlow />
      <div className="algo-box">
        <div className="algo-header"><span>Algorithm 3 - Adam</span><span className="algo-tag">Defaults: beta1=0.9, beta2=0.999, eps=1e-8</span></div>
        <div className="algo-body">
          g_t = grad L_t(theta_(t-1))<br />
          m_t = beta1 m_(t-1) + (1 - beta1) g_t<br />
          v_t = beta2 v_(t-1) + (1 - beta2) g_t^2<br />
          m_hat_t = m_t / (1 - beta1^t)<br />
          v_hat_t = v_t / (1 - beta2^t)<br />
          theta_t = theta_(t-1) - alpha m_hat_t / (sqrt(v_hat_t) + epsilon)
        </div>
      </div>
      <Eq label="(8.1)">{`|Delta theta_j^t| = alpha |m_hat_j^t| / (sqrt(v_hat_j^t) + epsilon)`}</Eq>

      <p className="section-number">09 -</p>
      <h2 id="s9">Bias Correction in Adam</h2>
      <p>Both moments start at zero, so early estimates are biased downward unless corrected.</p>
      <Eq>{`m_t = (1 - beta1) sum_{k=1}^t beta1^(t-k) g_k`}</Eq>
      <Eq>{`m_hat_t = m_t / (1 - beta1^t),   v_hat_t = v_t / (1 - beta2^t)`}</Eq>
      <Callout title="Warm-Up Effect" tone="note">
        Without bias correction, Adam's early effective learning rate is artificially small. Modern transformer
        training often adds a separate linear warmup on top of Adam or AdamW.
      </Callout>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Parameter</th><th>Default</th><th>Role</th></tr></thead>
          <tbody>
            <tr><td>beta1</td><td>0.9</td><td>Momentum window, roughly 10 steps.</td></tr>
            <tr><td>beta2</td><td>0.999</td><td>Scale normalization window, roughly 1000 steps.</td></tr>
            <tr><td>epsilon</td><td>1e-8</td><td>Numerical stability.</td></tr>
            <tr><td>alpha</td><td>1e-3</td><td>Global step size.</td></tr>
          </tbody>
        </table>
      </div>

      <p className="section-number">10 -</p>
      <h2 id="s10">AdaGrad and AdaDelta</h2>
      <h3>AdaGrad</h3>
      <Eq label="(10.1)">{`G_t = sum_{k=1}^t g_k * g_k`}</Eq>
      <Eq label="(10.2)">{`theta_(t+1) = theta_t - alpha g_t / sqrt(G_t + epsilon)`}</Eq>
      <p>AdaGrad is strong for sparse gradients but its accumulator grows forever, shrinking learning rates toward zero.</p>
      <h3>AdaDelta</h3>
      <Eq label="(10.3)">{`s_t = rho s_(t-1) + (1-rho) g_t^2,   Delta theta_t = -sqrt(u_(t-1)+eps)/sqrt(s_t+eps) * g_t`}</Eq>
      <Eq label="(10.4)">{`u_t = rho u_(t-1) + (1-rho) Delta theta_t^2,   theta_(t+1) = theta_t + Delta theta_t`}</Eq>

      <p className="section-number">11 -</p>
      <h2 id="s11">Comparative Analysis</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Optimizer</th><th>Memory</th><th>Adaptive LR</th><th>Momentum</th><th>Best For</th></tr></thead>
          <tbody>
            <tr><td>Batch GD</td><td>O(d)</td><td>No</td><td>No</td><td>Small datasets and convex losses.</td></tr>
            <tr><td>SGD</td><td>O(d)</td><td>No</td><td>No</td><td>Large datasets, generalization-critical training.</td></tr>
            <tr><td>SGD + Momentum</td><td>O(2d)</td><td>No</td><td>Yes</td><td>Vision and structured problems.</td></tr>
            <tr><td>AdaGrad</td><td>O(2d)</td><td>Yes</td><td>No</td><td>Sparse NLP and linear models.</td></tr>
            <tr><td>RMSProp</td><td>O(2d)</td><td>Yes</td><td>No</td><td>RNNs and online learning.</td></tr>
            <tr><td>Adam</td><td>O(3d)</td><td>Yes</td><td>Yes</td><td>Transformers, GANs, most deep networks.</td></tr>
            <tr><td>AdamW</td><td>O(3d)</td><td>Yes</td><td>Yes</td><td>LLMs and ViTs with weight decay.</td></tr>
          </tbody>
        </table>
      </div>
      <h3>AdamW: Decoupled Weight Decay</h3>
      <Eq label="(11.1)">{`theta_t = theta_(t-1) - alpha m_hat_t/(sqrt(v_hat_t)+eps) - alpha lambda theta_(t-1)`}</Eq>
      <Callout title="Adam's Generalization Gap" tone="warning">
        Adam can converge faster but sometimes generalizes worse than SGD with momentum, likely because it can find
        sharper minima. AdamW fixes the weight-decay interaction and is the default for many modern architectures.
      </Callout>

      <p className="section-number">12 -</p>
      <h2 id="s12">Practical Guidelines and Hyperparameter Tuning</h2>
      <div className="algo-box">
        <div className="algo-header">Decision Tree - Optimizer Selection</div>
        <div className="algo-body">
          <strong>If</strong> training a Transformer: use AdamW with cosine decay and linear warmup.<br />
          <strong>Else if</strong> training a CNN for top accuracy: use SGD + Nesterov momentum.<br />
          <strong>Else if</strong> sparse data or recommendation features: use Adam or AdaGrad.<br />
          <strong>Else if</strong> prototyping: start with Adam at alpha = 1e-3.
        </div>
      </div>
      <h3>Gradient Clipping</h3>
      <Eq label="(12.1)">{`g_t <- g_t * min(1, tau / ||g_t||)`}</Eq>
      <h3>Learning Rate Finder</h3>
      <Eq label="(12.2)">{`alpha_t = alpha_min * (alpha_max / alpha_min)^(t/T)`}</Eq>

      <h2>Implementation</h2>
      <pre>{optimizerImplementation}</pre>

      <h2>Engineering Perspective</h2>
      <ul>
        <li>Inspect loss curves; optimizer bugs often look like bad models.</li>
        <li>Tune learning rate before most other hyperparameters.</li>
        <li>Use mini-batches sized for hardware throughput, then scale alpha carefully.</li>
        <li>Use AdamW, not Adam plus L2, when you intend true weight decay.</li>
      </ul>

      <h2>Interview Questions</h2>
      <ul>
        <li>Why does the negative gradient reduce loss locally?</li>
        <li>Compare batch GD, SGD, and mini-batch SGD.</li>
        <li>What problem does momentum solve?</li>
        <li>How does RMSProp differ from AdaGrad?</li>
        <li>Why does Adam need bias correction?</li>
      </ul>
    </main>
  );
}
