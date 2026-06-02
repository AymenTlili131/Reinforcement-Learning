# Reinforcement Learning — Handwritten Notes Transcription

> *Transcribed from 39 scanned notebook pages (IMG_20260601_091807 – IMG_20260601_092641)*

---

## Part 1: The Explore–Exploit Dilemma

### The Lazy Programmer's Artificial Intelligence & Reinforcement Planning

- **explore – exploit** dilemma
- Does an experiment have a huge reward? → **Markov** → exploit
- $\mu = \bar{X} = \frac{1}{N}\sum_{k=0}^{N} X_k$ → best estimate

We do the experiment 1 million times for each non-trivial task:

1. How do we know what is a realistic sample size for the experiment? → collect $\infty$
2. → doing the worst → #0 of times → **Huge loss** (unoptimal)
3. We are measuring a success rate + permitting we are measuring a reward for a Gaussian

---

### 1st Solution: ε-Greedy

- $\varepsilon$ is a small number; at each round we choose to **explore** or **exploit**
- If $\varepsilon$ is too small → explore = 100 best
- If random $< \varepsilon$ → explore; strategy will not be optimal
- If $n \to \infty$ → ε-greedy allows comparing the finds sample size for the experiment, because all rates will converge
  - **find estimated value**

**A/B Testing** can be helpful at a predetermined time to test after $t_0$ (statistical significance found)

---

### 2nd Solution: Optimistic Initial Values

- True mean $\ll$ estimated value (high ceiling) → goes down  
  → **Exploitation**: sample near then converges
  - Estimated sample start → **maximum** → true mean → converges
- If you don't explore a bandit, the plan will remain high → **more exploration**
- It won't be forced to explore the greedy policy + use optimistic values from last

---

### 3rd Solution: UCB1 — Upper Confidence Bounds

- Confidence bounds = UCB1
- Small sample → large confidence bounds; big sample → small confidence bound

**Chebyshev–Hoeffding bound:**

$$P\bigl(|\bar{X} - \mu| \geq \varepsilon\bigr) \leq 2e^{-2N\varepsilon^2}$$

- **Upper bound:** $\hat{\varepsilon} = \sqrt{\frac{2\ln N}{N_j}}$

$$X_{\text{UCB}} = \bar{X}_j + \sqrt{\frac{2\ln(N)}{N_j}}$$

**Be greedy** (but will respect to $X_{\text{UCB}-j}$):
- $N_j$ small → $N_j \ll \ln(N) \Rightarrow \hat{\varepsilon}$ large → using only sample → you'll have large uncertainty
- $N_j$ large → you'll have large confidence → small interval

**For** $t = 1 \ldots N$:

$$\text{argmax}[\bar{X}_j] = \text{argmax}\left[\bar{X}_j + \sqrt{\frac{2\ln(n)}{N_j}} + \varepsilon_j\right]$$

- On being large → if $N_j \ll$ if $N_j$ → very small, start → not chosen → prob(start, r=1 | y, start) = 0.8
- prob(safe, r=0 | y, start) = 0.5
- prob(hit, r = -1 | d, start) = 0.4
- prob(safe, r = 0 | duck, start) = 0.6
- → all other probabilities conditional yield 0

---

## Part 2: Estimating Bandit Rewards

### Estimating Bandit Rewards

$$\bar{X} = \frac{1}{N}\sum_{k=0}^{N} X_k \qquad C_l(0, 1) \neq \text{mean} \quad \leftarrow N \text{ flips}$$

**Problem:** Storing all history + updating requires recalculating

**Solution:**

$$X_n = \frac{1}{N}\sum_{k=1}^{N} X_k = \frac{N-1}{N}\bar{X}_{n-1} + \frac{1}{N}X_N$$

$$= \bar{X}_{n-1} + \frac{1}{N}(X_N - \bar{X}_{n-1})$$

---

### 4th Method: Bayesian Sampling / Thompson Sampling

- Confidence intervals → approximately Gaussian
- $\bar{X} \sim \mathcal{N}\!\left(\mu,\, \frac{\delta^2}{N}\right)$, $\bar{X} \sim \mathcal{N}\!\left(\frac{1}{\sqrt{N}}(\mu, \delta^2)\right)$

**What if $\mu$ is also a random variable?** $\theta$  
- Parameters are fixed
- Parameters are random → **Bayesian**

$$P(\theta | X) = \frac{P(X|\theta) \cdot P(\theta)}{P(\theta)} = \frac{\int P(X|\theta) P(\theta)\, d\theta}{\text{likelihood} \times \text{prior}}$$

- → **posterior distribution**: approximate it if you can't
- Conjugate prior: special $p(x|a), p(a)$ pairs where we can solve analytically — i.e., integral

---

### Coin Toss → Bernoulli → Prior of Bernoulli

- $X_{1\theta} \to B(p)$ — prior of Bernoulli
- $\text{Beta}(\alpha, \beta) \equiv \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}$ — normalising factor
  - $\alpha$: reflect belief (prior)  
  - $\beta$: belief of prior information
- Prior: $P(\theta) = \frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)}$ ← mean
- $(\alpha, \beta) \in \mathbb{R}^2$ — hyperparameters of the prior hyperparameters = 1
- $\theta \in [0,1]$ — interior hyperparameters − 1

**Posterior:**

$$P(\theta|X) = \prod_{k=1}^{N} \theta^{x_k=1}(1-\theta)^{1(x_k=0)}\frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)}$$

$$= \theta^{\alpha + \sum 1(x_k=1)-1}(1-\theta)^{\beta + \sum 1(x_k=0)-1}$$

$$\Rightarrow \alpha' = \alpha + \#1_A \qquad \beta' = \beta + \#0_A$$

---

### Gaussian / Thompson Sampling — Data Likelihood

$$P(X|\theta) \text{ likelihood}: \quad X \sim \mathcal{N}(\mu_0, \lambda^{-1});\quad \mu \sim \mathcal{N}(m_0, \lambda_0^{-1})$$

**Update** $\mu$ and $N$:

$$P(\mu|X) = \frac{1}{\sqrt{2\pi}} \cdot e^{-\frac{N\sigma^2}{2}(\mu - \bar{\mu})^2}$$

$$= e^{-\frac{N\sigma_0}{2}(\mu - \mu_0)^2} \quad \text{← posterior Gaussian}$$

$$\Rightarrow \frac{1}{\sqrt{2\pi}} \cdot \frac{1}{\sqrt{2\pi}} \sqrt{\frac{N}{\delta^2}} \cdot e^{-\frac{1}{2}\left(\frac{(x_t - \mu)^2}{\delta^2/N}\right)} \sqrt{\frac{N}{2\pi\delta^2}}$$

$$= \frac{1}{2\delta^2}\left[(N_0 + N)\mu^2 - 2(m_0\lambda_0 + \sum x_t)\mu + (N_0\mu^2 - 2N_0\mu)\right]$$

$$= \frac{1}{2}\left[(N_0 + N)\left(\mu - \frac{m_0\lambda_0 + \sum x_t}{N_0 + N}\right)^2\right]$$

Here we are looking for a Gaussian — it's a Gaussian!

**Identity** $\hat{\theta}$:

$$\hat{\theta} = \mu_0 + t_0 \hat{\Sigma} X_t \to \text{no} \to A_0 t^{\hat{\Sigma}N}$$

- $P(A|X)$ sampling from Bayesian posterior:
  - $\mu$ is a random variable → can sample from it
  - more confident → more sample → more exploitation → **exploitation**
  - rough → rare → **explore**

---

## Part 3: The RL Framework

### Sensors → Environment → Feedback → Neurons → Parameterising

- **Finite states:** The factor $= \{0, X, \text{empty}\} = 3 \times 3$ grid = $3^9$ states

**MDP loop:**
1. Start in State $S(t)$
2. Take action $A(t)$
3. Receive reward $R(t+1)$: $(s, a, s') \equiv (5, a)$ helper

- A Tic-Tac-Toe board → has board pieces numbers
- Player 2 pieces in a row → block → player's board → bad for win
- Play 2 pieces in a row → add → bad to win

---

### Episodes

- One run of the game
- \# of episodes = hyperparameter
- Learning across multiple → episode increases performance task: 3 processors run → draw
- → **episodic task** → end → **episodic problem**
- cont. → stop → gradient (policy following)
- Beyond a certain $\theta$ we can't stop by moving point
  - some life → pole by moving pole
- The good reward to maintain → is better/worse:
  - In chess → like good is to win
  - sub-goals well → encourage multi-lined behaviour

---

### Credit Assignment

- **Credit assignment** — if 10 miles were seen with your app correctly, who gets the credit for the success → who gets the credit for the success
- **Delayed rewards** — When to choose bigger later rewards? VS short-term rewards, the long run
- What value should we attribute to state A?

  **Value(A) = 0.5 × 1 + 0.5 × 0 = 0.5**

  Value function → partially future rewards
  - Value(A) = 1 × 1 — reaching A is just as good as reaching B
  - since like my move from A is to B
  - Rewards are immediate; prefer states that should receive max

---

## Part 4: Markov Decision Processes

### MDP (Markov Decision Process)

$$P(s'_t, s_t) = P\!\left(s_t | s_{t+1}\right)^{X_{t-1} \cdots X_{t-2}}$$

$$P\!\left(X_{t+1} | X_t\right) = P\!\left(X_t | X_{t-1}, \ldots, X_2\right)$$

**Policies:**

$$\Pi = \Pi(a|s) \qquad \Pi_1 : \begin{cases} R=0 \\ R_{t-p}\end{cases} \qquad \Pi_2 : \begin{cases} R=0 \\ R_{t-p}\end{cases}$$

- $\Pi$ → Optimal policies are **not unique**

**Returns:** $G(t) = \sum_{t=1}^{+\infty} R(t+\xi)$

**Discounting future rewards** — $|\delta| < 1$:

$$G(t) = \sum_{\xi=0}^{\infty} \delta^\xi R(t+\xi) \qquad \text{(convergent)}$$

---

### State-Value Function

$$V(s) = \mathbb{E}\!\left[G \mid S_t = s\right] = \mathbb{E}\left[r + \delta V(s') \mid S_t = s\right]$$

$$V_\pi(s) = \mathbb{E}_\pi\!\left[G \mid S_t = s\right] = \mathbb{E}_\pi\!\left[\sum_{\xi=0}^{\infty} \delta^\xi R(t+\xi+1)\right]$$

**Action-Value function:**

$$Q(s, a) = \mathbb{E}\!\left[r + \delta V(s')\mid S_t = s, A_t = a\right]$$

**Optimal:**

$$Q^*(s,a) = \max_a \left[\mathbb{E}\!\left(Q = \frac{s, a}{1}\right)\right] = \mathbb{E}\!\left[R(t+1) + \delta V^*(s_{t+1}) \mid S_t = s, A_t = s\right]$$

$$= \mathbb{E}\!\left[R(t+1) + \delta V^*(s_{t+1})\right] \quad \forall s \in S,\; A_t = s$$

---

### Bellman Equations (recursive)

$$V_\pi(s) = \sum_a \Pi(a|s) \cdot \sum_{s'} P(s', r|s, a)\!\left[r + \delta V(s')\right]$$

$$V(s) = \mathbb{E}\!\left[r + \delta V(s')\right]$$

**Bellman Optimality Equations (one unique optimal value):**

$$V^*(s) = \max_a \mathbb{E}\!\left[R(t+1) + \delta V^*(S_{t+1}) \mid S_t = s, A_t = a\right]$$

$$V^*(s) = \max_a \sum_{s,r} P(s, a | s, a)\!\left[r + \delta V^*(s)\right]$$

**How to find V given $\pi$?** → finding $\Pi^*$ and $V^*$

- $P(r|s) = \sum_{s \in S} P(s', r|s, a)$
- $P(s'|s, a) = \sum_r P(s', r|s, a)$
- $I$ is in state $s$ in the future: $V(s) = \mathbb{E}[G|s]$ = the value of all future states (the value of all possible trajectories)

---

### Law of Total Expectation & Implementation

$$E[E(X)] \equiv E(X)$$

$$E_\pi[R(t+1) | S_t = s] = \sum_a \Pi(a|s) \sum_{s'} \sum_r P(s', r|s, a) \cdot r$$

$$E_\pi[\text{Any}|S_t = s] = \sum_a \Pi(a|s) \cdot E[\text{Any}]$$

$$E_\pi[E(X|Y)] = E(X) \quad \Leftarrow \text{Law of Total Expectation}$$

$$E\!\left(E(G(t+1)) | \text{Any}\right) = -\; \varepsilon^{S_{t+1} = S'}$$

**Implementing** $\Pi^*$:
1. Greedily choose the action that leads to the best $V(s)$
2. 1-step lookahead search
3. With $Q(s,a)$ choose diagram (over $a$)
4. $Q(s,a)$ → store the brute-force search result → $Q(s,a)$

---

## Part 5: Dynamic Programming

### I. Dynamic Programming — Iterative Policy Evaluation

$$V(s) = 0 \quad \forall s \in S \quad \text{(initially)}$$

**WHILE** True: $\Delta = 0$
- $V(s) = \sum_a \Pi(a|s) \cdot P(s' | s, a)\!\left[r + \delta V(s')\right]$
- $\Delta = \max(\Delta, |V_{\text{old}} - V(s)|)$
- if $\Delta < \text{threshold}$: break

→ **return V(s)** (instead of looping)

→ We can use the updated $V(s)$ for faster convergence  
→ **Policy evaluation** → control problem  
→ **Control** → $\Pi^*$ (policy improvement)

Finding $V(s)$: initial value ($V_0$)  
  I. Finding $\Pi$ → for I in range (max iteration): update $V(s)$  
  II. Finding $\Pi^*$  
  → print (change in $V(s)$ vs limit / policy)

---

### II. Policy Improvement / Policy Iteration

- Initialise value function and policy
- for $k = \infty$ → large (non-Markovian reward):
  - state, action, newrewards → play game (policy)
  - update value function $\&$ $\Pi$

**Find policy** (non-$\Pi$):
- if $|A| < \infty$ → find $a \in A$ → s.t. $Q_\Pi(s,a) > Q_\Pi(s)$ ← optimal policy $\Pi$
- if $|A| = \infty$: $\Pi' \geq \Pi$

**$V_\Pi(s) \leq V_{\Pi'}(s)$** ← indicates non-convergence (indicates convergence)

$$\Pi'(s) = \text{argmax}_{Q_\Pi}(s, a) = \text{argmax}_a \sum_{s,\pi} P(s', a | s, a)\!\left[r + \delta V_\Pi(s')\right]$$

→ **Value** $V$

---

### Policy Iteration Convergence Problem

- Policy changing → the policy stops changing
- All of the values stop changing
- → value starts changing → value stops changing: important

- Alternating between **policy evaluation** and **policy improvement** (Back-tracking)

**V(s)** = iterations + returns $(V(s))$ and policy $\Pi(s)$

1. Randomly initialise $V(s)$ and policy $\Pi$
2. $V(s) = \text{iterations}$ → (Bootstrapping + $\delta V(s)$)
3. policy-change = False — for all states:  
   - old = policy(s);  
   - policy(s) = argmax $Q_\Pi(s, a)$  
   - if policy(s) $\neq$ old: policy-changed = True  
   - if policy-changed → go back to 2

**Disadvantage:** slow — waiting for convergence (exploitation)  
→ winning → **policy iteration** improvement

---

### Value Iteration — 2nd Solution

**Value iteration** = 2nd solution to find Bellman:  
The necessarily greedy policy wouldn't change before V converges  
$\Rightarrow$ Key: No need to work policy evaluation  
$\Rightarrow$ when does policy iteration?

$$V_{k+1}(s) = \max_a \sum_{s,r} P(s', r|s,a)\!\left[r + \delta V_k(s)\right]$$

- **Online Evaluation** and important policy: calculating the policy **explicitly**
- doesn't read V for $(k+1)$-th value
- → calculating $V = V_\Pi$ **doesn't need** finish calculation
- → **policy iteration** — disadvantage: the whole environment from the environment
- **Bootstrapped value** (diagram): $V = V_\Pi$, $V^* = V_\Pi^*$ → policy iteration  
  → policy iteration → the world → **policy iteration** — improvement

---

## Part 6: Monte Carlo Methods

### II. Monte Carlo Methods

*(page 092454 — sparse, rest on back pages)*

**Episode:** $G = \lambda(1 + \gamma^1 + \gamma^2 + \ldots) = \lambda(t+1) + \gamma b(t+1)$

**Monte Carlo** → First Visit:
1. Every visit method: $t = 1 \times t = 3$ — are samples
2. Moving average (not recursion) when looping over ψ
   - values are only updated for visited states
   - only updating from "start" — more states will never be visited
- ← "**Exploring Starts**" method

**b) Only if 0.5 probability:** $\varepsilon = 0$ — cannot policy  
- $\Theta$ we don't know the action that leads to a better V(s)

**Use Exploring Starts** — to explore all $|S| \times |A|$ space action  
- $|S| \times |A|$ for each action (instead of $\frac{|A|}{|A|}$ — state actions)  
- ← random initial → (random actions)

**Remove Exploring Starts** (if we can't: is too good → more blocking bad position):  
→ the ε-greedy = $\Pi$ sometimes blocks

1. **Monte Carlo** — Cost to obtain samples and evaluate
2. **Improvement** = argmax over $a$:  
   $Q(s, a) = \Pi(a|s, Q)$

**Best policy:**  
$\Pi_b(a) = \text{argmax}(Q)$  
The epsilon-greedy = $\Pi$ — sometimes blocks

---

## Part 7: Approximation Methods

### Approximation Methods

1. = not useful until now → of all states and actions:  
   - estimate $V$ for all $S$
   - estimate $V$ for all states and actions

2. **Neural network** $\to$ $x = \text{state} \to x = \xi(a)$  
   - **Goal:** $V(s) = P(x; \theta)$ — differentiable

   **Linear** ↓ **deep learning**  
   **approximation**

3. Can't use it now will use first:  
   - Linear models are not very expressive → feature engineering
   - **I states:** $V(s)$ — estimate $V(s)$ gives... (1 predict)
   - **II. TD(0) prediction**
   - **III. SARSA** → replace $Q$ with linear function + control → approximator

---

### Linear Model for RL

- $V(s)$ for $k \in \mathbb{R}$
- $G(s) \in \mathbb{R}$
- We will be doing regression; ↓ Monte Carlo ↓ sample mean

$$\text{Error} = [E[G|s_1, s_2 \ldots s_T] - V(s)]^2 = [I \cdot G_{N+s} - V(s)]^2$$

- Treat $G_n$ as a Moving sample → minimise the (individually squared) differences simultaneously
- $e = \sum_i (\hat{y}_i - y_i)^2$ — $\hat{y}_i$: 1 sample $\to$ 1 gradient decent
  - at each step after we only have to look at
  - a learning note

$$\theta = \theta + \alpha \cdot (G - \hat{V}(s))$$

$$\hat{V}(s,\theta) = \theta \cdot e(s) = \theta^T \cdot \nabla_\theta \hat{V}(s,\theta) \times$$

$$\nabla_\theta \hat{V}(s,\theta) = x$$

$$\theta = \theta + \alpha(G - \hat{V}(s,\theta)) \times \nabla_\theta \hat{V}(s,\theta)$$

**Feature** = creating categories $g$:  
- → category 1 → feature creating $(\varepsilon_{(s)})$ for each state  
- → category 2

- $(x,y) \in \mathbb{R}^2$ → scale by $n = 0, \ldots, 3^2$ — will a $x$ $(x_1, x_2) \neq (x, j)$
  - The linear model is **not always** decreasing — **increasing** in monotone
  - **Solution** → create polynomial → Taylor expansion to approximate any function → for it works!

$($ Stop not parading $)$: $\hat{V}(s) = \hat{V}(s) + \alpha \cdot (G_s - \hat{V}(s)) \cdot \frac{\partial \hat{V}(s)}{\partial V(s)}$

**Feature** → ($\hat{V}_{n-1} = \hat{V}_n$ — equivalent):  
$V(s)$ is not parameterised

---

## Part 8: Temporal Difference (TD) Methods

### II. TD(0) Approximation

$$G \neq TD(0)$$
$$G = r + \delta V(s')$$

$$\theta = \theta + \alpha(r + \delta\hat{V}(s') - \hat{V}(s,\theta)) \times \nabla_\theta \hat{V}(s,\theta)$$

- Target represents neural model prediction
- This is called **semi-gradient descent**

---

### 2nd Technique for Solving MDPs: Temporal Difference Learning (TD)

- Fully online Bootstrapping
  - $(MC \neq)$ completing $(\theta)$ → anything
  - $[MC\,\#]$: $\mathcal{E}(N)$ → completing → $\{$Dynamic programming$\}$

**Finding V** given $\Pi$: TD(0) | Hyperparameters | Bootstrapping  
- approximates $V$
- $\cdot$ proportion of long form in each state  
- Control families: **SARSA**

**On average finding:**

$$V(S_t) \leftarrow V(S_t) + \alpha \cdot [G(t) - V(S_t)]$$

1. Exponential decay
2. Moving average

**Q-Learning:** — Bootstrapping:

$$V(S_t) \leftarrow V(S_t) + \alpha\cdot[G(t) - V(S_t)]$$

**On average finding:**  
$\text{SA}\equiv\text{ID}\times\frac{x}{(s,a)}$ = workaround $x$ → unique $x$  
→ word → unique $x = [...]$

**Goal** = find optimal $\Pi$: find optimal hyperparameters  
- a model hyperparameters: $1$ to $2$ → learn  
- learning rate decreases at different rates  
- ε: decreases at a different rate

---

### III. SARSA — non-greedy

- approximate $Q = \theta + \alpha(A + \delta\hat{Q}(\text{(next)}) - \hat{Q}(\text{(next)}))^2 / 2n$

**Feature:** $(m, n)$ → new plane  
$G = \theta + \alpha \times (A + \delta C, D, L, R)$  
$x = (c, e, NC, UD, L, R)$

**Batter features** for each state  
$x = [8 \times 8 = 64$ features, one of $d$ — append $]$ — features for $|A|$ actions  
After 3 actions × 4 … append if  

$\text{After}\ 6\ \text{states} \times 4\ \text{actions} = 36$

---

### Deep Reinforcement Learning

**Advanced AI: Deep Reinforcement Learning**

- Agent: State → **Input layer** → DNN $\Pi_\theta(S|a)$ → parameters → Action → Environment
  - Reward
  - Observed state

**Controls:**
- II. N-step method
- I. TD(A)
- Policy Gradients
- F. Network

**Actor/Brain function:**  
- R, B, F → N
- closely related SVM and NN

**II.** 1 — Controls with bias

---
*End of transcription — 39 pages*
