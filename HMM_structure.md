# Reference-Specific HMM Structure For Real-Time Alignment

## Purpose

This document defines a reference-specific Hidden Markov Model (HMM) for
real-time alignment of a live query performance to a fixed reference
performance. It is meant to be mathematically explicit enough to guide both
implementation and project writeup.

This formulation is intentionally:

- reference-specific rather than piece-specific
- compatible with online inference
- based on query chroma vectors as observations
- based on Gaussian emission models
- trainable from pseudo-labels produced by offline DTW

## Problem Setting

We observe a live query performance frame by frame and want to estimate its
current position in a fixed reference recording.

Let:

- `T` = number of query frames
- `R` = number of reference frames
- `d` = chroma dimension, usually `d = 12`

For each query frame `t = 1, ..., T`:

- hidden state: `x_t ∈ {1, ..., R}`
- observation: `y_t ∈ R^d`

Interpretation:

- `x_t = j` means the query at time `t` is aligned to reference frame `j`
- `y_t` is the query chroma vector at time `t`

The reference is represented by a sequence of chroma vectors:

- `phi_j ∈ R^d` for `j = 1, ..., R`

where `phi_j` is the chroma feature vector of reference frame `j`.

## Data Flow

The intended pipeline is:

1. Choose one reference recording.
2. Extract reference chroma sequence `phi_1, ..., phi_R`.
3. Stream the query audio.
4. For each incoming query frame, extract chroma `y_t`.
5. Run online HMM inference to estimate `x_t`.

This is different from offline DTW:

- offline DTW finds one globally optimal path using the full query
- the HMM performs causal inference using only observations seen so far

## Feature Extraction

Let the query audio signal be `q[n]` and the reference audio signal be `r[n]`.

Using window length `N`, hop size `H`, and analysis window `w[m]`, define the
STFTs:

```math
Q_t[k] = \sum_{m=0}^{N-1} q[tH + m]\, w[m]\, e^{-i 2\pi km/N}
```

```math
R_j[k] = \sum_{m=0}^{N-1} r[jH + m]\, w[m]\, e^{-i 2\pi km/N}
```

Let `M ∈ R^{d × K}` be a mapping from frequency bins to chroma bins. Then the
query and reference chroma vectors are

```math
y_t = \mathrm{norm}(M |Q_t|^p)
```

```math
\phi_j = \mathrm{norm}(M |R_j|^p)
```

where:

- `p` is usually `1` or `2`
- `norm` is usually L1 or L2 normalization

In practice, the implementation already computes chroma-like framewise features,
so this document treats `y_t` and `phi_j` as given.

## HMM Definition

The HMM is defined by:

- state space: `x_t ∈ {1, ..., R}`
- initial distribution: `pi_j = P(x_1 = j)`
- transition matrix: `a_ij = P(x_t = j | x_{t-1} = i)`
- emission density: `p(y_t | x_t = j)`

The joint distribution is

```math
P(x_{1:T}, y_{1:T})
= P(x_1) P(y_1 | x_1) \prod_{t=2}^T P(x_t | x_{t-1}) P(y_t | x_t).
```

## Initial Distribution

If the alignment always starts at the beginning of the reference, use

```math
\pi_1 = 1, \qquad \pi_j = 0 \text{ for } j > 1.
```

If a softer start is desired, define a narrow prior near the start, for example

```math
\pi_j \propto \exp(-\gamma (j-1)^2)
```

for small `j`.

For the first implementation, the hard start prior is simplest.

## Transition Model

The transition model should encode causal forward motion through the reference.

Define

```math
a_{ij} = P(x_t = j \mid x_{t-1} = i).
```

For a monotone, bounded-jump model:

```math
a_{ij} = 0 \quad \text{if } j < i
```

and

```math
a_{ij} = 0 \quad \text{if } j - i > K
```

for some maximum allowed jump `K`.

It is often convenient to parameterize by jump size:

```math
\Delta = j - i
```

with

```math
P(x_t = j \mid x_{t-1} = i) = a_\Delta, \qquad \Delta \in \{0, 1, ..., K\}.
```

subject to

```math
a_\Delta \ge 0, \qquad \sum_{\Delta=0}^K a_\Delta = 1.
```

Interpretation:

- `Delta = 0`: hold at the same reference frame
- `Delta = 1`: advance by one reference frame
- `Delta = 2, 3, ...`: move forward faster

To avoid pathological end behavior, the final state can be made absorbing:

```math
P(x_t = R \mid x_{t-1} = R) = 1.
```

## Gaussian Emission Model

The observation is the query chroma vector `y_t ∈ R^d`.

For each reference frame `j`, define a Gaussian emission density:

```math
p(y_t \mid x_t = j) = \mathcal{N}(y_t; \mu_j, \Sigma_j).
```

The multivariate Gaussian density is

```math
\mathcal{N}(y; \mu, \Sigma)
= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2} (y-\mu)^T \Sigma^{-1} (y-\mu)
\right).
```

A natural first choice is to tie the mean to the reference chroma:

```math
\mu_j = \phi_j.
```

Then the emission becomes

```math
p(y_t \mid x_t = j) = \mathcal{N}(y_t; \phi_j, \Sigma_j).
```

This says:

- if the query chroma at time `t` resembles reference chroma `phi_j`, state `j`
  is likely
- if it is far away in chroma space, state `j` is unlikely

## Covariance Choices

There are several reasonable choices for the covariance.

### Option 1: Shared isotropic covariance

```math
\Sigma_j = \sigma^2 I \quad \text{for all } j
```

Pros:

- simplest
- few parameters
- stable

Cons:

- may be too crude

### Option 2: Shared diagonal covariance

```math
\Sigma_j = \mathrm{diag}(\sigma_1^2, ..., \sigma_d^2)
```

Pros:

- still stable
- lets different chroma bins have different variance

Cons:

- still ignores correlations

### Option 3: State-specific diagonal covariance

```math
\Sigma_j = \mathrm{diag}(\sigma_{j,1}^2, ..., \sigma_{j,d}^2)
```

Pros:

- more flexible

Cons:

- many more parameters
- likely data-hungry

For a first implementation, shared isotropic or shared diagonal covariance is
the safest choice.

## Relation To Local Cost

Even though the official observation in this HMM is `y_t`, it is useful to note
the connection to DTW local costs.

A local chroma distance can be defined as

```math
C(t,j) = d(y_t, \phi_j).
```

For cosine distance:

```math
C(t,j)
= 1 - \frac{y_t^T \phi_j}{\|y_t\|_2 \|\phi_j\|_2 + \varepsilon}.
```

If all chroma vectors are L2-normalized:

```math
C(t,j) = 1 - y_t^T \phi_j.
```

The full local cost row at query time `t` is

```math
C_t = [C(t,1), C(t,2), ..., C(t,R)].
```

This local cost row is not the HMM observation in the present formulation, but
it is a useful diagnostic object because it describes framewise similarity
between the query and the reference.

## Online Filtering

Define the filtered posterior

```math
\alpha_t(j) = P(x_t = j \mid y_{1:t}).
```

Initialization:

```math
\alpha_1(j) \propto \pi_j \, p(y_1 \mid x_1 = j).
```

Prediction step:

```math
\hat{\alpha}_t(j) = \sum_{i=1}^R \alpha_{t-1}(i) a_{ij}.
```

Update step:

```math
\alpha_t(j) \propto p(y_t \mid x_t = j)\, \hat{\alpha}_t(j).
```

Normalize so that

```math
\sum_{j=1}^R \alpha_t(j) = 1.
```

This yields a causal estimate of the current reference position.

Useful online outputs:

- MAP state:

```math
\hat{x}_t^{\mathrm{MAP}} = \arg\max_j \alpha_t(j)
```

- posterior mean position:

```math
\hat{x}_t^{\mathrm{mean}} = \sum_{j=1}^R j \, \alpha_t(j)
```

The MAP estimate is simpler. The posterior mean can be smoother.

## Viterbi Decoding

If the goal is to recover the most likely entire path rather than the filtered
state posterior, use Viterbi.

Define

```math
\delta_t(j) = \log \max_{x_{1:t-1}} P(x_{1:t-1}, x_t = j, y_{1:t}).
```

Initialization:

```math
\delta_1(j) = \log \pi_j + \log p(y_1 \mid x_1 = j).
```

Recursion:

```math
\delta_t(j)
= \log p(y_t \mid x_t = j)
+ \max_i \left[ \delta_{t-1}(i) + \log a_{ij} \right].
```

Backpointer:

```math
\psi_t(j)
= \arg\max_i \left[ \delta_{t-1}(i) + \log a_{ij} \right].
```

Backtracking from the best final state gives the most likely state sequence.

For strictly online accompaniment, filtering is more natural. For analysis and
evaluation, Viterbi is also useful.

## Supervised Parameter Estimation From Pseudo-Labels

The easiest training strategy is supervised estimation using pseudo-ground-truth
alignments from offline DTW.

### Step 1: Generate pseudo-labels

For a chosen reference recording and a training query recording of the same
piece:

1. extract chroma features for both recordings
2. run offline DTW
3. obtain an alignment path
4. convert that path into frame labels `x_t^*`

Here `x_t^*` is the aligned reference frame index for query frame `t`.

### Step 2: Estimate transitions

From pseudo-labels, estimate transition counts:

```math
N_{ij} = \#\{ t : x_{t-1}^* = i, \; x_t^* = j \}.
```

Then estimate

```math
\hat{a}_{ij}
= \frac{N_{ij} + \lambda}{\sum_{k=1}^R (N_{ik} + \lambda)}
```

with smoothing parameter `lambda > 0`.

If using a jump-based model, estimate counts by jump size:

```math
N_\Delta = \#\{ t : x_t^* - x_{t-1}^* = \Delta \}
```

and then

```math
\hat{a}_\Delta
= \frac{N_\Delta + \lambda}{\sum_{d=0}^K (N_d + \lambda)}.
```

### Step 3: Estimate Gaussian means

For each state `j`, collect all query feature vectors aligned to that reference
frame:

```math
Y_j = \{ y_t : x_t^* = j \}.
```

The empirical mean is

```math
\hat{\mu}_j = \frac{1}{|Y_j|} \sum_{y \in Y_j} y.
```

A simpler tied-mean model uses

```math
\mu_j = \phi_j.
```

This is attractive because it directly uses the reference template and avoids
estimating a separate mean for every state.

### Step 4: Estimate covariance

If using shared covariance:

```math
\hat{\Sigma}
= \frac{1}{N}
\sum_{j=1}^R \sum_{y \in Y_j}
(y - \hat{\mu}_j)(y - \hat{\mu}_j)^T
```

where `N` is the total number of aligned training frames.

If using diagonal covariance, keep only the diagonal entries.

If using isotropic covariance:

```math
\hat{\Sigma} = \hat{\sigma}^2 I
```

with

```math
\hat{\sigma}^2
= \frac{1}{Nd}
\sum_{j=1}^R \sum_{y \in Y_j} \| y - \hat{\mu}_j \|_2^2.
```

## Recommended First Training Recipe

A practical first training setup is:

- fixed reference recording
- states = reference frames
- pseudo-labels from offline DTW
- transition model estimated by bounded forward jumps
- emission means set to `mu_j = phi_j`
- shared diagonal covariance estimated from aligned residuals

This is much easier to fit than a fully state-specific Gaussian model.

## What The Existing Data Looks Like

The raw data layout under `data/raw/` contains:

- audio recordings grouped by piece in `data/raw/wav_22050_mono/<piece>/`
- beat annotations grouped by piece in `data/raw/annotations_beat/<piece>/`

At the time of writing, the dataset contains five pieces:

- `Chopin_Op017No4`
- `Chopin_Op024No2`
- `Chopin_Op030No2`
- `Chopin_Op063No3`
- `Chopin_Op068No3`

Each piece has many performances with beat annotations. Approximate recording
counts observed in the repository:

- `Chopin_Op017No4`: 63
- `Chopin_Op024No2`: 64
- `Chopin_Op030No2`: 34
- `Chopin_Op063No3`: 88
- `Chopin_Op068No3`: 50

This structure is useful because:

- there are many query/reference pairings available per piece
- beat annotations provide extra supervision for evaluation
- the same piece can be aligned under many different expressive renderings

## How The Data Can Be Used To Train The Reference-Specific HMM

For one experiment:

1. Choose a piece.
2. Choose one recording of that piece as the reference.
3. Use other recordings of the same piece as training queries.
4. Run offline DTW for each query/reference pair.
5. Convert each offline alignment into pseudo-labels `x_t^*`.
6. Pool the aligned query frames to estimate transitions and emission
   parameters.

This trains one HMM for one chosen reference recording.

That is still fully compatible with the project goal of real-time alignment,
because the online inference problem remains:

- align a live query to a fixed reference in real time

The offline DTW and labels are only used during training and analysis.

## How Beat Labels Can Still Help

Even though the HMM is reference-specific, the beat files are still useful.

They can be used to:

- evaluate alignment accuracy at musically meaningful landmarks
- filter out bad pseudo-labels
- compare the HMM alignment path to beat-to-beat ground truth
- analyze early/late drift and failure modes

They could also be used later to improve pseudo-label quality within local
regions, but they are not required for the first HMM formulation.

## Important Modeling Notes

### Query Longer Than Reference

This model still works when the query has more frames than the reference.

The reason is that:

- `T` and `R` do not need to be equal
- self-transitions allow many query frames to align to the same reference frame

If the query continues after the reference effectively ends, the final reference
state can be made absorbing.

### Why Use Query Chroma As The Observation

Using `y_t` as the observation is cleaner than using an entire cost row because:

- the observation dimension is fixed (`d = 12` for chroma)
- the Gaussian emission model is natural in this space
- the model remains comparable across different reference lengths
- the cost row can still be derived as a diagnostic object

### Why This Is Still Dynamic Programming

The HMM does not remove dynamic programming. It replaces DTW-style path search
with HMM inference:

- forward recursion for online filtering
- Viterbi recursion for best-path decoding

So the project still uses DP-based sequential inference, but in a probabilistic
form rather than a pure cumulative-cost form.

## Implementation Notes

For numerical stability, all inference should be implemented in log space or
with explicit normalization at every step.

In practice:

- use `log p(y_t | x_t = j)` rather than raw densities in Viterbi
- normalize filtering distributions every frame
- constrain transitions to a small forward band for speed
- start with shared covariance before attempting state-specific covariance

## Suggested First Ablation Questions

Once the HMM is implemented, the first questions worth testing are:

1. Does a reference-specific HMM with Gaussian chroma emissions beat naive
   online DTW on the development benchmark?
2. Does using `mu_j = phi_j` work better than estimating `mu_j` from
   pseudo-labels?
3. Does shared isotropic covariance underfit compared to shared diagonal
   covariance?
4. How sensitive is performance to the maximum jump size `K`?
5. Is online filtering or online Viterbi a better output for accompaniment?

## Summary

The proposed reference-specific HMM is:

- hidden state: reference frame index `x_t`
- observation: query chroma vector `y_t`
- transitions: bounded forward Markov chain
- emissions: multivariate Gaussian
- training: supervised from offline-DTW pseudo-labels
- evaluation: beat-aligned accuracy using the existing Mazurka annotations

This gives a mathematically clean HMM baseline that remains aligned with the
original project goal: real-time alignment of a live query to a fixed reference
recording.
