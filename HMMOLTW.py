import numpy as np
from numpy.linalg import inv, slogdet
from scipy.special import logsumexp
from dataclasses import dataclass


def ComputeLogGaussianEmission(
    Observation: np.ndarray,
    Mean: np.ndarray,
    CovarianceInverse: np.ndarray,
    LogCovarianceDeterminant: float
) -> float:
    """
    Evaluates log P(Observation | state) for a multivariate Gaussian.

    Uses precomputed covariance inverse and log-determinant to avoid
    repeated matrix operations during online inference.

    Parameters:
        - Observation:               (FeatureDim,) query chroma vector
        - Mean:                      (FeatureDim,) per-state emission mean
        - CovarianceInverse:         (FeatureDim, FeatureDim) precomputed inverse covariance
        - LogCovarianceDeterminant:  log |Covariance| for this state
    Returns:
        - Log-likelihood scalar
    """
    FeatureDim = len(Mean)
    Diff = Observation - Mean

    # This computes the distance between the observation and the mean of the distribution.
    # We do this to help avoid having to build and sample a multivariate Gaussian pdf with scipy.
    # Note that this is the term multiplied by -1/2 in the exponent of the multivariate Gaussian.
    MahalanobisSquared = float(Diff @ CovarianceInverse @ Diff)

    # This is just the log probability of the multivariate Gaussian pdf
    return -0.5 * (FeatureDim * np.log(2 * np.pi) + LogCovarianceDeterminant + MahalanobisSquared)


@dataclass
class HMMParameters:
    """
    Preprocessed HMM parameters for efficient online inference.

    Stores the transition matrix in log-space and precomputes per-state
    covariance inverses and log-determinants to avoid repeated matrix
    inversions as StreamingDTW runs.

    Fields:
        - LogTransitionMatrix:        (StateCount, StateCount) log A[i,j] = log P(j | i)
        - Means:                      list of (FeatureDim,) per-state emission mean vectors
        - CovarianceInverses:         list of (FeatureDim, FeatureDim) precomputed inverses
        - LogCovarianceDeterminants:  (StateCount,) log |Covariance_i| per state
        - StateCount:                 number of HMM states (= number of reference frames)
    """
    LogTransitionMatrix: np.ndarray
    Means: list
    CovarianceInverses: list
    LogCovarianceDeterminants: np.ndarray
    StateCount: int


@dataclass
class ViterbiRunningState:
    """
    Running state for the online Viterbi decoder.

    NormalizedLogProbabilities[i] holds an approximation of log P(state = i | obs_0..t),
    computed with the Viterbi algorithm, and normalized at each step via
    log-sum-exp to prevent underflow over long sequences (Claude rec).
    """
    NormalizedLogProbabilities: np.ndarray  # (StateCount,)


@dataclass
class StreamingDTWRunningState:
    """
    Running state for the online time-warping aligner.

    Fields:
        - PreviousRowCosts:         (RefFrameCount,) accumulated DTW costs across all reference
                                    frames for the most recently processed query frame (row Q - 1)
        - CurrentReferenceEstimate: index of the reference frame with minimum path-length-normalized
                                    accumulated cost at the current query position
        - CurrentQueryFrameIndex:   how many query frames have been processed - 1
    """
    PreviousRowCosts: np.ndarray
    CurrentReferenceEstimate: int
    CurrentQueryFrameIndex: int


def PreprocessHMMParameters(
    TransitionMatrix: np.ndarray,
    Means: list,
    Covars: list,
    CovarianceRegularization: float = 1e-6
) -> HMMParameters:
    """
    Converts raw HMM parameters into a form ready for efficient online inference.

    Takes the outputs of IterativeTrainHMM.Exec_IterativeTrainHMM (A, Means, Covars, no Pi)
    and produces log-space transitions, precomputed covariance inverses, and
    log-determinants so that per-frame emission computation has no matrix operations.

    Parameters:
        - TransitionMatrix:          (StateCount, StateCount) row-stochastic, A[i,j] = P(j | i)
        - Means:                     list of (FeatureDim,) per-state emission mean vectors
        - Covars:                    list of (FeatureDim, FeatureDim) per-state covariance matrices
        - CovarianceRegularization:  small diagonal term added to each covariance to prevent
                                     singularity for states with few training observations (Claude rec)
    Returns:
        - HMMParameters ready for streaming inference
    """
    StateCount = len(Means)

    # Set negative- and zero-probability entries of the state transition matrix to
    # a small number to avoid bad logs
    SafeTransitionMatrix = np.where(TransitionMatrix > 0, TransitionMatrix, 1e-300)
    LogTransitionMatrix = np.log(SafeTransitionMatrix)

    CovarianceInverses = []
    LogCovarianceDeterminants = []

    for StateIndex in range(StateCount):
        Covariance = Covars[StateIndex]
        FeatureDim = Covariance.shape[0]
        RegularizedCovariance = Covariance + CovarianceRegularization * np.eye(FeatureDim)

        CovarianceInverses.append(inv(RegularizedCovariance))
        _, LogDet = slogdet(RegularizedCovariance)
        LogCovarianceDeterminants.append(LogDet)

    return HMMParameters(
        LogTransitionMatrix = LogTransitionMatrix,
        Means = Means,
        CovarianceInverses = CovarianceInverses,
        LogCovarianceDeterminants = np.array(LogCovarianceDeterminants),
        StateCount = StateCount
    )


def InitializeViterbi(InitialDistribution: np.ndarray) -> ViterbiRunningState:
    """
    Creates the initial Viterbi state from the HMM initial distribution Pi.

    Parameters:
        - InitialDistribution: (StateCount,) or (1, StateCount) initial state distribution Pi
    Returns:
        - ViterbiRunningState with log-probabilities initialized to log(Pi), normalized
    """
    PiFlat = np.array(InitialDistribution).flatten().astype(float)

    # Replace 0 or negative entries with a value that's effectively zero to avoid 
    # log of a bad value.
    SafePi = np.where(PiFlat > 0, PiFlat, 1e-300)

    LogPi = np.log(SafePi)

    # This isn't needed in the case that the initial distribution sums to 1, 
    # but we offset zero entries by a very small value. We need valid log probabilities,
    # but they're currently offset. So, we subtract off the log of the sum of the distribution,
    # which, outside of log-world, is dividing by the sum of the initial distribution, returning
    # it to a valid distribution with a sum of 1.
    NormalizedLogPi = LogPi - logsumexp(LogPi)

    return ViterbiRunningState(NormalizedLogProbabilities = NormalizedLogPi)


def StepViterbi(
    RunningState: ViterbiRunningState,
    HMM: HMMParameters,
    NewObservation: np.ndarray
) -> ViterbiRunningState:
    """
    Advances the Viterbi run by one observation frame.

    For each destination state j, computes:
        log_prob[j] = log P(obs | j) + max_i( log_prob[i] + log A[i,j] )
    then normalizes via log-sum-exp to prevent underflow across long sequences.
    The normalized result approximates log P(state = j | observations 0..t).

    Parameters:
        - RunningState:  current Viterbi state (just the log probabilites of each current referece frame)
        - HMM:           preprocessed HMM parameters
        - NewObservation: (FeatureDim,) current query chroma vector
    Returns:
        - Updated ViterbiRunningState
    """

    # Determine the log probability of the current observation for each state
    LogEmissions = np.array([
        ComputeLogGaussianEmission(
            NewObservation,
            HMM.Means[StateIndex],
            HMM.CovarianceInverses[StateIndex],
            HMM.LogCovarianceDeterminants[StateIndex]
        )
        for StateIndex in range(HMM.StateCount)
    ])

    # Shape (StateCount, StateCount): row i = log_prob[i] + log A[i, j] for all j
    # In words: adds the log probability of each state (given last observation) 
    # from Viterbi to the log probability of transitioning from that state to any other state.
    LogJointFromAllPredecessors = RunningState.NormalizedLogProbabilities[:, np.newaxis] + HMM.LogTransitionMatrix

    # If we're going to state j (that's columns), what's the highest probability of the last state?
    BestPredecessorLogProb = LogJointFromAllPredecessors.max(axis = 0)  # (StateCount,)

    # Add the probability of going to state j and the probability that the current observation
    # corresponds to state j
    RawLogProbabilities = LogEmissions + BestPredecessorLogProb

    # See InitializeViterbi for what we're doing here
    NormalizedLogProbabilities = RawLogProbabilities - logsumexp(RawLogProbabilities)

    return ViterbiRunningState(NormalizedLogProbabilities = NormalizedLogProbabilities)


def InitializeStreamingDTW(RefFrameCount: int) -> StreamingDTWRunningState:
    """
    Creates the initial StreamingDTW state, setting the alignment start at reference frame 0.

    Parameters:
        - RefFrameCount: total number of reference frames
    Returns:
        - StreamingDTWRunningState ready to receive the first query frame
    """

    PreviousRowCosts = np.full(RefFrameCount, np.inf)
    PreviousRowCosts[0] = 0.0

    return StreamingDTWRunningState(
        PreviousRowCosts = PreviousRowCosts,
        CurrentReferenceEstimate = 0,
        CurrentQueryFrameIndex = 0
    )


def ComputeChromaCosineDistance(FrameA: np.ndarray, FrameB: np.ndarray) -> float:
    """Determines the cosine distance between two chroma feature vectors (in [0, 1])."""

    NormA = np.linalg.norm(FrameA)
    NormB = np.linalg.norm(FrameB)

    # If either feature vector has length 0, assume that the frames are maximally distant
    if NormA == 0.0 or NormB == 0.0:
        return 1.0
    
    CosineSimilarity = np.dot(FrameA, FrameB) / (NormA * NormB)

    # Ensure that the result of the cosine similarity is in the valid of range of cosine, just in case
    return float(1.0 - np.clip(CosineSimilarity, -1.0, 1.0))


def StepStreamingDTW(
    RunningState: StreamingDTWRunningState,
    ReferenceChroma: np.ndarray,
    NewQueryFrame: np.ndarray,
    ViterbiNormalizedLogProbs: np.ndarray,
    SearchHalfWidth: int,
    HMMWeight: float
) -> StreamingDTWRunningState:
    """
    Processes one new query frame in the online DTW aligner, weighting cell costs
    by the current Viterbi state distribution.

    For each reference frame r in the active window the adjusted cell cost is:
        AdjustedCost(r) = ChromaCosineDistance(ref[r], query) - HMMWeight * ViterbiLogProb[r]
    so reference frames the HMM considers likely incur a lower alignment cost,
    steering the warping path toward HMM-consistent positions.

    The DTW then selects the cheapest predecessor between:
        diagonal  (t-1, r-1) : both query and reference advance
        above     (t-1, r  ) : query advances, reference stays
        left      (t,   r-1) : reference advances, query stays (built left-to-right)

    The search window is centered on the HMM's best state estimate (argmax of ViterbiNormalizedLogProbs)
    rather than the previous DTW estimate.

    Accumulated costs are normalized by path length before selecting a NewReferenceEstimate,
    so that longer paths are not unfairly penalized relative to shorter ones.

    Parameters:
        - RunningState:               current StreamingDTW state
        - ReferenceChroma:            (FeatureDim, RefFrameCount) reference feature matrix
        - NewQueryFrame:              (FeatureDim,) current query chroma vector
        - ViterbiNormalizedLogProbs:  (StateCount,) normalized log-probs from the current Viterbi step
        - SearchHalfWidth:            half-width of the active window around the HMM MAP state
        - HMMWeight:                  scaling factor for the Viterbi cost bias (0 = pure DTW)
    Returns:
        - Updated StreamingDTWRunningState with new row costs, path lengths, and reference frame estimate
    """
    ReferenceFrameCount = ReferenceChroma.shape[1]

    # Center the search window on the HMM estimated state. We assume Viterbi is a semi-accurate, 
    # so let's search around its estimate.
    HMMMapState = int(np.argmax(ViterbiNormalizedLogProbs))
    WindowStart = max(0, HMMMapState - SearchHalfWidth)
    WindowEnd = min(ReferenceFrameCount, HMMMapState + SearchHalfWidth + 1)

    CurrentRowCosts = np.full(ReferenceFrameCount, np.inf)
    CurrentPathLengths = np.full(ReferenceFrameCount, np.inf)

    for ReferenceFrame in range(WindowStart, WindowEnd):
        LocalCost = ComputeChromaCosineDistance(ReferenceChroma[:, ReferenceFrame], NewQueryFrame)

        # -ViterbiNormalizedLogProbs[RefFrame] >= 0 always, so likely states (log-prob near 0)
        # incur little extra cost while unlikely states are penalized.
        HMMAdjustedLocalCost = LocalCost + HMMWeight * (-ViterbiNormalizedLogProbs[ReferenceFrame])

        DiagonalPredecessorCost = RunningState.PreviousRowCosts[ReferenceFrame - 1] if ReferenceFrame > 0 else np.inf
        VerticalPredecessorCost = RunningState.PreviousRowCosts[ReferenceFrame]
        HorizontalPredecessorCost = CurrentRowCosts[ReferenceFrame - 1] if ReferenceFrame > 0 else np.inf

        PredecessorCosts = [DiagonalPredecessorCost, VerticalPredecessorCost, HorizontalPredecessorCost]
        BestPredecessorIndex = int(np.argmin(PredecessorCosts))

        CurrentRowCosts[ReferenceFrame] = HMMAdjustedLocalCost + PredecessorCosts[BestPredecessorIndex]

    WindowCosts = CurrentRowCosts[WindowStart:WindowEnd]
    WindowPathLengths = CurrentPathLengths[WindowStart:WindowEnd]

    # Normalize by path length so longer accumulated-cost paths aren't unfairly penalized
    # relative to shorter ones when picking the best reference position estimate.
    # Ensure that we don't divide by a negative number or 0 (just in case).
    NormalizedWindowCosts = WindowCosts / np.maximum(WindowPathLengths, 1.0)
    NewReferenceEstimate = WindowStart + int(np.argmin(NormalizedWindowCosts))

    return StreamingDTWRunningState(
        PreviousRowCosts = CurrentRowCosts,
        CurrentReferenceEstimate = NewReferenceEstimate,
        CurrentQueryFrameIndex = RunningState.CurrentQueryFrameIndex + 1
    )

def ProcessNextFrame(
    ViterbiState: ViterbiRunningState,
    StreamingDTWState: StreamingDTWRunningState,
    HMM: HMMParameters,
    ReferenceChroma: np.ndarray,
    NewQueryChroma: np.ndarray,
    SearchHalfWidth: int,
    HMMWeight: float
) -> tuple[int, ViterbiRunningState, StreamingDTWRunningState]:
    """
    Processes one incoming query frame through the combined HMM-StreamingDTW aligner.

    Viterbi is updated before StreamingDTW so the HMM has already incorporated the new
    observation when its state distribution is used to bias the DTW costs.

    Parameters:
        - ViterbiState:    current running Viterbi state
        - StreamingDTWState:       current running StreamingDTW state
        - HMM:             preprocessed HMM parameters
        - ReferenceChroma: (FeatureDim, RefFrameCount) reference feature matrix
        - NewQueryChroma:   (FeatureDim,) current query chroma vector
        - SearchHalfWidth: half-width of the StreamingDTW active window in frames
        - HMMWeight:       scaling factor for the Viterbi cost bias
    Returns:
        - ReferenceFrameEstimate: best estimate of the current reference position
        - UpdatedViterbiState
        - UpdatedStreamingDTWState
    """
    UpdatedViterbiState = StepViterbi(ViterbiState, HMM, NewQueryChroma)
    UpdatedStreamingDTWState = StepStreamingDTW(
        StreamingDTWState,
        ReferenceChroma,
        NewQueryChroma,
        UpdatedViterbiState.NormalizedLogProbabilities,
        SearchHalfWidth,
        HMMWeight
    )
    return UpdatedStreamingDTWState.CurrentReferenceEstimate, UpdatedViterbiState, UpdatedStreamingDTWState


def InitializeHMMStreamingDTW(
    ReferenceChroma: np.ndarray,
    TransitionMatrix: np.ndarray,
    InitialDistribution: np.ndarray,
    Means: list,
    Covars: list,
    SearchHalfWidth: int = 50,
    HMMWeight: float = 0.1
) -> tuple[HMMParameters, ViterbiRunningState, StreamingDTWRunningState]:
    """
    Initializes all state required to run the online HMM-StreamingDTW aligner.

    Accepts the outputs of IterativeTrainHMM.Exec_IterativeTrainHMM directly.
    The returned objects are passed to ProcessNextFrame on each incoming query frame.

    Parameters:
        - ReferenceChroma:      (FeatureDim, RefFrameCount) reference chroma features
        - TransitionMatrix:     (StateCount, StateCount) HMM transition matrix A
        - InitialDistribution:  (StateCount,) or (1, StateCount) initial distribution Pi
        - Means:                list of (FeatureDim,) per-state emission mean vectors
        - Covars:               list of (FeatureDim, FeatureDim) per-state covariance matrices
        - SearchHalfWidth:      reference frames on each side of the current estimate to
                                consider in StreamingDTW (default: 50 frames is about 1.2 s at 512-sample hop)
        - HMMWeight:            how strongly Viterbi log-probabilities bias StreamingDTW costs;
                                0 = pure StreamingDTW, larger values increase HMM influence (default: 0.1)
    Returns:
        - HMM:          preprocessed HMM parameters
        - ViterbiState: initial Viterbi state
        - StreamingDTWState:    initial StreamingDTW state
    """
    RefFrameCount = ReferenceChroma.shape[1]
    StateCount = len(Means)

    HMMStateReferenceMessage = f"HMM state count ({StateCount}) must equal reference frame count ({RefFrameCount})"

    assert (StateCount == RefFrameCount), HMMStateReferenceMessage

    HMM = PreprocessHMMParameters(TransitionMatrix, Means, Covars)
    ViterbiState = InitializeViterbi(InitialDistribution)
    StreamingDTWState = InitializeStreamingDTW(RefFrameCount)

    return HMM, ViterbiState, StreamingDTWState
