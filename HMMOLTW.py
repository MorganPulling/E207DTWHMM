import numpy as np
import Constants
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


def LogPositiveOrNegativeInfinity(Values: np.ndarray) -> np.ndarray:
    """
    Computes log(x) for positive entries and -inf for zero or negative entries.

    np.where(condition, np.log(values), -np.inf) still evaluates np.log(values)
    for every entry, so zeros produce RuntimeWarnings even though they are later
    replaced. The masked ufunc form avoids evaluating log at impossible entries.
    """
    Values = np.asarray(Values, dtype = float)
    LogValues = np.full(Values.shape, -np.inf, dtype = float)
    np.log(Values, out = LogValues, where = Values > 0)
    return LogValues


@dataclass
class HMMParameters:
    """
    Preprocessed HMM parameters for efficient online inference.

    Stores the transition matrix in log-space and precomputes per-state
    covariance inverses and log-determinants to avoid repeated matrix
    inversions as StreamingDTW runs.

    Fields:
        - LogTransitionMatrix:        (StateCount, StateCount) log A[i,j] = log P(j | i)
        - Means:                      (StateCount, FeatureDim) per-state emission mean vectors
        - CovarianceInverses:         (StateCount, FeatureDim, FeatureDim) precomputed inverses
        - LogCovarianceDeterminants:  (StateCount,) log |Covariance_i| per state
        - LogEmissionConstants:       (StateCount,) constant part of each Gaussian log emission
        - TransitionDestinations:     per-state arrays of reachable transition destinations
        - StateCount:                 number of HMM states (= number of reference frames)
    """
    LogTransitionMatrix: np.ndarray
    Means: np.ndarray
    CovarianceInverses: np.ndarray
    LogCovarianceDeterminants: np.ndarray
    LogEmissionConstants: np.ndarray
    TransitionDestinations: list
    StateCount: int


def NormalizeLogProbabilities(LogProbabilities: np.ndarray) -> np.ndarray:
    Normalizer = logsumexp(LogProbabilities)
    if not np.isfinite(Normalizer):
        return LogProbabilities.copy()
    return LogProbabilities - Normalizer


@dataclass
class ViterbiRunningState:
    """
    Running state for the online Viterbi decoder.

    Fields:
        - NormalizedLogProbabilities[i] holds an approximation of log P(state = i | obs_0..t),
          computed with the Viterbi algorithm, and normalized at each step via
          log-sum-exp to prevent underflow over long sequences (Claude rec).
        - CurrentReferenceEstimate: index of the reference frame with maximum
                                    accumulated probability at the current query position
    """
    NormalizedLogProbabilities: np.ndarray  # (StateCount,)
    CurrentReferenceEstimate: int


@dataclass
class StreamingDTWRunningState:
    """
    Running state for the online time-warping aligner.

    Fields:
        - PreviousRowCosts:         (RefFrameCount,) accumulated DTW costs across all reference
                                    frames for the most recently processed query frame (row Q - 1)
        - CurrentReferenceEstimate: index of the reference frame with minimum
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
    Means = np.asarray(Means, dtype = float)

    # Ensure that transitions with negative (impossible) probabilities or zero probabilities are never reached
    LogTransitionMatrix = LogPositiveOrNegativeInfinity(TransitionMatrix)
    TransitionDestinations = [
        np.flatnonzero(TransitionMatrix[StateIndex] > 0)
        for StateIndex in range(StateCount)
    ]

    CovarianceInverses = []
    LogCovarianceDeterminants = []
    LogEmissionConstants = []

    for StateIndex in range(StateCount):
        Covariance = Covars[StateIndex]
        FeatureDim = Covariance.shape[0]
        RegularizedCovariance = Covariance + CovarianceRegularization * np.eye(FeatureDim)

        CovarianceInverses.append(inv(RegularizedCovariance))
        _, LogDet = slogdet(RegularizedCovariance)
        LogCovarianceDeterminants.append(LogDet)
        LogEmissionConstants.append(-0.5 * (FeatureDim * np.log(2 * np.pi) + LogDet))

    return HMMParameters(
        LogTransitionMatrix = LogTransitionMatrix,
        Means = Means,
        CovarianceInverses = np.array(CovarianceInverses),
        LogCovarianceDeterminants = np.array(LogCovarianceDeterminants),
        LogEmissionConstants = np.array(LogEmissionConstants),
        TransitionDestinations = TransitionDestinations,
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

    LogPi = LogPositiveOrNegativeInfinity(PiFlat)

    # This is unnecessary when the initial distribution already sums to 1, but
    # it also handles non-normalized positive entries. Zero entries stay at -inf.
    NormalizedLogPi = NormalizeLogProbabilities(LogPi)

    return ViterbiRunningState(NormalizedLogProbabilities = NormalizedLogPi, CurrentReferenceEstimate = 0)

def MakeWindowCenter(
    QueryIndex: int,
    QueryFrameCount: int,
    ReferenceFrameCount: int,
) -> int:
    """
    Determines a global window center point for Viterbi windowing.
    """
    if QueryFrameCount <= 1:
        return 0

    QueryProgress = QueryIndex / (QueryFrameCount - 1)

    # Simply, how far are we through the query? Estimate that we've made the same progress through the reference.
    return int(round(QueryProgress * (ReferenceFrameCount - 1)))

def StepViterbi(
    RunningState: ViterbiRunningState,
    HMM: HMMParameters,
    NewObservation: np.ndarray,
    WindowHalfWidth: int
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
        - WindowHalfWdith: THe number of frames around the current estimate to search for the next state.
    Returns:
        - Updated ViterbiRunningState
    """
    ReferenceFrameCount = HMM.StateCount

    # For performance reasons, let's only consider a range around the current reference estimate.
    WindowCenter = RunningState.CurrentReferenceEstimate
    WindowCenter = int(np.clip(WindowCenter, 0, ReferenceFrameCount - 1))

    # Only states in the window are currently possible. Let's only look forward from the current estimate.
    WindowStart = max(0, WindowCenter)
    WindowEnd = min(ReferenceFrameCount, WindowCenter + WindowHalfWidth + 1)
    CurrentPossibleStates = np.arange(WindowStart, WindowEnd)

    # flatnonzero returns indices where the argument, flattened, has nonzero values. So, with this line,
    # we find indices where the NormalizedLogProbabilities are not +-infty (i.e., where transitions can be taken)
    PossiblePreviousStates = np.flatnonzero(np.isfinite(RunningState.NormalizedLogProbabilities))

    # Determine the log probability of the current observation for each current possible state
    LogEmissions = np.array([
        ComputeLogGaussianEmission(
            NewObservation,
            HMM.Means[State],
            HMM.CovarianceInverses[State],
            HMM.LogCovarianceDeterminants[State]
        )
        for State in CurrentPossibleStates
    ])

    # Find the log probabilities of all states we could have come from
    PreviousLogProbabilities = RunningState.NormalizedLogProbabilities[PossiblePreviousStates]

    # Build a reduced transition matrix so that we only take possible transitions. np.ix_ creates 
    # an array of indices at which we find the values of the LogTransitionMatrix such that each PossiblePreviousState 
    # is considered for each CurrentPossibleState.
    PossibleTransitionMatrix = HMM.LogTransitionMatrix[np.ix_(PossiblePreviousStates, CurrentPossibleStates)]

    # Take the previous log probabilities and turn them into a column vector. Then, add that vector to each column of the 
    # state transition matrix and take the max. This finds the best probability after combining the possible previous log
    # probabilities with corresponding possible transtions.
    BestPredecessorLogProbabilities = (PreviousLogProbabilities[:, np.newaxis] + PossibleTransitionMatrix).max(axis = 0)

    # We didn't consider emission probabilities before because they only depend on the state, so they're a constant factor 
    # inside the max(). Let's account for them now to consider them in cumulative probability. 
    RawCurrentLogProbabilities = Constants.LAMBDA * LogEmissions + BestPredecessorLogProbabilities

    # Normalize to create a valid PDF (logsumexp correctly ignores -inf entries). First, initialize the normalized probabilities
    # to -infty so that states we couldn't reach are still unreachable.
    NormalizedLogProbabilities = np.full(ReferenceFrameCount, -np.inf)

    # See InitializeViterbi for what we're doing here
    NormalizedLogProbabilities[CurrentPossibleStates] = NormalizeLogProbabilities(RawCurrentLogProbabilities)

    # Our best-estimate state is the one with the highest probability
    NewReferenceEstimate = int(np.argmax(NormalizedLogProbabilities))

    return ViterbiRunningState(NormalizedLogProbabilities = NormalizedLogProbabilities, CurrentReferenceEstimate = NewReferenceEstimate)


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
    SearchHalfWidth: int,
) -> StreamingDTWRunningState:
    """
    Processes one new query frame in the online DTW aligner, weighting cell costs
    by the current Viterbi state distribution.

    For each reference frame r in the active window the adjusted cell cost is:
        AdjustedCost(r) = ChromaDistance(ref[r], query) - HMMWeight * ViterbiLogProb[r]
    so reference frames the HMM considers likely incur a lower alignment cost,
    steering the warping path toward HMM-consistent positions.

    The DTW recurrence then selects the cheapest predecessor among:
        diagonal  (t-1, r-1) : both query and reference advance
        above     (t-1, r  ) : query advances, reference stays
        left      (t,   r-1) : reference advances, query stays (built left-to-right)

    Parameters:
        - RunningState:               current StreamingDTW state
        - ReferenceChroma:            (FeatureDim, RefFrameCount) reference feature matrix
        - NewQueryFrame:              (FeatureDim,) current query chroma vector
        - SearchHalfWidth:            half-width of the active window around CurrentReferenceEstimate
    Returns:
        - Updated StreamingDTWRunningState with new row costs and reference frame estimate
    """
    RefFrameCount = ReferenceChroma.shape[1]

    # This imposes a global constraint (like Sakoe-Chiba) so we don't use as much memory when running StreamingDTW
    WindowStart = max(0, RunningState.CurrentReferenceEstimate - SearchHalfWidth)
    WindowEnd = min(RefFrameCount, RunningState.CurrentReferenceEstimate + SearchHalfWidth + 1)

    # Initialize current query row at infinite cost (we want un-reached entries to never be hit). Let's find a way
    # to avoid initializing an array that's the full width of the reference recording.
    CurrentRowCosts = np.full(RefFrameCount, np.inf)

    for RefFrame in range(WindowStart, WindowEnd):
        LocalCost = ComputeChromaCosineDistance(ReferenceChroma[:, RefFrame], NewQueryFrame)

        # What was the cost of the last row at the previous reference frame?
        DiagonalPredecessorCost = (RunningState.PreviousRowCosts[RefFrame - 1] if RefFrame > 0 else np.inf)

        # What was the cost of the last row at the current reference frame?
        VerticalPredecessorCost = RunningState.PreviousRowCosts[RefFrame]

        # What is the cost of the current row at the previous reference frame?
        HorizontalPredecessorCost = (CurrentRowCosts[RefFrame - 1] if RefFrame > 0 else np.inf)

        # Again, this is not exactly what we discussed. What's happening here is that HMM is modifying the cosine distance, so
        # we're using a method that statically weights Viterbi and StreamingDTW against each other. We'd like to weight transitions based
        # on how well they achieve the state outlined by the HMM (this has its own faults, however).
        CurrentRowCosts[RefFrame] = LocalCost + min(
            DiagonalPredecessorCost,
            VerticalPredecessorCost,
            HorizontalPredecessorCost
        )

    # Grab only the costs within the window
    WindowCosts = CurrentRowCosts[WindowStart:WindowEnd]

    # Create a new reference estimate based on the lowest-cost frame
    NewReferenceEstimate = WindowStart + int(np.argmin(WindowCosts))

    # Update the StreamingDTW state
    return StreamingDTWRunningState(
        PreviousRowCosts = CurrentRowCosts,
        CurrentReferenceEstimate = NewReferenceEstimate,
        CurrentQueryFrameIndex =  RunningState.CurrentQueryFrameIndex + 1
    )

def StepHMMInfluencedDTW(
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
        AdjustedCost(r) = ChromaDistance(ref[r], query) - HMMWeight * ViterbiLogProb[r]
    so reference frames the HMM considers likely incur a lower alignment cost,
    steering the warping path toward HMM-consistent positions.

    The DTW recurrence then selects the cheapest predecessor among:
        diagonal  (t-1, r-1) : both query and reference advance
        above     (t-1, r  ) : query advances, reference stays
        left      (t,   r-1) : reference advances, query stays (built left-to-right)

    Parameters:
        - RunningState:               current StreamingDTW state
        - ReferenceChroma:            (FeatureDim, RefFrameCount) reference feature matrix
        - NewQueryFrame:              (FeatureDim,) current query chroma vector
        - ViterbiNormalizedLogProbs:  (StateCount,) normalized log-probs from the current Viterbi step
        - SearchHalfWidth:            half-width of the active window around CurrentReferenceEstimate
        - HMMWeight:                  scaling factor for the Viterbi cost bias (0 = pure DTW)
    Returns:
        - Updated StreamingDTWRunningState with new row costs and reference frame estimate
    """
    RefFrameCount = ReferenceChroma.shape[1]

    # This imposes a global constraint (like Sakoe-Chiba) so we don't use as much memory when running StreamingDTW
    WindowStart = max(0, RunningState.CurrentReferenceEstimate - SearchHalfWidth)
    WindowEnd = min(RefFrameCount, RunningState.CurrentReferenceEstimate + SearchHalfWidth + 1)

    # Initialize current query row at infinite cost (we want un-reached entries to never be hit). Let's find a way
    # to avoid initializing an array that's the full width of the reference recording.
    CurrentRowCosts = np.full(RefFrameCount, np.inf)

    for RefFrame in range(WindowStart, WindowEnd):
        LocalCost = ComputeChromaCosineDistance(ReferenceChroma[:, RefFrame], NewQueryFrame)

        # Now, we factor in the HMM weight to the cost of this frame. We decrease the cost of frames where 
        # ViterbiNormalizedLogProbs is high (this is not exactly what we discussed, but it accomplishes something similar).
        # The normalized log probabilities are exponentiated to return them to linear probabilities in the range [0,1]
        HMMAdjustedLocalCost = LocalCost - HMMWeight * np.exp(ViterbiNormalizedLogProbs[RefFrame])

        # What was the cost of the last row at the previous reference frame?
        DiagonalPredecessorCost = (RunningState.PreviousRowCosts[RefFrame - 1] if RefFrame > 0 else np.inf)

        # What was the cost of the last row at the current reference frame?
        VerticalPredecessorCost = RunningState.PreviousRowCosts[RefFrame]

        # What is the cost of the current row at the previous reference frame?
        HorizontalPredecessorCost = (CurrentRowCosts[RefFrame - 1] if RefFrame > 0 else np.inf)

        # Again, this is not exactly what we discussed. What's happening here is that HMM is weighting the cosine distance, so
        # we're using a method that statically weights Viterbi and StreamingDTW against each other. We'd like to weight transitions based
        # on how well they achieve the state outlined by the HMM (this has its own faults, however).
        CurrentRowCosts[RefFrame] = HMMAdjustedLocalCost + min(
            DiagonalPredecessorCost,
            VerticalPredecessorCost,
            HorizontalPredecessorCost
        )

    # Grab only the costs within the window
    WindowCosts = CurrentRowCosts[WindowStart:WindowEnd]

    # Create a new reference estimate based on the lowest-cost frame
    NewReferenceEstimate = WindowStart + int(np.argmin(WindowCosts))

    # Update the StreamingDTW state
    return StreamingDTWRunningState(
        PreviousRowCosts = CurrentRowCosts,
        CurrentReferenceEstimate = NewReferenceEstimate,
        CurrentQueryFrameIndex =  RunningState.CurrentQueryFrameIndex + 1
    )

def ProcessNextHMMInfluencedFrame(
    ViterbiState: ViterbiRunningState,
    StreamingDTWState: StreamingDTWRunningState,
    HMM: HMMParameters,
    ReferenceChroma: np.ndarray,
    NewQueryFrame: np.ndarray,
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
        - NewQueryFrame:   (FeatureDim,) current query chroma vector
        - SearchHalfWidth: half-width of the StreamingDTW active window in frames
        - HMMWeight:       scaling factor for the Viterbi cost bias
    Returns:
        - ReferenceFrameEstimate: best estimate of the current reference position
        - UpdatedViterbiState
        - UpdatedStreamingDTWState
    """

    UpdatedViterbiState = StepViterbi(ViterbiState, HMM, NewQueryFrame, SearchHalfWidth)
    UpdatedStreamingDTWState = StepHMMInfluencedDTW(
        StreamingDTWState,
        ReferenceChroma,
        NewQueryFrame,
        UpdatedViterbiState.NormalizedLogProbabilities,
        SearchHalfWidth,
        HMMWeight
    )

    return UpdatedStreamingDTWState.CurrentReferenceEstimate, UpdatedViterbiState, UpdatedStreamingDTWState


def ProcessNextHMMBlendedFrame(
    ViterbiState: ViterbiRunningState,
    StreamingDTWState: StreamingDTWRunningState,
    HMM: HMMParameters,
    ReferenceChroma: np.ndarray,
    NewQueryFrame: np.ndarray,
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
        - NewQueryFrame:   (FeatureDim,) current query chroma vector
        - SearchHalfWidth: half-width of the StreamingDTW active window in frames
        - HMMWeight:       scaling factor for the Viterbi cost bias
    Returns:
        - ReferenceFrameEstimate: best estimate of the current reference position
        - UpdatedViterbiState
        - UpdatedStreamingDTWState
    """

    UpdatedViterbiState = StepViterbi(ViterbiState, HMM, NewQueryFrame, SearchHalfWidth)
    UpdatedStreamingDTWState = StepStreamingDTW(
        StreamingDTWState,
        ReferenceChroma,
        NewQueryFrame,
        SearchHalfWidth
    )

    StreamingDTWEstimate = UpdatedStreamingDTWState.CurrentReferenceEstimate
    ViterbiEstimate = UpdatedViterbiState.CurrentReferenceEstimate

    WeightedMidpointState = int(StreamingDTWEstimate + HMMWeight * (ViterbiEstimate - StreamingDTWEstimate))

    return WeightedMidpointState, UpdatedViterbiState, UpdatedStreamingDTWState

def InitializeHMMStreamingDTW(
    ReferenceChroma: np.ndarray,
    TransitionMatrix: np.ndarray,
    InitialDistribution: np.ndarray,
    Means: list,
    Covars: list
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
