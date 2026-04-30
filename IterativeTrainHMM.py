import numpy as np
import librosa
import TrainHMM
import Constants


def UpdateHMMParameters(
    OldTransitionCounts: np.ndarray,
    OldObservationSums: list[np.ndarray],
    OldOuterProductSums: list[np.ndarray],
    NewQueryChromaMatrix: np.ndarray,
    NewBacktracePath: np.ndarray
):
    """
    Accumulates HMM parameter contributions from one query recording into running totals.

    All three accumulators grow with each call; the caller divides by counts after the
    loop to calculate the final transition matrix, means, and covariance matrices.

    Parameters:
        - OldTransitionCounts: (StateCount, StateCount) matrix of raw transition counts
        - OldObservationSums:  list of (FeatureDim,) arrays, one sum-of-observations per state
        - OldOuterProductSums: list of (FeatureDim, FeatureDim) arrays, one sum of outer
                               products (x @ x^T) per state, used to calculate covariance later
        - NewQueryChromaMatrix: (FeatureDim, QueryFrameCount) chroma features for one query
        - NewBacktracePath:     (K, 2) DTW path where column 0 is reference (state) indices
                                and column 1 is query frame indices
    Returns:
        - UpdatedTransitionCounts: OldTransitionCounts with new transitions added
        - UpdatedObservationSums:  OldObservationSums with new observations added
        - UpdatedOuterProductSums: OldOuterProductSums with new outer products added
    """
    ReferenceFrameSequence = NewBacktracePath[:, 0]
    QueryFrameSequence = NewBacktracePath[:, 1]

    # A transition is counted only at DTW steps where the query frame increments by 1.
    QueryIncrements = np.diff(QueryFrameSequence) == 1
    FromStates = ReferenceFrameSequence[:-1][QueryIncrements]
    ToStates = ReferenceFrameSequence[1:][QueryIncrements]
    
    # Add transition counts. The caller row-normalizes to get probabilities.
    np.add.at(OldTransitionCounts, (FromStates, ToStates), 1)

    # For each state, gather all observations aligned to it and accumulate sums
    StateCount = len(OldObservationSums)
    for StateIndex in range(StateCount):

        # Which query frames correspond to the current reference frame (which we treat as a state)?
        # We need this to determine which chroma features correspond to any given reference frame (i.e., state).
        AlignedQueryFrames = QueryFrameSequence[ReferenceFrameSequence == StateIndex]

        if len(AlignedQueryFrames) == 0:
            continue

        # Shape (FeatureDim, AlignedFrameCount)
        # Which chroma features are aligned to which reference frames?
        AlignedObservations = NewQueryChromaMatrix[:, AlignedQueryFrames]

        # This function is run within a loop, so addition is necessary
        OldObservationSums[StateIndex] += AlignedObservations.sum(axis = 1)

        # Accumulate a sum of outer products. NormalizeHMMParameters calculates covariance as:
        # Covariance = (OuterProductSum - N * Mean @ Mean^T) / (N - 1)
        OldOuterProductSums[StateIndex] += AlignedObservations @ AlignedObservations.T

    return OldTransitionCounts, OldObservationSums, OldOuterProductSums


def NormalizeHMMParameters(
    TransitionCounts: np.ndarray,
    ObservationSums: list[np.ndarray],
    OuterProductSums: list[np.ndarray],
    ObservationCounts: np.ndarray
):
    """
    Converts accumulated sums into a normalized transition matrix, means, and covariances.

    Parameters:
        - TransitionCounts: (StateCount, StateCount) raw transition counts
        - ObservationSums:  list of (FeatureDim,) summed observations per state
        - OuterProductSums: list of (FeatureDim, FeatureDim) summed outer products per state
        - ObservationCounts: (StateCount,) number of observations aligned to each state
    Returns:
        - A:      row-normalized state transition probability matrix
        - Means:  list of per-state mean vectors
        - Covars: list of per-state sample covariance matrices
    """
    RowTotalTransitions = TransitionCounts.sum(axis = 1, keepdims = True)

    # Avoid dividing by 0, and assume that every state is reached once
    RowTotalTransitions[RowTotalTransitions == 0] = 1
    A = TransitionCounts / RowTotalTransitions

    StateCount = len(ObservationSums)
    Means = []
    Covars = []

    for StateIndex in range(StateCount):
        N = ObservationCounts[StateIndex]

        # Avoid dividing by 0
        Mean = ObservationSums[StateIndex] / max(N, 1)
        Means.append(Mean)

        if N > 1:
            # Compute covariance from accumulated outer products:
            # Covariance = (sum(x_i @ x_i^T) - N * mu @ mu^T) / (N - 1)
            Covariance = (OuterProductSums[StateIndex] - N * np.outer(Mean, Mean)) / (N - 1)
        else:
            # Use the identity matrix for the covariance matrix when there aren't enough observations to estimate covariance
            Covariance = np.eye(len(Mean))

        Covars.append(Covariance)

    return A, Means, Covars


def Exec_IterativeTrainHMM(ReferenceRecordingPath: str, QueryRecordingPaths: list[str]):
    """
    Loads a reference recording and a set of query recordings, aligns each query to the
    reference via DTW, then trains HMM parameters by accumulating statistics across all
    alignments and normalizing once at the end.

    Parameters:
        - ReferenceRecordingPath : Path to the reference audio file
        - QueryRecordingPaths :    List of paths to query audio files
    Returns:
        - A :      State transition probability matrix
        - Pi :     Initial distribution
        - Means :  List of per-state mean vectors
        - Covars : List of per-state covariance matrices
    """

    ReferenceAudio, SampleRate = librosa.load(ReferenceRecordingPath, sr = Constants.DEFAULT_SAMPLE_RATE)
    ReferenceChroma = librosa.feature.chroma_stft(
        y = ReferenceAudio, sr = SampleRate, hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES
    )

    # Each reference frame is one HMM state
    FeatureCount, StateCount = ReferenceChroma.shape

    # Initialize accumulators to 0
    TransitionCounts = np.zeros((StateCount, StateCount))
    ObservationSums = [np.zeros(FeatureCount) for _ in range(StateCount)]
    OuterProductSums = [np.zeros((FeatureCount, FeatureCount)) for _ in range(StateCount)]

    ObservationCounts = np.zeros(StateCount, dtype = int)

    for QueryRecordingPath in QueryRecordingPaths:
        QueryAudio, _ = librosa.load(QueryRecordingPath, sr = Constants.DEFAULT_SAMPLE_RATE)
        QueryChromaMatrix = librosa.feature.chroma_stft(
            y = QueryAudio, sr = SampleRate, hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES
        )

        # dtw returns the path from [end, end] back to [0, 0], so flip it to get chronological order.
        # WarpingPath[:, 0] = reference frame indices, WarpingPath[:, 1] = query frame indices.
        ### NOTE: For memory purposes, maybe we consider using a Sakoe-Chiba band to store fewer values (Claude rec.) ###
        _, WarpingPath = librosa.sequence.dtw(X = ReferenceChroma, Y = QueryChromaMatrix, backtrack = True)
        BacktracePath = WarpingPath[::-1]

        TransitionCounts, ObservationSums, OuterProductSums = UpdateHMMParameters(
            TransitionCounts, ObservationSums, OuterProductSums,
            QueryChromaMatrix, BacktracePath
        )

        ReferenceStateSequence = BacktracePath[:, 0]

        # bincount gives the number of times each reference frame index appears in the path,
        # i.e., how many query frames (and equivalently, how many observations) were aligned to each state.
        # So, ObservationCounts[0] indicates how many times reference frame 0 was found with an observation.
        ObservationCounts += np.bincount(ReferenceStateSequence, minlength = StateCount)

    A, Means, Covars = NormalizeHMMParameters(
        TransitionCounts, ObservationSums, OuterProductSums, ObservationCounts
    )

    Pi = TrainHMM.MakeInitialDistribution(StateCount)

    return A, Pi, Means, Covars
