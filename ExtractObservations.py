import numpy as np
import librosa as lb
from Constants import *

def DEPRECATED_MakeStatesToObservationsDictionary(Query: np.ndarray, Reference: np.ndarray, SampleRate: int = DEFAULT_SAMPLE_RATE, HopSizeSamples: int = DEFAULT_HOP_SIZE_SAMPLES):
    """
    Maps reference frames to zero or more query chroma features (observations).
    """

    # Big idea: perform DTW to match query frames to reference frames. Using the query chroma feature matrix (which matches chroma features to query
    # frames) and DTW results, build a dictionary mapping state ID (reference frame) to query chroma features.

    QueryChroma = lb.feature.chroma_stft(y = Query, sr = SampleRate, hop_length = HopSizeSamples)
    ReferenceChroma = lb.feature.chroma_stft(y = Reference, sr = SampleRate, hop_length = HopSizeSamples)

    _, Path = lb.sequence.dtw(ReferenceChroma, QueryChroma)

    ReferencePathIndices = Path[:, 0]
    QueryPathIndices = Path[:, 1]

    StatesToObservationsDictionary = {}

    for Index, QueryPathIndex in enumerate(QueryPathIndices):
        
        QueryChromaFeature = QueryChroma[:, QueryPathIndex]
        ReferencePathIndex = ReferencePathIndices[Index]

        if (ReferencePathIndex) in StatesToObservationsDictionary:
            # ...add a new observation.
            StatesToObservationsDictionary[ReferencePathIndex].append(QueryChromaFeature)
        else:
            # Otherwise, track it
            StatesToObservationsDictionary[ReferencePathIndex] = [QueryChromaFeature]

    return StatesToObservationsDictionary

def MakeStatesToObservationsDictionary(QueryChromaList: list[np.ndarray], BacktracePathsList: list[np.ndarray]):
    """
    Maps reference frames to zero or more query chroma features (observations).
    Parameters:
    - QueryChromaList: A list of chroma matrices for several query recordings.
    - BacktracePathsList: A list of backtrace paths for several query-reference recording pairs.
    Returns:
    - StatesToObservationsDictionary: A dictionary mapping all found states in a reference recording to query chroma features.
    """

    # Big idea: perform DTW to match query frames to reference frames. Using the query chroma feature matrix (which matches chroma features to query
    # frames) and DTW results, build a dictionary mapping state ID (reference frame) to query chroma features.

    assert(len(QueryChromaList) == len(BacktracePathsList))

    StatesToObservationsDictionary = dict()

    for QueryChromaMatrix, BacktracePath in zip(QueryChromaList, BacktracePathsList):

        ReferencePathIndices = BacktracePath[:, 0]
        QueryPathIndices = BacktracePath[:, 1]

        StatesToObservationsDictionary = {}

        for Index, QueryPathIndex in enumerate(QueryPathIndices):
            
            QueryChromaFeature = QueryChromaMatrix[:, QueryPathIndex]
            ReferencePathIndex = ReferencePathIndices[Index]

            if (ReferencePathIndex) in StatesToObservationsDictionary:

                StatesToObservationsDictionary[ReferencePathIndex].append(QueryChromaFeature)
            else:

                StatesToObservationsDictionary[ReferencePathIndex] = [QueryChromaFeature]

    return StatesToObservationsDictionary