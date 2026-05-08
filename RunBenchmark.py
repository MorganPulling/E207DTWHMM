from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import Constants
import HMMOLTW
from BatchTrainAndSaveHMM import HMM, GetSortedWavPaths, SplitTrainingRecordings


# ── Configuration ────────────────────────────────────────────────────────────

DATA_WAV_DIR = Path("data/wav_22050_mono")
DATA_BEAT_DIR = Path("data/annotations_beat")
TRAINED_HMM_DIR = Path("TrainedHMMs")
PLOTS_DIR = Path("benchmark_plots")

SEARCH_HALF_WIDTH = 250
VITERBI_SEARCH_HALF_WIDTH = 150
HMM_INFLUENCED_WEIGHT = 0.55
HMM_BLENDED_WEIGHT = 0.5

METHOD_NAMES = [
    "Offline DTW",
    "Streaming DTW",
    "Viterbi",
    "HMM-Influenced DTW",
    "HMM-Blended DTW",
]
METHOD_COLORS = ["tab:gray", "tab:blue", "tab:red", "tab:green", "tab:orange"]


# ── Beat annotation helpers ───────────────────────────────────────────────────

def FindBeatFilePath(RecordingPath: Path) -> Optional[Path]:
    """
    Finds the .beat file for a recording.

    Most beat files live under DATA_BEAT_DIR/{PieceName}/{stem}.beat, but a few
    exceptions (listed in structure_exceptions.txt) sit directly under DATA_BEAT_DIR.
    """
    Stem = RecordingPath.stem
    PieceName = RecordingPath.parent.name

    PieceSubdirCandidate = DATA_BEAT_DIR / PieceName / f"{Stem}.beat"
    if PieceSubdirCandidate.exists():
        return PieceSubdirCandidate

    TopLevelCandidate = DATA_BEAT_DIR / f"{Stem}.beat"
    if TopLevelCandidate.exists():
        return TopLevelCandidate

    return None


def LoadBeatTimestamps(BeatFilePath: Path) -> np.ndarray:
    """
    Returns an array of beat start times (seconds) from a .beat annotation file.
    Lines beginning with '%' and blank lines are ignored.
    """
    BeatTimes = []
    with open(BeatFilePath) as BeatFile:
        for Line in BeatFile:
            Line = Line.strip()
            if not Line or Line.startswith("%"):
                continue

            # Only the start of the timestamp
            BeatTimes.append(float(Line.split()[0]))
    return np.array(BeatTimes)


# ── Reference-frame series helpers ───────────────────────────────────────────

def PathToQueryReferenceFrameSeries(WarpingPath: np.ndarray, QueryFrameCount: int) -> np.ndarray:
    """
    Converts a chronological DTW warping path (K, 2) into a reference-frame index
    per query frame. Query frames absent from the path are filled by linear interpolation.
    """
    RefFrameByQueryFrame = np.full(QueryFrameCount, np.nan)
    for QueryIndex in range(QueryFrameCount):
        MatchingRefFrames = WarpingPath[WarpingPath[:, 1] == QueryIndex, 0]
        if len(MatchingRefFrames):
            RefFrameByQueryFrame[QueryIndex] = np.median(MatchingRefFrames)

    ValidIndices = np.flatnonzero(~np.isnan(RefFrameByQueryFrame))
    if len(ValidIndices) == 0:
        raise ValueError("DTW warping path covered no query frames")

    MissingIndices = np.flatnonzero(np.isnan(RefFrameByQueryFrame))
    if len(MissingIndices):
        RefFrameByQueryFrame[MissingIndices] = np.interp(
            MissingIndices, ValidIndices, RefFrameByQueryFrame[ValidIndices]
        )
    return RefFrameByQueryFrame


# ── Five estimation methods ───────────────────────────────────────────────────
#
# Each runner accepts (ReferenceChroma, QueryChroma, PreprocessedHMM,
# InitialDistribution) and returns a (QueryFrameCount,) array of estimated
# reference frame indices. RunOfflineDTW ignores the HMM arguments via **_.

def RunOfflineDTW(
    ReferenceChroma: np.ndarray,
    QueryChroma: np.ndarray,
    **_,
) -> np.ndarray:
    _, WarpingPath = librosa.sequence.dtw(
        X = ReferenceChroma, Y = QueryChroma, backtrack = True, global_constraints = True, band_rad = 0.2 
    )
    return PathToQueryReferenceFrameSeries(WarpingPath[::-1], QueryChroma.shape[1])


def RunStreamingDTW(
    ReferenceChroma: np.ndarray,
    QueryChroma: np.ndarray,
    PreprocessedHMM: HMMOLTW.HMMParameters,
    InitialDistribution: np.ndarray,
) -> np.ndarray:
    RefFrameCount = ReferenceChroma.shape[1]
    StreamingDTWState = HMMOLTW.InitializeStreamingDTW(RefFrameCount)
    ViterbiState = HMMOLTW.InitializeViterbi(InitialDistribution)

    Estimates = []
    for QueryFrameIndex in range(QueryChroma.shape[1]):
        ReferenceEstimate, ViterbiState, StreamingDTWState = HMMOLTW.ProcessNextHMMBlendedFrame(
            ViterbiState, StreamingDTWState, PreprocessedHMM,
            ReferenceChroma, QueryChroma[:, QueryFrameIndex],
            SEARCH_HALF_WIDTH, VITERBI_SEARCH_HALF_WIDTH, 0
        )
        Estimates.append(ReferenceEstimate)
    return np.array(Estimates)


def RunViterbi(
    ReferenceChroma: np.ndarray,
    QueryChroma: np.ndarray,
    PreprocessedHMM: HMMOLTW.HMMParameters,
    InitialDistribution: np.ndarray,
) -> np.ndarray:
    RefFrameCount = ReferenceChroma.shape[1]
    StreamingDTWState = HMMOLTW.InitializeStreamingDTW(RefFrameCount)
    ViterbiState = HMMOLTW.InitializeViterbi(InitialDistribution)

    Estimates = []
    for QueryFrameIndex in range(QueryChroma.shape[1]):
        ReferenceEstimate, ViterbiState, StreamingDTWState = HMMOLTW.ProcessNextHMMBlendedFrame(
            ViterbiState, StreamingDTWState, PreprocessedHMM,
            ReferenceChroma, QueryChroma[:, QueryFrameIndex],
            SEARCH_HALF_WIDTH, VITERBI_SEARCH_HALF_WIDTH, 1
        )
        Estimates.append(ReferenceEstimate)
    return np.array(Estimates)


def RunHMMInfluencedStreamingDTW(
    ReferenceChroma: np.ndarray,
    QueryChroma: np.ndarray,
    PreprocessedHMM: HMMOLTW.HMMParameters,
    InitialDistribution: np.ndarray,
) -> np.ndarray:
    RefFrameCount = ReferenceChroma.shape[1]
    StreamingDTWState = HMMOLTW.InitializeStreamingDTW(RefFrameCount)
    ViterbiState = HMMOLTW.InitializeViterbi(InitialDistribution)

    Estimates = []
    for QueryFrameIndex in range(QueryChroma.shape[1]):
        ReferenceEstimate, ViterbiState, StreamingDTWState = HMMOLTW.ProcessNextHMMInfluencedFrame(
            ViterbiState, StreamingDTWState, PreprocessedHMM,
            ReferenceChroma, QueryChroma[:, QueryFrameIndex],
            SEARCH_HALF_WIDTH, VITERBI_SEARCH_HALF_WIDTH, HMM_INFLUENCED_WEIGHT
        )
        Estimates.append(ReferenceEstimate)
    return np.array(Estimates)


def RunHMMBlendedStreamingDTW(
    ReferenceChroma: np.ndarray,
    QueryChroma: np.ndarray,
    PreprocessedHMM: HMMOLTW.HMMParameters,
    InitialDistribution: np.ndarray,
) -> np.ndarray:
    RefFrameCount = ReferenceChroma.shape[1]
    StreamingDTWState = HMMOLTW.InitializeStreamingDTW(RefFrameCount)
    ViterbiState = HMMOLTW.InitializeViterbi(InitialDistribution)

    Estimates = []
    for QueryFrameIndex in range(QueryChroma.shape[1]):
        ReferenceEstimate, ViterbiState, StreamingDTWState = HMMOLTW.ProcessNextHMMBlendedFrame(
            ViterbiState, StreamingDTWState, PreprocessedHMM,
            ReferenceChroma, QueryChroma[:, QueryFrameIndex],
            SEARCH_HALF_WIDTH, VITERBI_SEARCH_HALF_WIDTH, HMM_BLENDED_WEIGHT
        )
        Estimates.append(ReferenceEstimate)
    return np.array(Estimates)


METHOD_RUNNERS = [
    RunOfflineDTW,
    RunStreamingDTW,
    RunViterbi,
    RunHMMInfluencedStreamingDTW,
    RunHMMBlendedStreamingDTW,
]


# ── Error metric ──────────────────────────────────────────────────────────────

def ComputeMAEPercentAtBeats(
    EstimatedRefFramesByQueryFrame: np.ndarray,
    QueryBeatTimestamps: np.ndarray,
    ReferenceBeatTimestamps: np.ndarray,
    ReferenceDurationSeconds: float,
) -> float:
    """
    Computes mean absolute error between estimated and ground-truth reference
    timestamps at each shared beat position, expressed as a percentage of the
    total reference recording duration.

    Ground truth: beat N in the query should align to beat N in the reference.
    The estimated reference time at each query beat is found by interpolating
    the per-frame reference-time series at the beat's timestamp.
    """
    QueryFrameCount = len(EstimatedRefFramesByQueryFrame)
    QueryFrameTimes = librosa.frames_to_time(
        np.arange(QueryFrameCount),
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )
    EstimatedReferenceTimes = librosa.frames_to_time(
        EstimatedRefFramesByQueryFrame,
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )

    EstimatedRefAtQueryBeats = np.interp(
        QueryBeatTimestamps, QueryFrameTimes, EstimatedReferenceTimes
    )

    SharedBeatCount = min(len(QueryBeatTimestamps), len(ReferenceBeatTimestamps))
    AbsoluteErrors = np.abs(
        EstimatedRefAtQueryBeats[:SharedBeatCount] - ReferenceBeatTimestamps[:SharedBeatCount]
    )
    MAESeconds = float(np.mean(AbsoluteErrors))
    return MAESeconds / ReferenceDurationSeconds * 100.0


# ── Per-recording evaluation ──────────────────────────────────────────────────

def EvaluateHeldOutRecording(
    QueryRecordingPath: Path,
    ReferenceChroma: np.ndarray,
    ReferenceBeatTimestamps: np.ndarray,
    ReferenceDurationSeconds: float,
    PreprocessedHMM: HMMOLTW.HMMParameters,
    InitialDistribution: np.ndarray,
) -> Optional[dict]:
    """
    Loads one held-out recording, runs all five alignment methods against the
    reference, and returns per-method MAE percentages keyed by method name.
    Returns None if no beat annotation is found for the query.
    """
    QueryBeatFilePath = FindBeatFilePath(QueryRecordingPath)
    if QueryBeatFilePath is None:
        print(f"    No beat annotation for {QueryRecordingPath.name}, skipping.")
        return None
    QueryBeatTimestamps = LoadBeatTimestamps(QueryBeatFilePath)

    QueryAudio, _ = librosa.load(str(QueryRecordingPath), sr = Constants.DEFAULT_SAMPLE_RATE)
    QueryChroma = librosa.feature.chroma_stft(
        y = QueryAudio,
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )

    SharedKwargs = dict(PreprocessedHMM = PreprocessedHMM, InitialDistribution = InitialDistribution)
    MAEPercents = {}
    RefFrameEstimates = {}
    for MethodName, Runner in zip(METHOD_NAMES, METHOD_RUNNERS):
        EstimatedRefFrames = Runner(ReferenceChroma, QueryChroma, **SharedKwargs)
        MAEPercents[MethodName] = ComputeMAEPercentAtBeats(
            EstimatedRefFrames,
            QueryBeatTimestamps,
            ReferenceBeatTimestamps,
            ReferenceDurationSeconds,
        )
        RefFrameEstimates[MethodName] = EstimatedRefFrames

    return {
        "Recording": QueryRecordingPath.stem,
        "QueryFrameCount": QueryChroma.shape[1],
        "QueryBeatTimestamps": QueryBeatTimestamps,
        **MAEPercents,
        **{f"{Name}_RefFrames": Frames for Name, Frames in RefFrameEstimates.items()},
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _DrawGroupedBars(
    Ax,
    XPositions: np.ndarray,
    MAEMatrix: np.ndarray,
    BarWidth: float,
) -> None:
    MethodCount = len(METHOD_NAMES)
    for MethodIndex, (MethodName, Color) in enumerate(zip(METHOD_NAMES, METHOD_COLORS)):
        Offset = (MethodIndex - MethodCount / 2 + 0.5) * BarWidth
        Bars = Ax.bar(
            XPositions + Offset,
            MAEMatrix[:, MethodIndex],
            width = BarWidth,
            label = MethodName,
            color = Color,
            alpha = 0.85,
        )
        Ax.bar_label(Bars, fmt = "%.1f%%", padding = 2, fontsize = 7)


def PlotPieceResults(PieceName: str, Results: list[dict], PlotsDir: Path) -> None:
    RecordingLabels = [R["Recording"].replace(f"{PieceName}_", "") for R in Results]
    MAEMatrix = np.array([[R[M] for M in METHOD_NAMES] for R in Results])

    FigWidth = max(10, 2.5 * len(RecordingLabels))
    Fig, Ax = plt.subplots(figsize = (FigWidth, 5), constrained_layout = True)

    XPositions = np.arange(len(RecordingLabels))
    _DrawGroupedBars(Ax, XPositions, MAEMatrix, BarWidth = 0.15)

    Ax.set_xticks(XPositions)
    Ax.set_xticklabels(RecordingLabels, rotation = 30, ha = "right", fontsize = 8)
    Ax.set_ylabel("Mean Absolute Error (% of reference duration)")
    Ax.set_title(f"Alignment Error on Held-Out Recordings - {PieceName}")
    Ax.legend(loc = "upper right", fontsize = 8)
    Ax.grid(axis = "y", alpha = 0.3)

    PlotsDir.mkdir(parents = True, exist_ok = True)
    OutputPath = PlotsDir / f"{PieceName}_error.png"
    Fig.savefig(str(OutputPath), dpi = 150)
    plt.close(Fig)
    print(f"  Plot saved -> {OutputPath}")


def PlotSummaryResults(AllPieceResults: dict[str, list[dict]], PlotsDir: Path) -> None:
    PieceNames = list(AllPieceResults.keys())
    MeanMAEMatrix = np.array([
        [np.mean([R[M] for R in AllPieceResults[Piece]]) for M in METHOD_NAMES]
        for Piece in PieceNames
    ])
    ShortPieceNames = [P.replace("Chopin_", "") for P in PieceNames]

    FigWidth = max(10, 3 * len(PieceNames))
    Fig, Ax = plt.subplots(figsize = (FigWidth, 5), constrained_layout = True)

    XPositions = np.arange(len(PieceNames))
    _DrawGroupedBars(Ax, XPositions, MeanMAEMatrix, BarWidth = 0.15)

    Ax.set_xticks(XPositions)
    Ax.set_xticklabels(ShortPieceNames, rotation = 15, ha = "right")
    Ax.set_ylabel("Mean Absolute Error (% of reference duration)")
    Ax.set_title("Mean Alignment Error Across Held-Out Recordings - All Pieces")
    Ax.legend(loc = "upper right", fontsize = 8)
    Ax.grid(axis = "y", alpha = 0.3)

    PlotsDir.mkdir(parents = True, exist_ok = True)
    OutputPath = PlotsDir / "summary_error.png"
    Fig.savefig(str(OutputPath), dpi = 150)
    plt.close(Fig)
    print(f"Summary plot saved -> {OutputPath}")


def PlotAlignmentPaths(
    PieceName: str,
    Result: dict,
    ReferenceBeatTimestamps: np.ndarray,
    PlotsDir: Path,
) -> None:
    """
    Saves one figure per held-out recording showing the alignment path produced
    by each of the five methods (reference time on x, query time on y), with
    the ground-truth beat correspondence overlaid as scatter points.
    """
    RecordingName = Result["Recording"]
    QueryFrameCount = Result["QueryFrameCount"]
    QueryBeatTimestamps = Result["QueryBeatTimestamps"]

    QueryFrameTimes = librosa.frames_to_time(
        np.arange(QueryFrameCount),
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )

    Fig, Ax = plt.subplots(figsize = (13, 5), constrained_layout = True)

    for MethodName, Color in zip(METHOD_NAMES, METHOD_COLORS):
        ReferenceTimes = librosa.frames_to_time(
            Result[f"{MethodName}_RefFrames"],
            sr = Constants.DEFAULT_SAMPLE_RATE,
            hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
        )
        Ax.plot(ReferenceTimes, QueryFrameTimes, label = MethodName, color = Color, linewidth = 1.4, alpha = 0.85)

    SharedBeatCount = min(len(QueryBeatTimestamps), len(ReferenceBeatTimestamps))
    Ax.scatter(
        ReferenceBeatTimestamps[:SharedBeatCount],
        QueryBeatTimestamps[:SharedBeatCount],
        color = "black", s = 10, zorder = 5, label = "Ground truth beats",
    )

    ShortName = RecordingName.replace(f"{PieceName}_", "")
    Ax.set_xlabel("Reference time (s)")
    Ax.set_ylabel("Query time (s)")
    Ax.set_title(f"Alignment Paths - {ShortName}")
    Ax.legend(fontsize = 8)
    Ax.grid(alpha = 0.25)

    OutputDir = PlotsDir / "alignment_paths" / PieceName
    OutputDir.mkdir(parents = True, exist_ok = True)
    OutputPath = OutputDir / f"{RecordingName}.png"
    Fig.savefig(str(OutputPath), dpi = 150)
    plt.close(Fig)
    print(f"  Alignment path saved -> {OutputPath}")


# ── Per-piece benchmark ───────────────────────────────────────────────────────

def BenchmarkPiece(PieceDir: Path, JobCount: int = -1) -> Optional[tuple[str, list[dict], np.ndarray]]:
    PieceName = PieceDir.name
    HMMFilePath = TRAINED_HMM_DIR / f"{PieceName}.npz"

    if not HMMFilePath.exists():
        print(f"[{PieceName}] No trained HMM at {HMMFilePath}, skipping.")
        return None

    print(f"\n[{PieceName}] Loading HMM...")
    TrainedHMM = HMM.Load(HMMFilePath)

    # Reproduce the same split used during training so we evaluate only held-out recordings.
    AllRecordingPaths = GetSortedWavPaths(PieceDir)
    ReferenceRecordingPath = AllRecordingPaths[0]
    _, TrainingQueryPaths = SplitTrainingRecordings(AllRecordingPaths)
    HeldOutRecordingPaths = AllRecordingPaths[1 + len(TrainingQueryPaths):]

    if not HeldOutRecordingPaths:
        print(f"[{PieceName}] No held-out recordings, skipping.")
        return None

    ReferenceBeatFilePath = FindBeatFilePath(ReferenceRecordingPath)
    if ReferenceBeatFilePath is None:
        print(f"[{PieceName}] No beat annotation for reference recording, skipping.")
        return None

    ReferenceAudio, _ = librosa.load(str(ReferenceRecordingPath), sr = Constants.DEFAULT_SAMPLE_RATE)
    ReferenceChroma = librosa.feature.chroma_stft(
        y = ReferenceAudio,
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )
    ReferenceDurationSeconds = librosa.get_duration(
        y = ReferenceAudio, sr = Constants.DEFAULT_SAMPLE_RATE
    )
    ReferenceBeatTimestamps = LoadBeatTimestamps(ReferenceBeatFilePath)

    PreprocessedHMM = HMMOLTW.PreprocessHMMParameters(
        TrainedHMM.TransitionMatrix,
        TrainedHMM.Means,
        TrainedHMM.Covars,
    )

    print(f"[{PieceName}] Evaluating {len(HeldOutRecordingPaths)} held-out recordings with {JobCount} job(s)...")
    RawResults = Parallel(n_jobs = JobCount, backend = "loky", verbose = 11)(
            delayed(EvaluateHeldOutRecording)(
                QueryRecordingPath,
                ReferenceChroma,
                ReferenceBeatTimestamps,
                ReferenceDurationSeconds,
                PreprocessedHMM,
                TrainedHMM.InitialDistribution,
            )
            for QueryRecordingPath in HeldOutRecordingPaths
        )
    Results = [Result for Result in RawResults if Result is not None]
    return PieceName, Results, ReferenceBeatTimestamps


# ── Entry point ───────────────────────────────────────────────────────────────

def Exec_RunBenchmark(
    DataWavDir: Path = DATA_WAV_DIR,
    PlotsDir: Path = PLOTS_DIR,
    JobCount: int = -1,
) -> dict[str, list[dict]]:
    """
    Evaluates all five alignment methods on the held-out recordings for every
    piece that has a trained HMM in TRAINED_HMM_DIR.

    Saves one bar-chart figure per piece and one aggregate summary figure to
    PlotsDir. Returns a dict mapping piece name to list of per-recording results.
    """
    PieceDirs = sorted(Directory for Directory in DataWavDir.iterdir() if Directory.is_dir())
    AllPieceResults: dict[str, list[dict]] = {}

    for PieceDir in PieceDirs:
        Outcome = BenchmarkPiece(PieceDir, JobCount)
        if Outcome is None:
            continue
        PieceName, Results, ReferenceBeatTimestamps = Outcome
        if Results:
            AllPieceResults[PieceName] = Results
            PlotPieceResults(PieceName, Results, PlotsDir)
            for Result in Results:
                PlotAlignmentPaths(PieceName, Result, ReferenceBeatTimestamps, PlotsDir)

    if AllPieceResults:
        PlotSummaryResults(AllPieceResults, PlotsDir)

    print("\nBenchmark complete.")
    return AllPieceResults


if __name__ == "__main__":
    Exec_RunBenchmark()
