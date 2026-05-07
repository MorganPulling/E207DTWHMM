import argparse
import csv
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import Constants
import HMMOLTW
from BatchTrainAndSaveHMM import HMM
from RunBenchmark import (
    EvaluateHeldOutRecording,
    FindBeatFilePath,
    LoadBeatTimestamps,
    METHOD_NAMES,
)
from TrainHMMPerRecording import GetQueryRecordingPaths, GetSortedWavPaths


DATA_WAV_DIR = Path("data/wav_22050_mono")
TRAINED_HMM_DIR = Path("TrainedHMMsPerRecording")
OUTPUT_CSV_PATH = Path("per_recording_hmm_benchmark_results.csv")
TRAINING_FRACTION = 0.80


def GetHeldOutQueryPaths(
    AllRecordingPaths: list[Path],
    ReferenceRecordingPath: Path,
    TrainingFraction: float,
) -> list[Path]:
    TrainingQueryPaths = set(GetQueryRecordingPaths(
        AllRecordingPaths,
        ReferenceRecordingPath,
        TrainingFraction,
    ))
    return [
        RecordingPath
        for RecordingPath in AllRecordingPaths
        if RecordingPath != ReferenceRecordingPath and RecordingPath not in TrainingQueryPaths
    ]


def BuildBenchmarkTasks(
    DataWavDir: Path,
    TrainedHMMDir: Path,
    TrainingFraction: float,
    RequestedPieceName: Optional[str],
    RequestedReferenceName: Optional[str],
) -> list[tuple[str, str, str, list[str]]]:
    Tasks = []

    for PieceDir in sorted(Directory for Directory in DataWavDir.iterdir() if Directory.is_dir()):
        PieceName = PieceDir.name
        if RequestedPieceName is not None and PieceName != RequestedPieceName:
            continue

        AllRecordingPaths = GetSortedWavPaths(PieceDir)
        if len(AllRecordingPaths) < 2:
            print(f"[{PieceName}] Skipping: fewer than 2 recordings found.")
            continue

        for ReferenceRecordingPath in AllRecordingPaths:
            ReferenceRecordingName = ReferenceRecordingPath.stem
            if RequestedReferenceName is not None and ReferenceRecordingName != RequestedReferenceName:
                continue

            HMMFilePath = TrainedHMMDir / PieceName / f"{ReferenceRecordingName}.npz"
            if not HMMFilePath.exists():
                print(f"[{PieceName}] No trained HMM for {ReferenceRecordingName} at {HMMFilePath}, skipping.")
                continue

            HeldOutQueryPaths = GetHeldOutQueryPaths(
                AllRecordingPaths,
                ReferenceRecordingPath,
                TrainingFraction,
            )
            if not HeldOutQueryPaths:
                print(f"[{PieceName}] No held-out queries for {ReferenceRecordingName}, skipping.")
                continue

            Tasks.append(
                (
                    PieceName,
                    str(ReferenceRecordingPath),
                    str(HMMFilePath),
                    [str(QueryPath) for QueryPath in HeldOutQueryPaths],
                )
            )

    return Tasks


def LoadReferenceContext(ReferenceRecordingPath: Path) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    ReferenceBeatFilePath = FindBeatFilePath(ReferenceRecordingPath)
    if ReferenceBeatFilePath is None:
        print(f"    No beat annotation for reference {ReferenceRecordingPath.name}, skipping.")
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

    return ReferenceChroma, ReferenceBeatTimestamps, ReferenceDurationSeconds


def BuildPerBeatRows(
    PieceName: str,
    ReferenceRecordingName: str,
    QueryRecordingName: str,
    Result: dict,
    ReferenceBeatTimestamps: np.ndarray,
    ReferenceDurationSeconds: float,
) -> list[dict]:
    QueryFrameTimes = librosa.frames_to_time(
        np.arange(Result["QueryFrameCount"]),
        sr = Constants.DEFAULT_SAMPLE_RATE,
        hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
    )
    SharedBeatCount = min(len(Result["QueryBeatTimestamps"]), len(ReferenceBeatTimestamps))
    Rows = []

    for BeatIndex in range(SharedBeatCount):
        Row = {
            "Piece": PieceName,
            "ReferenceRecording": ReferenceRecordingName,
            "QueryRecording": QueryRecordingName,
            "BeatIndex": BeatIndex,
            "QueryBeatTimeSeconds": Result["QueryBeatTimestamps"][BeatIndex],
            "ReferenceBeatTimeSeconds": ReferenceBeatTimestamps[BeatIndex],
        }

        for MethodName in METHOD_NAMES:
            EstimatedReferenceTimes = librosa.frames_to_time(
                Result[f"{MethodName}_RefFrames"],
                sr = Constants.DEFAULT_SAMPLE_RATE,
                hop_length = Constants.DEFAULT_HOP_SIZE_SAMPLES,
            )
            EstimatedReferenceTime = float(np.interp(
                Result["QueryBeatTimestamps"][BeatIndex],
                QueryFrameTimes,
                EstimatedReferenceTimes,
            ))
            AbsoluteErrorSeconds = abs(EstimatedReferenceTime - ReferenceBeatTimestamps[BeatIndex])
            Row[f"{MethodName} Estimated Reference Time Seconds"] = EstimatedReferenceTime
            Row[f"{MethodName} Absolute Error Seconds"] = AbsoluteErrorSeconds
            Row[f"{MethodName} Absolute Error Percent"] = AbsoluteErrorSeconds / ReferenceDurationSeconds * 100.0

        Rows.append(Row)

    return Rows


def BenchmarkOneReference(
    PieceName: str,
    ReferenceRecordingPathString: str,
    HMMFilePathString: str,
    HeldOutQueryPathStrings: list[str],
) -> list[dict]:
    ReferenceRecordingPath = Path(ReferenceRecordingPathString)
    HMMFilePath = Path(HMMFilePathString)
    ReferenceRecordingName = ReferenceRecordingPath.stem

    ReferenceContext = LoadReferenceContext(ReferenceRecordingPath)
    if ReferenceContext is None:
        return []

    ReferenceChroma, ReferenceBeatTimestamps, ReferenceDurationSeconds = ReferenceContext
    TrainedHMM = HMM.Load(HMMFilePath)
    PreprocessedHMM = HMMOLTW.PreprocessHMMParameters(
        TrainedHMM.TransitionMatrix,
        TrainedHMM.Means,
        TrainedHMM.Covars,
    )

    Results = []
    for QueryPathString in HeldOutQueryPathStrings:
        QueryRecordingPath = Path(QueryPathString)
        Result = EvaluateHeldOutRecording(
            QueryRecordingPath,
            ReferenceChroma,
            ReferenceBeatTimestamps,
            ReferenceDurationSeconds,
            PreprocessedHMM,
            TrainedHMM.InitialDistribution,
        )
        if Result is None:
            continue

        Results.extend(BuildPerBeatRows(
            PieceName,
            ReferenceRecordingName,
            QueryRecordingPath.stem,
            Result,
            ReferenceBeatTimestamps,
            ReferenceDurationSeconds,
        ))

    return Results


def WriteResultsCsv(Results: list[dict], OutputCsvPath: Path) -> None:
    OutputCsvPath.parent.mkdir(parents = True, exist_ok = True)
    FieldNames = [
        "Piece",
        "ReferenceRecording",
        "QueryRecording",
        "BeatIndex",
        "QueryBeatTimeSeconds",
        "ReferenceBeatTimeSeconds",
    ]
    for MethodName in METHOD_NAMES:
        FieldNames.extend([
            f"{MethodName} Estimated Reference Time Seconds",
            f"{MethodName} Absolute Error Seconds",
            f"{MethodName} Absolute Error Percent",
        ])

    with open(OutputCsvPath, "w", newline = "") as CsvFile:
        Writer = csv.DictWriter(CsvFile, fieldnames = FieldNames)
        Writer.writeheader()
        Writer.writerows(Results)


def PrintSummary(Results: list[dict]) -> None:
    if not Results:
        print("\nNo benchmark results to summarize.")
        return

    print("\nMean MAE (% of reference duration) across held-out queries:")
    for MethodName in METHOD_NAMES:
        MeanError = float(np.mean([
            Result[f"{MethodName} Absolute Error Percent"]
            for Result in Results
        ]))
        print(f"  {MethodName}: {MeanError:.3f}")


def ResolveJobCount(JobCount: int) -> int:
    if JobCount == -1:
        return -1
    if JobCount < 1:
        raise ValueError("--jobs must be -1 or a positive integer")
    return JobCount


def Exec_BenchmarkHMMPerRecording(
    DataWavDir: Path = DATA_WAV_DIR,
    TrainedHMMDir: Path = TRAINED_HMM_DIR,
    OutputCsvPath: Path = OUTPUT_CSV_PATH,
    TrainingFraction: float = TRAINING_FRACTION,
    JobCount: int = -1,
    RequestedPieceName: Optional[str] = None,
    RequestedReferenceName: Optional[str] = None,
) -> list[dict]:
    if not 0.0 < TrainingFraction <= 1.0:
        raise ValueError("TrainingFraction must be greater than 0.0 and no greater than 1.0")

    WorkerCount = ResolveJobCount(JobCount)
    Tasks = BuildBenchmarkTasks(
        DataWavDir,
        TrainedHMMDir,
        TrainingFraction,
        RequestedPieceName,
        RequestedReferenceName,
    )
    TotalTaskCount = len(Tasks)

    print(f"Found {TotalTaskCount} recording-level HMM benchmark task(s).")
    print(f"Data directory: {DataWavDir}")
    print(f"Trained HMM directory: {TrainedHMMDir}")
    print(f"Output CSV: {OutputCsvPath}")
    print(f"Training fraction of non-reference recordings: {TrainingFraction:.0%}")
    print(f"Workers: {WorkerCount}\n")

    if TotalTaskCount == 0:
        return []

    if WorkerCount == 1:
        NestedResults = []
        for PieceName, ReferenceRecordingPathString, HMMFilePathString, HeldOutQueryPathStrings in tqdm(
            Tasks,
            desc = "Benchmarked HMMs",
            unit = "model",
        ):
            NestedResults.append(BenchmarkOneReference(
                PieceName,
                ReferenceRecordingPathString,
                HMMFilePathString,
                HeldOutQueryPathStrings,
            ))
    else:
        NestedResults = Parallel(n_jobs = WorkerCount, backend = "loky", verbose = 10)(
            delayed(BenchmarkOneReference)(
                PieceName,
                ReferenceRecordingPathString,
                HMMFilePathString,
                HeldOutQueryPathStrings,
            )
            for PieceName, ReferenceRecordingPathString, HMMFilePathString, HeldOutQueryPathStrings in Tasks
        )

    Results = [Result for ReferenceResults in NestedResults for Result in ReferenceResults]
    WriteResultsCsv(Results, OutputCsvPath)
    PrintSummary(Results)
    print(f"\nSaved {len(Results)} held-out query result(s) to {OutputCsvPath}.")
    return Results


def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description = (
            "Benchmark recording-level HMMs from TrainHMMPerRecording on the "
            "query recordings excluded from training for each reference."
        )
    )
    Parser.add_argument("--data-dir", type = Path, default = DATA_WAV_DIR)
    Parser.add_argument("--hmm-dir", type = Path, default = TRAINED_HMM_DIR)
    Parser.add_argument("--output-csv", type = Path, default = OUTPUT_CSV_PATH)
    Parser.add_argument("--training-fraction", type = float, default = TRAINING_FRACTION)
    Parser.add_argument("--piece", type = str, default = None)
    Parser.add_argument("--reference", type = str, default = None)
    Parser.add_argument(
        "--jobs",
        type = int,
        default = -1,
        help = "Number of parallel workers. Use -1 for all available CPU cores.",
    )
    return Parser.parse_args()


if __name__ == "__main__":
    Args = ParseArgs()
    Exec_BenchmarkHMMPerRecording(
        DataWavDir = Args.data_dir,
        TrainedHMMDir = Args.hmm_dir,
        OutputCsvPath = Args.output_csv,
        TrainingFraction = Args.training_fraction,
        JobCount = Args.jobs,
        RequestedPieceName = Args.piece,
        RequestedReferenceName = Args.reference,
    )
