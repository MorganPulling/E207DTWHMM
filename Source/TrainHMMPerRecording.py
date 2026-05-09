import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from Source.BatchTrainAndSaveHMM import HMM, VerifyAndNormalizeTransitionMatrix
from Source.IterativeTrainHMM import Exec_IterativeTrainHMM


DATA_WAV_DIR = Path("data/wav_22050_mono")
OUTPUT_DIR = Path("TrainedHMMsPerRecording")
TRAINING_FRACTION = 0.70


def GetSortedWavPaths(PieceDir: Path) -> list[Path]:
    return sorted(PieceDir.glob("*.wav"))


def GetQueryRecordingPaths(
    AllRecordingPaths: list[Path],
    ReferenceRecordingPath: Path,
    TrainingFraction: float,
) -> list[Path]:
    RemainingPaths = [RecordingPath for RecordingPath in AllRecordingPaths if RecordingPath != ReferenceRecordingPath]
    QueryCount = math.ceil(len(RemainingPaths) * TrainingFraction)
    return RemainingPaths[:QueryCount]


def BuildTrainingTasks(
    DataWavDir: Path,
    TrainingFraction: float,
) -> list[tuple[str, str, list[str]]]:
    Tasks = []

    for PieceDir in sorted(Directory for Directory in DataWavDir.iterdir() if Directory.is_dir()):
        AllRecordingPaths = GetSortedWavPaths(PieceDir)
        if len(AllRecordingPaths) < 2:
            print(f"[{PieceDir.name}] Skipping: fewer than 2 recordings found.")
            continue

        for ReferenceRecordingPath in AllRecordingPaths:
            QueryRecordingPaths = GetQueryRecordingPaths(
                AllRecordingPaths,
                ReferenceRecordingPath,
                TrainingFraction,
            )
            Tasks.append(
                (
                    PieceDir.name,
                    str(ReferenceRecordingPath),
                    [str(QueryRecordingPath) for QueryRecordingPath in QueryRecordingPaths],
                )
            )

    return Tasks


def TrainAndSaveOneReference(
    PieceName: str,
    ReferenceRecordingPathString: str,
    QueryRecordingPathStrings: list[str],
    OutputDirString: str,
) -> tuple[str, str, int, str]:
    ReferenceRecordingPath = Path(ReferenceRecordingPathString)
    OutputDir = Path(OutputDirString)
    ReferenceRecordingName = ReferenceRecordingPath.stem

    TransitionMatrix, InitialDistribution, Means, Covars = Exec_IterativeTrainHMM(
        str(ReferenceRecordingPath),
        QueryRecordingPathStrings,
    )
    TransitionMatrix = VerifyAndNormalizeTransitionMatrix(TransitionMatrix)

    TrainedHMM = HMM(
        TransitionMatrix = TransitionMatrix,
        InitialDistribution = InitialDistribution,
        Means = Means,
        Covars = Covars,
        PieceName = PieceName,
        ReferenceRecordingName = ReferenceRecordingName,
    )

    OutputPath = OutputDir / PieceName / f"{ReferenceRecordingName}.npz"
    TrainedHMM.Save(OutputPath)

    return PieceName, ReferenceRecordingName, len(QueryRecordingPathStrings), str(OutputPath)


def ResolveJobCount(JobCount: int) -> int:
    if JobCount == -1:
        return os.cpu_count() or 1
    if JobCount < 1:
        raise ValueError("--jobs must be -1 or a positive integer")
    return JobCount


def TrainAllPerRecording(
    DataWavDir: Path = DATA_WAV_DIR,
    OutputDir: Path = OUTPUT_DIR,
    TrainingFraction: float = TRAINING_FRACTION,
    JobCount: int = -1,
) -> None:
    if not 0.0 < TrainingFraction <= 1.0:
        raise ValueError("TrainingFraction must be greater than 0.0 and no greater than 1.0")

    Tasks = BuildTrainingTasks(DataWavDir, TrainingFraction)
    TotalTaskCount = len(Tasks)
    WorkerCount = ResolveJobCount(JobCount)

    OutputDir.mkdir(parents = True, exist_ok = True)

    print(f"Found {TotalTaskCount} recording-level HMM training task(s).")
    print(f"Data directory: {DataWavDir}")
    print(f"Output directory: {OutputDir}")
    print(f"Training fraction of non-reference recordings: {TrainingFraction:.0%}")
    print(f"Workers: {WorkerCount}\n")

    if TotalTaskCount == 0:
        return

    CompletedCount = 0
    FailedCount = 0

    with tqdm(total = TotalTaskCount, desc = "Trained HMMs", unit = "model") as ProgressBar:
        if WorkerCount == 1:
            for PieceName, ReferenceRecordingPathString, QueryRecordingPathStrings in Tasks:
                ReferenceName = Path(ReferenceRecordingPathString).stem
                tqdm.write(f"[{CompletedCount + 1}/{TotalTaskCount}] Starting {PieceName} / {ReferenceName}")
                try:
                    Result = TrainAndSaveOneReference(
                        PieceName,
                        ReferenceRecordingPathString,
                        QueryRecordingPathStrings,
                        str(OutputDir),
                    )
                except Exception as Error:
                    FailedCount += 1
                    CompletedCount += 1
                    tqdm.write(f"[{CompletedCount}/{TotalTaskCount}] FAILED {PieceName} / {ReferenceName}: {Error}")
                    continue

                CompletedCount += 1
                ProgressBar.update(1)
                _, ReferenceRecordingName, QueryCount, OutputPath = Result
                tqdm.write(
                    f"[{CompletedCount}/{TotalTaskCount}] Saved {PieceName} / "
                    f"{ReferenceRecordingName} using {QueryCount} query recording(s): {OutputPath}"
                )
        else:
            with ProcessPoolExecutor(max_workers = WorkerCount) as Executor:
                FutureToTask = {
                    Executor.submit(
                        TrainAndSaveOneReference,
                        PieceName,
                        ReferenceRecordingPathString,
                        QueryRecordingPathStrings,
                        str(OutputDir),
                    ): (PieceName, Path(ReferenceRecordingPathString).stem)
                    for PieceName, ReferenceRecordingPathString, QueryRecordingPathStrings in Tasks
                }

                for Future in as_completed(FutureToTask):
                    PieceName, ReferenceRecordingName = FutureToTask[Future]
                    try:
                        _, ReferenceRecordingName, QueryCount, OutputPath = Future.result()
                    except Exception as Error:
                        FailedCount += 1
                        CompletedCount += 1
                        tqdm.write(f"[{CompletedCount}/{TotalTaskCount}] FAILED {PieceName} / {ReferenceRecordingName}: {Error}")
                        continue

                    CompletedCount += 1
                    ProgressBar.update(1)
                    tqdm.write(
                        f"[{CompletedCount}/{TotalTaskCount}] Saved {PieceName} / "
                        f"{ReferenceRecordingName} using {QueryCount} query recording(s): {OutputPath}"
                    )

    print(f"\nDone. Succeeded: {CompletedCount - FailedCount}. Failed: {FailedCount}. Total: {TotalTaskCount}.")


def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description = (
            "Train one HMM per audio recording. Each recording is used once as "
            "the reference, with 70% of the remaining recordings from the same "
            "piece used as training queries by default."
        )
    )
    Parser.add_argument("--data-dir", type = Path, default = DATA_WAV_DIR)
    Parser.add_argument("--output-dir", type = Path, default = OUTPUT_DIR)
    Parser.add_argument("--training-fraction", type = float, default = TRAINING_FRACTION)
    Parser.add_argument(
        "--jobs",
        type = int,
        default = -1,
        help = "Number of parallel workers. Use -1 for all available CPU cores.",
    )
    return Parser.parse_args()


if __name__ == "__main__":
    Args = ParseArgs()
    TrainAllPerRecording(
        DataWavDir = Args.data_dir,
        OutputDir = Args.output_dir,
        TrainingFraction = Args.training_fraction,
        JobCount = Args.jobs,
    )
