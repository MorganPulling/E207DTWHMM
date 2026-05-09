import math
from pathlib import Path
from typing import Optional

import numba as nb
import numpy as np
from joblib import Parallel, delayed

from Source.IterativeTrainHMM import Exec_IterativeTrainHMM


DATA_WAV_DIR = Path("data/wav_22050_mono")
OUTPUT_DIR = Path("TrainedHMMs")
TRAINING_FRACTION = 0.70


@nb.njit
def VerifyAndNormalizeTransitionMatrix(TransitionMatrix: np.ndarray) -> np.ndarray:
    """
    Re-normalizes each row of TransitionMatrix to sum to 1.

    Exec_IterativeTrainHMM already normalizes, but floating-point accumulation
    across many query recordings can drift row sums slightly off 1.0. States
    that were never visited as a source state receive a self-loop of 1.0.
    """
    for RowIndex in range(TransitionMatrix.shape[0]):
        RowSum = TransitionMatrix[RowIndex].sum()
        if RowSum > 0.0:
            TransitionMatrix[RowIndex] /= RowSum
        else:
            TransitionMatrix[RowIndex, RowIndex] = 1.0
    return TransitionMatrix


class HMM:
    def __init__(
        self,
        TransitionMatrix: np.ndarray,
        InitialDistribution: np.ndarray,
        Means: list[np.ndarray],
        Covars: list[np.ndarray],
        PieceName: str,
        ReferenceRecordingName: str,
    ):
        self.TransitionMatrix = TransitionMatrix
        self.InitialDistribution = InitialDistribution
        self.Means = Means
        self.Covars = Covars
        self.PieceName = PieceName
        self.ReferenceRecordingName = ReferenceRecordingName

    def Save(self, OutputPath: Path) -> None:
        OutputPath.parent.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            str(OutputPath),
            TransitionMatrix = self.TransitionMatrix,
            InitialDistribution = self.InitialDistribution,
            Means = np.array(self.Means),     # (StateCount, FeatureDim)
            Covars = np.array(self.Covars),   # (StateCount, FeatureDim, FeatureDim)
            PieceName = np.array(self.PieceName),
            ReferenceRecordingName = np.array(self.ReferenceRecordingName),
        )

    @classmethod
    def Load(cls, FilePath: Path) -> "HMM":
        Data = np.load(str(FilePath), allow_pickle = True)
        return cls(
            TransitionMatrix = Data["TransitionMatrix"],
            InitialDistribution = Data["InitialDistribution"],
            Means = list(Data["Means"]),
            Covars = list(Data["Covars"]),
            PieceName = str(Data["PieceName"]),
            ReferenceRecordingName = str(Data["ReferenceRecordingName"]),
        )


def GetSortedWavPaths(PieceDir: Path) -> list[Path]:
    return sorted(PieceDir.glob("*.wav"))


def SplitTrainingRecordings(
    AllRecordingPaths: list[Path],
    TrainingFraction: float = TRAINING_FRACTION,
) -> tuple[Path, list[Path]]:
    """
    Splits recordings into a reference and a set of query recordings.

    The first (alphabetically sorted) recording is used as the reference.
    70% of the remaining recordings are used as queries for training.
    The held-out 30% are left on disk for later evaluation.
    """
    ReferenceRecordingPath = AllRecordingPaths[0]
    RemainingPaths = AllRecordingPaths[1:]
    TrainingCount = math.ceil(len(RemainingPaths) * TrainingFraction)
    QueryRecordingPaths = RemainingPaths[:TrainingCount]
    return ReferenceRecordingPath, QueryRecordingPaths


def TrainHMMForPiece(PieceDir: Path, OutputDir: Path) -> Optional[HMM]:
    PieceName = PieceDir.name
    print(f"[{PieceName}] Starting...")

    AllRecordingPaths = GetSortedWavPaths(PieceDir)
    if len(AllRecordingPaths) < 2:
        print(f"[{PieceName}] Skipping: fewer than 2 recordings found.")
        return None

    ReferenceRecordingPath, QueryRecordingPaths = SplitTrainingRecordings(AllRecordingPaths)
    ReferenceRecordingName = ReferenceRecordingPath.stem
    TotalAvailableQueryCount = len(AllRecordingPaths) - 1

    print(f"[{PieceName}] Reference:  {ReferenceRecordingName}")
    print(f"[{PieceName}] Training on {len(QueryRecordingPaths)} / {TotalAvailableQueryCount} query recordings.")

    TransitionMatrix, InitialDistribution, Means, Covars = Exec_IterativeTrainHMM(
        str(ReferenceRecordingPath),
        [str(P) for P in QueryRecordingPaths],
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

    OutputPath = OutputDir / f"{PieceName}.npz"
    TrainedHMM.Save(OutputPath)
    print(f"[{PieceName}] Saved to {OutputPath}")
    return TrainedHMM


def BatchTrainAndSave(
    DataWavDir: Path = DATA_WAV_DIR,
    OutputDir: Path = OUTPUT_DIR,
    JobCount: int = -1,
) -> list[HMM]:
    """
    Trains one HMM per Chopin piece in parallel and saves each to OutputDir.

    JobCount = -1 uses all available CPU cores. Pieces are trained concurrently
    via joblib's loky (process-based) backend so each worker gets a private
    Python interpreter and GIL, avoiding contention on librosa and numpy calls.
    """
    PieceDirs = sorted(Directory for Directory in DataWavDir.iterdir() if Directory.is_dir())
    print(f"Found {len(PieceDirs)} pieces in {DataWavDir}. Training with {JobCount} job(s).\n")
    OutputDir.mkdir(parents = True, exist_ok = True)

    Results = Parallel(n_jobs = JobCount, backend = "loky", verbose = 10)(
        delayed(TrainHMMForPiece)(PieceDir, OutputDir) for PieceDir in PieceDirs
    )

    SuccessfulHMMs = [HMM for HMM in Results if HMM is not None]
    print(f"\nDone. Trained {len(SuccessfulHMMs)} / {len(PieceDirs)} HMMs.")
    return SuccessfulHMMs


if __name__ == "__main__":
    BatchTrainAndSave()
