import numpy as np
import ExtractObservations


def MakeInitialDistribution(ReferenceLengthFrames: int):
    """
    Makes an initial distribution assmuing that the reference starts in it first audio frame
    """
    InitialDistribution = np.zeros(ReferenceLengthFrames)

    # We guarantee a start in the first state, which is just the first reference frame
    InitialDistribution[0] = 1;
    return InitialDistribution

def DEPRECATED_MakeStateTransitionMatrix(StateSequenceList: list[np.ndarray], StateCount: int):
    """
    Makes a state transition matrix given an array of state sequences and list of possible states
    Parameters:
        - StateSequenceList a list of state sequences, where each state sequence (an ndarray) specifies and ordering of states
        - StateCount the number of possible states
    Returns:
        - StateTransitionMatrix: A matrix specifying the probability of transitioning between any two states
    """
    StateTransitionMatrix = np.zeros((StateCount, StateCount))

    StateTransitionDictionary = {}
    StateCountDictionary = {}

    for StateSequence in StateSequenceList:

        # Iterate until the second-to-last state since the next state will always be checked.
        # This also ensures that the last state does not get counted for transitions
        for StateIndex in range(len(StateSequence) - 1):

            CurrentState = StateSequence[StateIndex]
            NextState = StateSequence[StateIndex + 1]

            # If the state transition is already tracked...
            if (CurrentState, NextState) in StateTransitionDictionary:
                # ...increment its count by 1...
                StateTransitionDictionary[(CurrentState, NextState)] += 1
            else:
                # ...Otherwise, track it.
                StateTransitionDictionary[(CurrentState, NextState)] = 1

            # If the current state is already tracked...
            if CurrentState in StateCountDictionary:
                # ...increment its count by 1...
                StateCountDictionary[CurrentState] += 1
            else:
                # ...Otherwise, track it
                StateCountDictionary[CurrentState] = 1
    
    StateTransitionRows, StateTransitionColumns = StateTransitionMatrix.shape

    for FromState in range(StateTransitionRows):
        for ToState in range(StateTransitionColumns):
            
            # If the state transition specified by the state transition matrix was ever tracked...
            if (FromState, ToState) in StateTransitionDictionary: 

                # ...its probability is the number of those transitions seen, divided by the number of times the
                # origin ('from') state was seen.
                TransitionProbability = StateTransitionDictionary[(FromState, ToState)] / StateCountDictionary[FromState]
            else:
                # Otherwise, it was never seen, and the probability of making the specified transition is 0
                TransitionProbability = 0
            
            StateTransitionMatrix[FromState, ToState] = TransitionProbability

    return StateTransitionMatrix

def MakeStateTransitionMatrix(StateSequenceList: list[np.ndarray], StateCount: int):
    """
    Makes a state transition matrix given an array of state sequences and list of possible states
    Parameters:
        - StateSequenceList a list of state sequences, where each state sequence (an ndarray) specifies and ordering of states
        - StateCount the number of possible states
    Returns:
        - StateTransitionMatrix: A matrix specifying the probability of transitioning between any two states
    """
    StateTransitionMatrix = np.zeros((StateCount, StateCount))

    for StateSequence in StateSequenceList:
        
        # "From" states are all states but the last
        FromStates = StateSequence[:-1]

        # "To" states are all by the first state
        ToStates = StateSequence[1:]

        # Add 1 at all [From, To] indices
        np.add.at(StateTransitionMatrix, (FromStates, ToStates), 1)

    # Sum along rows to find the total number of transitions from each state
    TotalTransitions = StateTransitionMatrix.sum(axis = 1, keepdims = True)

    # Avoid divide by 0
    TotalTransitions[TotalTransitions == 0] = 1

    # Convert number of transitions to probabilities
    StateTransitionMatrix /= TotalTransitions

    return StateTransitionMatrix

def MakeDistributionMeans(StatesToObservations: dict[int, np.ndarray]):
    """
    Computes the mean observation for all states in the States array

    Parameters:
        - StatesToObservations a mapping from state ID (reference frame) to all associated observations

    Returns:
        - DistributionMeans a vector containing the mean of each observation per state
    """

    StateCount = len(StatesToObservations)

    # Compute the average observation per state as the sum of all associated observations (element-wise, over columns) divided by the total number
    # of observations per state
    DistributionMeans = [np.sum(StatesToObservations[StateID], axis = 0) / len(StatesToObservations[StateID]) for StateID in range(StateCount)]

    return DistributionMeans

def MakeCovarianceMatrices(StatesToObservations: dict[int, np.ndarray]):
    """
    Computes the covariance matrix for all states in the States array.

    Parameters:
        - QueryChromaList :
            a matrix containing a vector of observations in each column
        - StateSequenceArray
            an array of state sequences
        - StatesToObservations
            a mapping from state (reference frame) to all associated observations
    """
    CovarianceMatrices = []

    # Find the average observation of each state
    StateMeans = MakeDistributionMeans(StatesToObservations)

    StateCount = len(StatesToObservations)

    for StateIndex in range(StateCount):

        CurrentStateObservations = StatesToObservations[StateIndex]
        ObservationCount = len(CurrentStateObservations)
        ScaleFactor =  ObservationCount - 1

        StateMean = np.array(StateMeans[StateIndex])
        StateMeanLength = len(StateMean)

        # Initialize the unscaled covariance matrix
        UnscaledCovariance = np.zeros((StateMeanLength, StateMeanLength))

        for Observation in CurrentStateObservations:
            
            Observation = np.array(Observation)
            assert (Observation.shape == StateMean.shape)

            DifferenceFromMean = Observation - StateMean

            # Ensure that the vector containing the difference from the average observation is a column vector
            DifferenceFromMean = DifferenceFromMean.reshape(-1, 1)

            # Compute the outer product of the difference vector with itself, and add it to the covariance matrix
            UnscaledCovariance += (DifferenceFromMean @ (DifferenceFromMean.T))

        CovarianceMatrix = UnscaledCovariance / ScaleFactor

        CovarianceMatrices.append(CovarianceMatrix)

    return CovarianceMatrices

def Exec_TrainHMM(ObservationsList: list[np.ndarray], BacktracePathList: list[np.ndarray]):
    """
    Trains an HMM given a list of observations and corresponding states. The HMM assumes a multivariate Gaussian emission probability model.

    Parameters:
      - ObservationsList a list of matrices, where each matrix contains a list of query chroma features in its columns, and where each column is associated with one audio frame
      - StatesList a list of lists, where each list contains elements specifying the state in each audio frame for a single reference training recording.
      - StateToString a list specifying the states in their string representation
      
    Outputs
      - A the state transition probability matrix
      - pi the distribution of the initial state
      - means a matrix where each row specifies the mean of the distribution for a single state
      - covars a 3D tensor where the first index specifies a state, and the remaining two indices specify the covariance 
        matrix for the state's distribution
    """
    
    StatesToObservations = ExtractObservations.MakeStatesToObservationsDictionary(ObservationsList, BacktracePathList)
    StateCount = len(StatesToObservations)

    # Reference path indices are stored in the first column of each backtrace path
    StateSequenceList = [BacktracePath[:, 0] for BacktracePath in BacktracePathList]

    A = MakeStateTransitionMatrix(StateSequenceList, StateCount)
    Pi = MakeInitialDistribution(len(ObservationsList))

    Means = MakeDistributionMeans(StatesToObservations)
    Covars = MakeCovarianceMatrices(StatesToObservations)

    return (A, Pi, Means, Covars)