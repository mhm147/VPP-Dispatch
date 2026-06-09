import numpy as np

def crps (actual:float, quantilePredictions: dict) -> float:

    """
    Computes CRPS (Continous Ranked Probability Score) for a single observation
    using pinball loss approximation.

    actual: The real observed value.
    quantilePredictions: dict of (quantile level: predicted value)
    """

    #Sort quantile levels and extract their predicted values in order
    quantiles = np.array(sorted(quantilePredictions.keys()))
    predictions = np.array([quantilePredictions[q] for q in quantiles])
    #Errors: How far off is each quantile prediction from actual value
    errors = actual - predictions
    
    #Pinball loss: penalize underestimation more at higher quantiles
    #If actual > predicition: loss = quantile * error
    #If actual < prediction: loss = (1-quantile) * error
    pinball = np.where(errors >=0, quantiles * errors, (quantiles - 1) * errors)

    #CRPS = average pinball loss across all quantiles
    return float (np.mean(pinball))

def meanCRPS(actuals: list, quantileForecasts:list) -> float:
    """
    Computes mean CRPS over multiple observations (hours)
    actuals: list of real observed values (actual cost)
    quantileForecasts: list of dicts, one per observation
    Returns average CRPS
    """

    #Calculating CRPS for each observation and stores it in a list: (zip to pair both lists)
    scores = [crps(actual, forecast) for actual, forecast in zip (actuals, quantileForecasts)]

    #Returns the average across all readings
    return float (np.mean(scores))
