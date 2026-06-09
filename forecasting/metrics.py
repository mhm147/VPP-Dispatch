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
