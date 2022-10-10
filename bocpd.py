import numpy as np
from scipy.special import logsumexp

def bocpd(data, hazard, model):

    T = len(data)
    log_R = -np.inf * np.ones((T + 1, T + 1))
    log_R[0, 0] = 0 
    log_message = 0

    for t in range(1,T+1):
        # Observe new datum
        x = data[t-1] 

        #Evaluate the hazard function for this interval
        H = hazard(np.array(range(t))) 
        log_H = np.log(H)
        log_1mH = np.log(1-H) 

        # Evaluate predictive probability. 
        log_pred_prob = model.log_pred_prob(t, x)

        # Calculate growth probabilities.
        log_growth_probs = log_pred_prob + log_message + log_1mH

        # Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pred_prob + log_message + log_H)

        # Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

         # 7. Determine run length distribution.
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R

    