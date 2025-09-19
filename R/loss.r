#' Calculate Multinomial Negative Log-Likelihood (Loss)
#'
#' This function computes the total negative log-likelihood for a multinomial
#' logistic regression model with a baseline class.
#'
#' @param coefficients A p x K matrix of model coefficients.
#' @param X A n x p design matrix of predictors.
#' @param Y A numeric vector of length 'n' containing the true class labels.
#'   The baseline class should be coded as 0.
#' @param offset A numeric vector of length 'n' for the offset term.
#' @return The total negative log-likelihood (a single numeric value).
calculate_multinomial_loss <- function(coefficients, X, Y, offset = NULL) {
    # Calculate linear scores (eta)
    eta <- X %*% coefficients
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    
    # Calculate the log of the denominator using the numerically stable log-sum-exp trick
    eta_with_baseline <- cbind(eta, 0)
    max_scores <- apply(eta_with_baseline, 1, max)
    log_denominators <- max_scores + log(rowSums(exp(sweep(eta_with_baseline, 1, max_scores, "-"))))
    
    # Get the linear score corresponding to the true class for each observation
    n <- nrow(X)
    numerators <- numeric(n)
    rows_with_event <- which(Y > 0)
    
    if (length(rows_with_event) > 0) {
        true_class_indices <- Y[rows_with_event]
        # Use matrix indexing to efficiently extract the scores for the true classes
        numerators[rows_with_event] <- eta[cbind(rows_with_event, true_class_indices)]
    }
    
    # Loss = - sum(log-likelihood) = - sum(numerators - log_denominators)
    total_loss <- -sum(numerators - log_denominators)
    
    return(total_loss/n)
}


#' Calculate Penalized Multinomial Negative Log-Likelihood
#'
#' Adds the Elastic Net penalty to the negative log-likelihood loss.
#'
#' @param coefficients A p x K matrix of model coefficients. Assumes the first
#'   row corresponds to the non-penalized intercept.
#' @param X A n x p design matrix.
#' @param Y A numeric vector of length n with class labels (0 for baseline).
#' @param offset A numeric vector of length n for the offset term.
#' @param lambda A single numeric value for the overall regularization strength.
#' @param alpha The Elastic Net mixing parameter (1 for Lasso, 0 for Ridge).
#' @param penalty_weights A numeric vector of weights for each penalized covariate,
#'   of length p-1. Defaults to 1 for all variables.
#' @return The total penalized negative log-likelihood.
calculate_penalized_multinomial_loss <- function(
        coefficients, X, Y, offset = NULL,
        lambda, alpha, penalty_weights = NULL
) {
    
    # --- 1. Calculate the Negative Log-Likelihood part of the loss ---
    nll_loss <- calculate_multinomial_loss(coefficients, X, Y, offset)
    
    # --- 2. Calculate the Regularization Penalty ---
    
    # Isolate coefficients to be penalized (exclude the intercept, assumed to be the first row)
    penalized_coefs <- coefficients
    
    # Set default penalty weights if not provided
    if (is.null(penalty_weights)) {
        penalty_weights <- rep(1, nrow(penalized_coefs))
    }
    
    if (length(penalty_weights) != nrow(penalized_coefs)) {
        stop("Length of 'penalty_weights' must match the number of penalized covariates.")
    }
    
    # Calculate the weighted L1 and L2 components of the penalty
    l1_component <- sum(penalty_weights * rowSums(abs(penalized_coefs)))
    l2_component <- sum(penalty_weights * rowSums(penalized_coefs^2))
    
    # Combine into the full Elastic Net penalty
    en_penalty <- lambda * (alpha * l1_component + (1 - alpha) / 2 * l2_component)
    
    # --- 3. Return the Total Penalized Loss ---
    total_loss <- nll_loss + en_penalty
    
    return(total_loss)
}