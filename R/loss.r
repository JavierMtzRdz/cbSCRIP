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
    
    #  log-sum-exp trick
    eta_with_baseline <- cbind(eta, 0)
    max_scores <- apply(eta_with_baseline, 1, max)
    log_denominators <- max_scores + log(rowSums(exp(sweep(eta_with_baseline, 1, max_scores, "-"))))
    
    # Get the linear score
    n <- nrow(X)
    numerators <- numeric(n)
    rows_with_event <- which(Y > 0)
    
    if (length(rows_with_event) > 0) {
        true_class_indices <- Y[rows_with_event]
        numerators[rows_with_event] <- eta[cbind(rows_with_event, true_class_indices)]
    }
    
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
    
    # Calculate the Negative Log-Likelihood 
    nll_loss <- calculate_multinomial_loss(coefficients, X, Y, offset)
    
    # Calculate the Regularization Penalty
    
    # Isolate coefficients to be penalized
    penalized_coefs <- coefficients
    
    # default penalty
    if (is.null(penalty_weights)) {
        penalty_weights <- rep(1, nrow(penalized_coefs))
    }
    
    if (length(penalty_weights) != nrow(penalized_coefs)) {
        stop("Length of 'penalty_weights' must match the number of penalized covariates.")
    }
    
    l1_component <- sum(penalty_weights * rowSums(abs(penalized_coefs)))
    l2_component <- sum(penalty_weights * rowSums(penalized_coefs^2))
    
    en_penalty <- lambda * (alpha * l1_component + (1 - alpha) / 2 * l2_component)
    
    total_loss <- nll_loss + en_penalty
    
    return(total_loss)
}