#' Check floating point equality
#' @noRd
#' @keywords internal
same <- function(x, y, tolerance = .Machine$double.eps) {
    abs(x - y) < tolerance
}

#' Calculate Mean Squared Error for Selected Coefficients
#'
#' This function computes the Mean Squared Error (MSE) *only* for the
#' coefficients that were selected by the model (i.e., are non-zero).
#'
#' @note This is not a standard MSE or bias calculation, but rather the
#'   "Mean Squared Error of the selected coefficients." It measures how
#'   accurately the model estimated the parameters it *thought* were important.
#'
#' @param coef A numeric vector of estimated coefficients from a model.
#' @param true_coefs A numeric vector of the same length as `coef` containing
#'   the true, underlying coefficient values.
#'
#' @return A single numeric value representing the mean squared error of the
#'   non-zero elements in `coef` against the corresponding elements in
#'   `true_coefs`. Returns `NaN` if no coefficients are selected (which
#'   can be handled with `na.rm = TRUE` in a downstream `mean()` call).
#'
#' @export
mse_bias <- function(coef, true_coefs) {
    # Find the indices of coefficients the model *selected* (non-zero)
    indices <- which(coef != 0)

    # Calculate MSE only for those selected coefficients
    mse <- mean((coef[indices] - true_coefs[indices])^2)
    return(mse)
}


#' Calculate Variable Selection Performance Metrics
#'
#' Compares a vector of estimated coefficients against a vector of true
#' coefficients to compute common variable selection performance metrics,
#' including sensitivity, specificity, and Matthew's Correlation Coefficient (MCC).
#'
#' @param model_coef A numeric vector of estimated coefficients from a model.
#' @param true_coef A numeric vector of the same length as `model_coef`
#'   containing the true, underlying coefficient values.
#' @param threshold A small numeric value (default: `1e-6`) below which
#'   a coefficient's absolute value is considered to be zero.
#'
#' @return A data.frame with one row containing the following columns:
#' \itemize{
#'   \item `TP`: True Positives (model correctly selected a non-zero-effect variable)
#'   \item `TN`: True Negatives (model correctly did not select a zero-effect variable)
#'   \item `FP`: False Positives (model incorrectly selected a zero-effect variable)
#'   \item `FN`: False Negatives (model incorrectly did not select a non-zero-effect variable)
#'   \item `Sensitivity`: Rate of true positives (TP / (TP + FN))
#'   \item `Specificity`: Rate of true negatives (TN / (TN + FP))
#'   \item `MCC`: Matthew's Correlation Coefficient, a balanced metric for
#'     binary classification that is robust to class imbalance. Ranges
#'     from -1 (total disagreement) to +1 (perfect agreement), with 0
#'     being random chance.
#' }
#'
#' @export
varsel_perc <- function(model_coef, true_coef, threshold = 1e-8) {
    # Identify which variables the model selected (predicted positive)
    selected_vars <- which(abs(model_coef) > threshold)

    # Identify which variables the model did not select (predicted negative)
    non_selected_vars <- which(abs(model_coef) <= threshold)

    # Identify which variables are truly non-zero (actual positive)
    true_vars <- which(abs(true_coef) > threshold)

    # Identify which variables are truly zero (actual negative)
    false_vars <- which(abs(true_coef) <= threshold)


    # Build the 2x2 confusion matrix
    TP <- sum(selected_vars %in% true_vars)
    FP <- sum(selected_vars %in% false_vars)
    TN <- sum(non_selected_vars %in% false_vars)
    FN <- sum(non_selected_vars %in% true_vars)

    # Calculate Sensitivity (True Positive Rate)
    # Handle division by zero if there are no actual positive cases (TP + FN = 0)
    sens <- tryCatch(
        TP / (TP + FN),
        warning = function(w) NaN,
        error = function(e) NaN
    )
    if (is.nan(sens)) sens <- 0 # Or NA, depending on desired behavior

    # Calculate Specificity (True Negative Rate)
    # Handle division by zero if there are no actual negative cases (TN + FP = 0)
    spec <- tryCatch(
        TN / (TN + FP),
        warning = function(w) NaN,
        error = function(e) NaN
    )
    if (is.nan(spec)) spec <- 0 # Or NA, depending on desired behavior

    # Calculate Matthew's Correlation Coefficient (MCC)
    # This is a robust metric that is well-behaved even with class imbalance

    # Pre-calculate the denominator components
    # Use as.numeric to avoid integer overflow issues
    sum_tp_fp <- as.numeric(TP + FP) # Total predicted positive
    sum_tp_fn <- as.numeric(TP + FN) # Total actual positive
    sum_tn_fp <- as.numeric(TN + FP) # Total actual negative
    sum_tn_fn <- as.numeric(TN + FN) # Total predicted negative

    mcc_num <- (as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))
    mcc_den <- sqrt(sum_tp_fp * sum_tp_fn * sum_tn_fp * sum_tn_fn)

    mcc <- if (mcc_den == 0) {
        # If the denominator is zero, it means at least one of the
        # row/column sums in the confusion matrix was zero.
        # By convention, MCC is 0 in this case (no correlation).
        0
    } else {
        mcc_num / mcc_den
    }

    # Calculate Correct Selection Rate (CSR)
    # CSR = 1 if the model selected exactly the true variables (TP == Total True AND FP == 0)
    # Otherwise CSR = 0
    total_true_vars <- length(true_vars)

    csr <- if (TP == total_true_vars && FP == 0) 1 else 0

    dat <- data.frame(
        TP = TP,
        TN = TN,
        FP = FP,
        FN = FN,
        Sensitivity = sens,
        Specificity = spec,
        MCC = mcc,
        CSR = csr
    )

    return(dat)
}
