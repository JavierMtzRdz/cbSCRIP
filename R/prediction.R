#' Predict Absolute Risk for a CompRisk Object
#'
#' This function predicts the absolute risk for a specified cause from a
#' `CompRisk` model object, compatible with the `riskRegression` package.
#'
#' @param object A model object of class `CompRisk`.
#' @param newdata A `data.frame` containing the predictor variables.
#' @param times A numeric vector of time points at which to predict risk.
#' @param cause The specific event type for which to predict the absolute risk.
#' @param ... Additional arguments passed to other methods.
#'
#' @return A matrix of predicted risks with subjects in rows and time points in columns.
#' @export
predictRisk.CompRisk <- function(object, newdata, times, cause, ...) {
    # Extract original covariates from the model object
    coVars <- colnames(object@originalData[, c(grepl("X", colnames(object@originalData)))])
    newdata <- data.matrix(drop(subset(newdata, select = coVars)))
    
    if (missing(cause)) {
        stop("Argument 'cause' is missing. Please specify the event type.")
    }
    
    # Calculate absolute risk and handle different time point scenarios
    if (length(times) == 1) {
        a <- absoluteRisk.CompRisk(object, newdata = newdata, time = times, addZero = FALSE)
        p <- matrix(a, ncol = 1)
    } else {
        a <- casebase::absoluteRisk.CompRisk(object, newdata = newdata, time = times)
        if (0 %in% times) {
            p <- t(a)
        } else {
            # Remove the added time point 0 from the result before transposing
            a <- a[-c(1), -c(1)]
            p <- t(a)
        }
    }
    
    # Validate prediction matrix dimensions
    if (NROW(p) != NROW(newdata) || NCOL(p) != length(times)) {
        stop(paste0("\nPrediction matrix has wrong dimensions:\n",
                    "Requested: ", NROW(newdata), " x ", length(times), "\n",
                    "Provided: ", NROW(p), " x ", NCOL(p), "\n"))
    }
    
    return(p)
}

#' Predict Log-Hazard Ratios for a CompRisk Object
#'
#' This function calculates the linear predictors (log-hazard ratios relative
#' to the baseline) for a competing risks model.
#'
#' @param object A fitted model object.
#' @param newdata A `data.frame` in which to look for variables with which to predict.
#'
#' @return A matrix of linear predictors.
#' @export
predict_CompRisk <- function(object, newdata = NULL) {
    ttob <- terms(object)
    contrasts_arg <- if (length(object@contrasts)) object@contrasts else NULL
    
    # Create the design matrix from newdata
    X <- model.matrix(delete.response(ttob),
                      newdata,
                      contrasts = contrasts_arg,
                      xlev = object@xlevels)
    
    # Reshape coefficients and make predictions
    coeffs <- matrix(coef(object), nrow = ncol(X), byrow = TRUE)
    preds <- X %*% coeffs
    
    # Set informative column names for the log-hazard ratios
    colnames(preds) <- paste0("log(mu[,",
                              seq(2, length(object@typeEvents)),
                              "]/mu[,1])")
    
    return(preds)
}

#' Predict Cumulative Incidence for an iCoxBoost Object
#'
#' This function predicts the cumulative incidence function (CIF) for a specified
#' cause from an `iCoxBoost` model object.
#'
#' @param object A model object of class `iCoxBoost`.
#' @param newdata A `data.frame` containing the predictor variables.
#' @param times A numeric vector of time points at which to predict risk.
#' @param cause The specific event type for which to predict the CIF.
#' @param ... Additional arguments passed to other methods.
#'
#' @return A matrix of predicted cumulative incidences with subjects in rows
#'   and time points in columns.
#' @export
predictRisk.iCoxBoost <- function(object, newdata, times, cause, ...) {
    p <- predict(object, newdata = newdata, type = "CIF", times = times)
    
    # Handle various output shapes from the predict method
    if (is.list(p)) {
        key <- if (!is.null(names(p)) && as.character(cause) %in% names(p)) as.character(cause) else cause
        p <- p[[key]]
    }
    
    if (length(dim(p)) == 3L) {
        p <- p[, , cause, drop = TRUE]
    }
    
    if (is.vector(p)) {
        p <- matrix(p, nrow = NROW(newdata), ncol = length(times), byrow = FALSE)
    }
    
    if (nrow(p) == length(times) && ncol(p) == NROW(newdata)) {
        p <- t(p)
    }
    
    # Validate dimensions and set column names
    stopifnot(nrow(p) == NROW(newdata), ncol(p) == length(times))
    colnames(p) <- format(times)
    
    return(p)
}

#' Predict Absolute Risk for a Penalized CompRisk Object
#'
#' This function predicts the absolute risk for a specified cause from a
#' `penalizedCompRisk` model object.
#'
#' @param object A model object of class `penalizedCompRisk`.
#' @param newdata A `data.frame` containing the predictor variables.
#' @param times A numeric vector of time points at which to predict risk.
#' @param cause The specific event type for which to predict the absolute risk.
#' @param ... Additional arguments passed to other methods.
#'
#' @return A matrix of predicted risks with subjects in rows and time points in columns.
#' @export
predictRisk.penalizedCompRisk <- function(object, newdata, times, cause, ...) {
    if (missing(cause)) {
        stop("Argument 'cause' is missing. Please specify the event type.")
    }
    
    # Calculate absolute risk for the penalized model
    if (length(times) == 1) {
        a <- absoluteRisk.penalized(object, newdata = newdata, time = times, addZero = FALSE)
        p <- matrix(a, ncol = 1)
    } else {
        a <- absoluteRisk.penalized(object, newdata = newdata, time = times)
        if (0 %in% times) {
            p <- t(a)
        } else {
            # Remove the added time point 0 from the result before transposing
            a <- a[-c(1), -c(1)]
            p <- t(a)
        }
    }
    
    # Validate prediction matrix dimensions
    if (NROW(p) != NROW(newdata) || NCOL(p) != length(times)) {
        stop(paste0("\nPrediction matrix has wrong dimensions:\n",
                    "Requested: ", NROW(newdata), " x ", length(times), "\n",
                    "Provided: ", NROW(p), " x ", NCOL(p), "\n"))
    }
    
    return(p)
}