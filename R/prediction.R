#' @importFrom survival Surv
#' @importFrom glmnet glmnet predict.glmnet
#' @importFrom riskRegression predictRisk
#' @importFrom pec predictSurvProb
#' @importFrom prodlim sindex
#' @importFrom casebase absoluteRisk.CompRisk
#' @importFrom stats terms model.matrix delete.response coef predict setNames plogis
NULL

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
#' @importFrom riskRegression predictRisk
#' @export
predictRisk.CompRisk <- function(object, newdata, times, cause, ...) {
    # Extract original covariates from the model object
    all_coef_names <- names(VGAM::coef(object))
    
    # Get unique variable names by removing the :1, :2, etc.
    all_var_names <- unique(gsub(":[1-9]$", "", all_coef_names))
    
    # Extract *only* the original covariate names, excluding 'time' and '(Intercept)'
    coVars <- all_var_names[!grepl("(Intercept)|time", all_var_names)]
    
    # Ensure all required covariates are present
    if (!all(coVars %in% colnames(newdata))) {
        stop(paste("newdata is missing required columns:",
                   paste(setdiff(coVars, colnames(newdata)), collapse=", ")))
    }
    
    # Subset newdata to required covariates
    newdata_subset <- data.matrix(drop(subset(newdata, select = coVars)))
    
    if (missing(cause)) {
        stop("Argument 'cause' is missing. Please specify the event type.")
    }
    
    if (length(times) == 1) {
        a <- casebase::absoluteRisk.CompRisk(object, 
                                             newdata = newdata_subset, 
                                             time = times, 
                                             addZero = FALSE)
        p <- matrix(a, ncol = 1)
    } else {
        a <- casebase::absoluteRisk.CompRisk(object, 
                                             newdata = newdata_subset, 
                                             time = times)
        
        # 'absoluteRisk.CompRisk' adds t=0 by default when length(times) > 1
        if (0 %in% times) {
            # If user *requested* t=0, keep it
            p <- t(a)
        } else {
            # If user did *not* request t=0, remove it
            # The result 'a' has times in columns and subjects in rows
            # We must remove the first column (t=0)
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
    ttob <- stats::terms(object)
    contrasts_arg <- if (length(object@contrasts)) object@contrasts else NULL
    
    # Create the design matrix from newdata
    X <- stats::model.matrix(stats::delete.response(ttob),
                             newdata,
                             contrasts = contrasts_arg,
                             xlev = object@xlevels)
    
    # Reshape coefficients and make predictions
    coeffs <- matrix(stats::coef(object), nrow = ncol(X), byrow = TRUE)
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
#' @importFrom riskRegression predictRisk
#' @export
predictRisk.iCoxBoost <- function(object, newdata, times, cause, ...) {
    p <- stats::predict(object, newdata = newdata, type = "CIF", times = times)
    
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
#' @importFrom riskRegression predictRisk
#' @export
predictRisk.penalizedCompRisk <- function(object, newdata, times, cause, ...) {
    stopifnot(cause == 1)
    
    cb <- object$cb_data
    cn <- colnames(cb$covariates)
    Xnew <- as.matrix(newdata[, cn, drop = FALSE])
    N <- nrow(Xnew); Tt <- length(times)
    
    # --- grab and align coefficients by name ---
    beta_mat <- object$coefficients
    # drop intercept if present
    if (!is.null(rownames(beta_mat)) && "(Intercept)" %in% rownames(beta_mat)) {
        beta_mat <- beta_mat[setdiff(rownames(beta_mat), "(Intercept)"), , drop = FALSE]
    }
    # choose the column you want (here first)
    if (is.matrix(beta_mat)) beta_mat <- beta_mat[, 1, drop = FALSE]
    
    # build beta vector matching covariate columns; fill absent ones with 0
    if (!is.null(rownames(beta_mat))) {
        beta <- stats::setNames(rep(0, length(cn)), cn)
        rn <- intersect(rownames(beta_mat), cn)
        beta[rn] <- as.numeric(beta_mat[rn, 1])
        beta <- as.numeric(beta) # now length(beta) == length(cn)
    } else {
        # no names available -> last resort: require matching length
        stopifnot(length(beta_mat) == length(cn))
        beta <- as.numeric(beta_mat)
    }
    
    # --- map requested times to the cb grid + offsets ---
    gTimes <- sort(unique(cb$time))
    off_by_time <- tapply(cb$offset, cb$time, mean)
    pos <- prodlim::sindex(jump.times = gTimes, eval.times = pmin(times, max(gTimes)))
    off_t <- as.numeric(off_by_time)[pos] # length Tt
    
    # --- hazard and CIF (single cause) ---
    linp <- as.vector(Xnew %*% beta)
    hmat <- sapply(off_t, function(o) stats::plogis(linp + o)) # N x Tt
    
    CIF <- matrix(0, N, Tt); S <- matrix(1, N, Tt)
    for (j in seq_len(Tt)) {
        add <- if (j == 1) hmat[, j] else S[, j-1] * hmat[, j]
        CIF[, j] <- if (j == 1) add else CIF[, j-1] + add
        S[, j] <- if (j == 1) (1 - hmat[, j]) else S[, j-1] * (1 - hmat[, j])
    }
    CIF <- pmin(1, pmax(0, CIF))
    
    return(CIF)
}

#' Predict Survival Probabilities for a oneCSlasso Object
#'
#' S3 method for `predictSurvProb` for an object of
#' class `oneCSlasso`.
#'
#' @param object A fitted object of class `oneCSlasso`.
#' @param newdata A data.frame for which to predict survival.
#' @param times A numeric vector of times to predict at.
#' @param lambdavec The lambda vector used for the fit.
#' @param index The specific index of the `lambdavec` to use
#'   for prediction.
#' @param ... Not used.
#'
#' @return A matrix of survival probabilities (rows=newdata, cols=times).
#' @export
predictSurvProb.oneCSlasso <- function(object, newdata, times, lambdavec, index, ...){
    
    newx <- data.frame(newdata)
    newx <- as.matrix(newx[, object$vars])
    
    lp <- as.numeric(stats::predict(object$glmnet.res, 
                                    newx = newx, 
                                    s = lambdavec[index], 
                                    type = "link"))
    
    # Calculate cumulative baseline hazard
    bsurv <- basesurv(object$response, 
                      object$linear.predictor[[index]], 
                      sort(unique(times)))$cumBaseHaz
    
    # Calculate survival probabilities: S(t) = exp(-H0(t) * exp(lp))
    p <- exp(exp(lp) %*% -t(bsurv))
    
    if (NROW(p) != NROW(newdata) || NCOL(p) != length(times)) {
        stop("Prediction failed")
    }
    p
}


#' Predict Cause-Specific Event Probabilities (Cumulative Incidence)
#'
#' Calculates the cumulative incidence for a specific cause from a
#' `twoCSlassos` object, based on the cause-specific hazards.
#'
#' @param object A fitted object of class `twoCSlassos`.
#' @param newdata A data frame for which to predict.
#' @param times A numeric vector of times to predict at.
#' @param cause The cause of interest.
#' @param lambdavecs A list of lambda vectors (from the object).
#' @param indices A numeric vector (length 2) of the specific lambda
#'   indices to use for prediction.
#' @param ... Not used.
#'
#' @return A matrix of cumulative incidence probabilities.
#' @export
predictEventProb.twoCSlassos <- function(object, newdata, times, cause, lambdavecs, indices, ...){
    
    eTimes <- object$eventTimes
    causes <- object$causes
    
    # Get cause-specific cumulative hazard for the cause of interest
    pred <- predictSurvProb(object$models[[paste("Cause", cause)]], 
                            times = eTimes,
                            newdata = newdata, 
                            lambdavec = lambdavecs[[cause]],
                            index = indices[cause])
    
    pred[pred < .000001] = .000001 # Numerical stability
    cumHaz1 <- -log(pred)
    
    # Get discrete hazards
    Haz1 <- t(apply(cbind(0, cumHaz1), 1, diff))
    
    # Get cumulative hazards for *other* causes
    cumHazOther <- lapply(causes[-match(cause, causes)], 
                          function(c) {
                              cumHaz.c <- -log(predictSurvProb(
                                  object$models[[paste("Cause", c)]],
                                  times = eTimes, 
                                  newdata = newdata,
                                  lambdavec = lambdavecs[[c]], 
                                  index = indices[c]))
                          })
    
    # Calculate overall survival (S(t) = exp(-H_cause1 - H_cause2 - ...))
    lagsurv <- exp(-cumHaz1 - Reduce("+", cumHazOther))
    
    # Calculate cumulative incidence: Int(S(t-) * dH_cause(t))
    cuminc1 <- t(apply(lagsurv * Haz1, 1, cumsum))
    
    # Map to requested time points
    pos <- prodlim::sindex(jump.times = eTimes, eval.times = times)
    p <- cbind(0, cuminc1)[, pos + 1, drop = FALSE]
    p
}


#' @title Predict Absolute Risk for a twoCSlassos Object
#'
#' @description
#' This function is the S3 method for \code{\link[riskRegression]{predictRisk}}
#' for an object of class \code{twoCSlassos}. It serves as a wrapper
#' around \code{predictEventProb.twoCSlassos}.
#'
#' @details
#' This function allows the model to be used with the
#' \code{\link[riskRegression]{Score}} function for evaluating Brier scores
#' and other metrics. It selects the last lambda value from each path
#' Details:
#' This function allows the model to be used with the
#' \code{\link[riskRegression]{Score}} function for evaluating Brier scores
#' and other metrics. It selects the last lambda value from each path
#' by default.
#'
#' @param object A fitted object of class \code{twoCSlassos}.
#' @param newdata A data frame containing the covariate values for which
#'   to predict.
#' @param times A vector of time points at which to predict the
#'   absolute risk.
#' @param cause The cause of interest for which to predict risk.
#' @param lambdavecs A list of lambda vectors. If \code{NULL}, defaults to
#'   \code{object$lambdas}.
#' @param indices An integer vector specifying which index from each
#'   lambda vector to use for prediction. If \code{NULL}, defaults to the
#'   last index (the smallest lambda) of each path.
#' @param ... Additional arguments passed to
#'   \code{predictEventProb.twoCSlassos}.
#'
#' @return A matrix of predicted absolute risks, with rows corresponding
#'   to \code{newdata} and columns to \code{times}.
#'
#' @export
predictRisk.twoCSlassos <- function(object, newdata, times, cause,
                                    lambdavecs = NULL, indices = NULL, ...) {
    
    if (is.null(lambdavecs)) {
        lambdavecs <- object$lambdas
    }
    
    if (is.null(indices)) {
        # Default to the last lambda of each path
        indices <- vapply(lambdavecs, length, integer(1)) 
    }
    
    # Call the internal prediction function
    predictEventProb.twoCSlassos(object, 
                                 newdata = newdata, 
                                 times = times,
                                 cause = cause, 
                                 lambdavecs = lambdavecs, 
                                 indices = indices, 
                                 ...)
}
