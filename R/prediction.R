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

#' Predict Absolute Risk for a Penalized Case-Base Competing Risks Model
#'
#' This function predicts the absolute risk for a specified cause from a
#' `penalizedCompRisk` model object.
#'
#' @param object A fitted model of class `penalizedCompRisk` (or `cbSCRIP`).
#' @param newdata A `data.frame` containing the predictor variables.
#' @param times A numeric vector of time points at which to predict risk.
#' @param cause The event type to predict, given as one of the event codes the
#'   model was fitted on (for example `1` or `2`), not a column position.
#' @param ... Additional arguments passed to other methods.
#'
#' @return A matrix of predicted risks with subjects in rows and time points in columns.
#' @importFrom riskRegression predictRisk
#' @export
predictRisk.penalizedCompRisk <- function(object, newdata, times, cause, ...) {
    if (missing(cause)) {
        cli::cli_abort("Argument {.arg cause} is missing. Please specify the event type.")
    }

    cb <- object$cb_data
    cn <- colnames(cb$covariates)

    # Ensure newdata has the required covariates
    missing_vars <- setdiff(cn, colnames(newdata))
    if (length(missing_vars) > 0) {
        cli::cli_abort("{.arg newdata} is missing covariate{?s}: {.val {missing_vars}}.")
    }

    Xnew <- as.matrix(newdata[, cn, drop = FALSE])
    N  <- nrow(Xnew)
    Tt <- length(times)

    # Extract coefficients
    beta_mat <- object$coefficients
    if (is.null(beta_mat)) cli::cli_abort("Model coefficients not found.")
    K <- ncol(beta_mat)

    # `cause` is an event code, not a column position: column k holds the k-th
    # non-baseline event level, so codes that are not 1..K still resolve.
    event_levels <- sort(unique(cb$event))
    causes <- if (length(event_levels) > 1L) event_levels[-1L] else event_levels[0L]
    if (length(causes) == 0L) {
        # No event levels recorded on the object; fall back to a column index.
        cause_idx <- suppressWarnings(as.integer(cause))
    } else {
        cause_idx <- match(as.character(cause), as.character(causes))
        if (is.na(cause_idx)) {
            cli::cli_abort("{.arg cause} must be one of {.val {causes}}; got {.val {cause}}.")
        }
    }
    if (is.na(cause_idx) || cause_idx < 1L || cause_idx > K) {
        cli::cli_abort("{.arg cause} does not resolve to one of the {K} fitted cause{?s}.")
    }

    # Coefficient rows must cover every covariate; filling gaps with zeros would
    # silently downgrade the prediction to an intercept-only model.
    required <- c(cn, "log(time)", "(Intercept)")
    absent <- setdiff(required, rownames(beta_mat))
    if (length(absent) > 0) {
        cli::cli_abort(c(
            "Coefficient matrix is missing {length(absent)} required row{?s}.",
            x = "Absent: {.val {absent}}"
        ))
    }

    beta_covs  <- beta_mat[cn, , drop = FALSE]
    beta_time  <- beta_mat["log(time)", ]
    beta_int   <- beta_mat["(Intercept)", ]

    # Linear predictor for covariates: N x K
    linp_covs <- Xnew %*% beta_covs   # N x K

    # Integration grid: exact times + intermediate points. When the causes have
    # different shapes the hazard share below varies like a fractional power of
    # t near the origin, so add log-spaced knots there to resolve it.
    nsamp       <- 20L
    time_sorted <- sort(unique(c(0, times)))
    tmax        <- max(time_sorted)
    knots <- seq(0, tmax,
                 length.out = max(2L, (length(time_sorted) - 1L) * nsamp))
    knots <- sort(unique(c(knots, time_sorted,
                           tmax * 10^seq(-6, -1, length.out = 24L))))
    nk    <- length(knots)

    # The model implies lambda_k(t|x) = c_k(x) * t^(gamma_k) exactly, with
    # c_k(x) = exp(intercept_k + x'beta_k) and gamma_k the log(time) coefficient,
    # so the cumulative hazard is closed form and needs no quadrature:
    #     H_k(t|x) = c_k(x) * t^(gamma_k + 1) / (gamma_k + 1)
    # Finite only for gamma_k > -1; at or below that the hazard is not integrable
    # at the origin and absolute risk genuinely does not exist.
    bad <- which(beta_time <= -1)
    if (length(bad) > 0) {
        cli::cli_abort(c(
            "log(time) coefficient is {.val {unname(beta_time[bad])}} for cause {bad}.",
            i = "Values <= -1 make the hazard non-integrable at t = 0, so absolute risk is undefined."
        ))
    }

    log_c <- sweep(linp_covs, 2L, beta_int, "+")   # N x K

    # Exact cumulative hazard, summed over causes: N x nk
    cum_H <- matrix(0, N, nk)
    for (k in seq_len(K)) {
        pw <- beta_time[k] + 1
        cum_H <- cum_H + outer(exp(log_c[, k]) / pw, knots^pw)
    }
    surv_mat <- exp(-cum_H)   # N x nk

    # CIF_k(t) = int_0^t lambda_k S du = int_0^t (lambda_k / lambda) S dH.
    # Integrating S dH over a cell is exactly S(u_j) - S(u_j+1), and the hazard
    # share lambda_k/lambda is taken at the cell midpoint: it is bounded in
    # [0, 1], varies slowly, and the midpoint avoids the 0/0 at the origin.
    # Exact when all causes share a shape, and it makes the CIFs across causes
    # sum to 1 - S(t) by construction.
    mids <- (knots[-nk] + knots[-1L]) / 2
    lam_all   <- matrix(0, N, nk - 1L)
    lam_cause <- matrix(0, N, nk - 1L)
    for (k in seq_len(K)) {
        lam_k <- outer(exp(log_c[, k]), mids^beta_time[k])
        lam_all <- lam_all + lam_k
        if (k == cause_idx) lam_cause <- lam_k
    }
    share <- lam_cause / lam_all
    share[!is.finite(share)] <- 0

    dS <- surv_mat[, -nk, drop = FALSE] - surv_mat[, -1L, drop = FALSE]
    cum_cif <- cbind(0, t(apply(share * dS, 1L, cumsum)))

    # Extract at requested times using findInterval (robust to float near-miss)
    time_idx <- findInterval(times, knots)
    time_idx <- pmax(1L, pmin(time_idx, nk))
    CIF <- cum_cif[, time_idx, drop = FALSE]   # N x Tt

    # Clamp to [0, 1]
    CIF[CIF < 0] <- 0
    CIF[CIF > 1] <- 1

    if (NROW(CIF) != NROW(newdata) || NCOL(CIF) != Tt) {
        cli::cli_abort(paste0("Prediction matrix has wrong dimensions: got ",
                              NROW(CIF), " x ", NCOL(CIF),
                              ", expected ", NROW(newdata), " x ", Tt))
    }

    return(CIF)
}

#' @rdname predictRisk.penalizedCompRisk
#' @export
predictRisk.cbSCRIP <- function(object, newdata, times, cause, ...) {
    predictRisk.penalizedCompRisk(object, newdata, times, cause, ...)
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
