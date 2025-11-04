#' @importFrom survival Surv
#' @importFrom glmnet glmnet predict.glmnet
#' @importFrom stats approx
NULL

#' Fit a Cause-Specific Cox Lasso Model
#'
#' Fits a penalized Cox model for a specific cause, treating other
#' events as censored. This is a wrapper around \code{glmnet}.
#'
#' @param data A data.frame containing survival time, status, and covariates.
#' @param cause The specific cause of interest to model.
#' @param lambdavec A numeric vector of lambda values for the penalty.
#' @param var_time The column name for the survival time.
#' @param var_status The column name for the event status.
#' @param ... Additional arguments passed to \code{\link[glmnet]{glmnet}}.
#'
#' @return An object of class \code{oneCSlasso} containing the
#'   \code{glmnet} fit, variable names, linear predictors, and response.
#' @export
oneCSlasso <- function(data, cause, lambdavec, var_time = "ftime",
                       var_status = "fstatus", ...){
    
    data <- data.frame(data)
    vars <- colnames(data)[(!colnames(data) %in% c(var_time, var_status))]
    X <- as.matrix(data[, vars])
    
    # Create a Surv object for the specific cause
    y <- survival::Surv(data[[var_time]], data[[var_status]] == cause)
    
    glmnet.res <- glmnet::glmnet(x = X, y = y, alpha = 0.5, standardize = FALSE,
                                 nfold = 5, 
                                 lambda = lambdavec,
                                 family = "cox", ...)
    
    # Pre-calculate linear predictors for all lambdas
    lp <- lapply(lambdavec, function(s) {
        as.numeric(predict(glmnet.res, newx = X, s = s, type = "link"))
    })
    
    out <- list('glmnet.res' = glmnet.res,
                'vars' = vars,
                'linear.predictor' = lp,
                'response' = y)
    
    out$call <- match.call()
    class(out) <- "oneCSlasso"
    out
}


#' Fit Two Cause-Specific Lasso Models
#'
#' A wrapper to fit two separate cause-specific Cox models for a
#' competing risks scenario (Cause 1 vs. Cause 2).
#'
#' @param data A data.frame containing survival time, status, and covariates.
#' @param lambdavecs A list of two numeric vectors, one for each cause's
#'   lambda path.
#' @param ... Additional arguments passed to \code{oneCSlasso}.
#'
#' @return An object of class \code{twoCSlassos}.
#' @export
twoCSlassos <- function(data, lambdavecs, ...){
    
    m1 <- oneCSlasso(data = data, cause = 1, lambdavec = lambdavecs[[1]], ...)
    m2 <- oneCSlasso(data = data, cause = 2, lambdavec = lambdavecs[[2]], ...)
    
    out <- list('models' = list('Cause 1' = m1, 'Cause 2' = m2),
                'eventTimes' = sort(unique(m1$response[, 'time'])),
                'causes' = c(1, 2),
                'lambdas' = lambdavecs)
    
    out$call <- match.call()
    class(out) <- "twoCSlassos"
    out
}

#' Fit Two Cause-Specific Lasso Models with Internal Lambda Grid
#'
#' A helper function that creates a default lambda grid and then
#' calls \code{twoCSlassos}.
#'
#' @param data A data.frame containing survival time, status, and covariates.
#' @param nlambda The number of lambda values to use.
#' @param lambda_min_ratio The ratio of smallest to largest lambda.
#' @param ... Additional arguments passed to \code{twoCSlassos}.
#'
#' @return An object of class \code{twoCSlassos}.
#' @export
two.i.CSlassos <- function(data, nlambda = 100, lambda_min_ratio = 1e-4, ...) {
    
    # Create a default log-linear lambda grid
    lv <- exp(seq(log(lambda_min_ratio), 0, length.out = nlambda))
    
    twoCSlassos(data = data, lambdavecs = list(lv, lv), ...)
}

#' Select Penalties for twoCSlassos
#'
#' A helper function to store a \code{twoCSlassos} object along with
#' a selected set of indices.
#'
#' @param object A fitted \code{twoCSlassos} object.
#' @param ind A vector of indices for the selected lambdas.
#'
#' @return An object of class \code{two.selpen.CSlassos}.
#' @export
two.selpen.CSlassos <- function(object, ind){
    
    out <- list('alllambdas' = object, 'indices' = ind)
    out$call <- match.call()
    class(out) <- "two.selpen.CSlassos"
    out
}


#' Calculate Baseline Survival
#'
#' Internal helper function to calculate the non-parametric baseline
#' hazard and survival estimates (Breslow estimator).
#'
#' @param response A \code{Surv} object.
#' @param lp A numeric vector of linear predictors.
#' @param times.eval A vector of times at which to evaluate.
#' @param centered Whether to center the linear predictors.
#'
#' @return A list containing times, cumulative baseline hazard,
#'   and baseline survival.
#' @export
basesurv <- function(response, lp, times.eval = NULL, centered = FALSE) {
    
    if (is.null(times.eval)) {
        times.eval <- sort(unique(response[, 1]))
    }
    
    # Unique event times
    t.unique <- sort(unique(response[, 1][response[, 2] == 1]))
    alpha <- numeric(length(t.unique)) # Initialize as numeric
    
    # Breslow estimator for baseline hazard steps
    if (length(t.unique) > 0) {
        for (i in 1:length(t.unique)) {
            # d_i / sum(exp(lp_j)) for all j in risk set R_i
            alpha[i] <- sum(response[, 1][response[, 2] == 1] == t.unique[i]) /
                sum(exp(lp[response[, 1] >= t.unique[i]]))
        }
    }
    
    # Interpolate cumulative hazard to requested times
    obj <- stats::approx(t.unique, cumsum(alpha), yleft = 0, xout = times.eval, rule = 2)
    
    if (centered) {
        obj$y <- obj$y * exp(mean(lp))
    }
    
    # Calculate baseline survival
    obj$z <- exp(-obj$y)
    
    names(obj) <- c("times", "cumBaseHaz", "BaseSurv")
    return(obj)
}
