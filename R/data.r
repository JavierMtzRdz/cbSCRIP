weibull_hazard <- Vectorize(function(gamma, lambda, t) {
    return(gamma * lambda * t^(gamma - 1))
})

#' Covariate Correlation Matrix Used by the Simulation Engines
#'
#' Both simulation engines and [gen_data()] must agree on this matrix, since
#' the signal rescaling in [gen_data()] is computed from it.
#'
#' @param p Integer, total number of covariates.
#' @param num.true Integer, number of non-zero ("true") covariates.
#' @param exchangeable Logical, if TRUE use an exchangeable structure among the
#'   true covariates instead of a block-diagonal one.
#' @param nblocks Integer, number of blocks for the block-diagonal structure.
#' @param cor_vals Numeric vector of length `nblocks`, correlation per block.
#' @param noise_cor Numeric, correlation among the remaining covariates.
#' @return A `p` by `p` correlation matrix.
#' @keywords internal
sim_cor_matrix <- function(p, num.true, exchangeable = FALSE, nblocks = 4,
                           cor_vals = c(0.7, 0.4, 0.6, 0.5), noise_cor = 0.1) {
    mat <- matrix(noise_cor, nrow = p, ncol = p)
    if (isTRUE(exchangeable)) {
        mat[1:num.true, 1:num.true] <- 0.5
    } else {
        vpb <- num.true / nblocks
        for (i in seq_len(nblocks)) {
            idx <- ((i - 1) * vpb + 1):(i * vpb)
            mat[idx, idx] <- cor_vals[i]
        }
    }
    diag(mat) <- 1
    mat
}

#' Rescale a Coefficient Vector to a Target Signal Strength
#'
#' Returns `beta` rescaled so that `sd(X %*% beta) == signal_sd` under the
#' covariate correlation matrix `Sigma`. The support is unchanged, so
#' variable-selection ground truth is unaffected. An all-zero `beta` is
#' returned as-is.
#'
#' @param beta Numeric coefficient vector.
#' @param Sigma Covariate correlation matrix from [sim_cor_matrix()].
#' @param signal_sd Numeric, target standard deviation of the linear predictor.
#' @return The rescaled coefficient vector.
#' @keywords internal
scale_signal <- function(beta, Sigma, signal_sd) {
    s <- sqrt(drop(crossprod(beta, Sigma %*% beta)))
    if (!is.finite(s) || s <= 0) {
        return(beta)
    }
    beta * (signal_sd / s)
}

#' Simulate Competing Risks Data from Cause-Specific Hazards
#'
#' This function generates competing risks survival data using the cause-specific
#' hazards (CSH) framework. It implements the inverse transform sampling method
#' described by Binder et al. (2009) assuming Weibull baseline hazards for each cause.
#'
#' @param p Integer, total number of covariates.
#' @param n Integer, number of subjects to simulate.
#' @param beta1 Numeric vector of length `p`, coefficients for cause 1.
#' @param beta2 Numeric vector of length `p`, coefficients for cause 2.
#' @param nblocks Integer, number of blocks for block-diagonal correlation.
#' @param cor_vals Numeric vector of length `nblocks`, correlation for each block.
#' @param num.true Integer, number of non-zero ("true") covariates.
#' @param lambda01 Numeric, the baseline rate parameter for the Weibull hazard of cause 1.
#' @param lambda02 Numeric, the baseline rate parameter for the Weibull hazard of cause 2.
#' @param gamma1 Numeric, the baseline shape parameter for the Weibull hazard of cause 1.
#' @param gamma2 Numeric, the baseline shape parameter for the Weibull hazard of cause 2.
#' @param max_time Numeric, the maximum follow-up time (administrative censoring).
#' @param noise_cor Numeric, the correlation for noise variables.
#' @param rate_cens Numeric, the rate parameter for the exponential censoring distribution.
#' @param min_time Numeric, the minimum possible event time.
#' @param exchangeable Logical, if TRUE, use an exchangeable correlation structure
#'   for true covariates instead of a block-diagonal one.
#'
#' @return A data.frame with `n` rows and `p+2` columns ('fstatus', 'ftime',
#'   and covariates X1...Xp).
#'
cause_hazards_sim <- function(p, n, beta1, beta2,
                              nblocks = 4, cor_vals = c(0.7, 0.4, 0.6, 0.5), num.true = 20,
                              lambda01 = 0.55, lambda02 = 0.10,
                              gamma1 = 1.5, gamma2 = 1.5, max_time = 1.5, noise_cor = 0.1,
                              rate_cens = 0.05, min_time = 1e-4, exchangeable = FALSE) {
    if (length(beta1) != p || length(beta2) != p) stop("Length of beta1 and beta2 must match p.")
    if (!exchangeable && nblocks != length(cor_vals)) stop("Length of cor_vals must match nblocks.")

    # Covariate Generation
    correlation_matrix <- sim_cor_matrix(
        p, num.true,
        exchangeable = exchangeable, nblocks = nblocks,
        cor_vals = cor_vals, noise_cor = noise_cor
    )
    X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = correlation_matrix)

    # vent Time Generation
    # Calculate individual-specific rate parameters
    lambda1_i <- as.vector(lambda01 * exp(X %*% beta1))
    lambda2_i <- as.vector(lambda02 * exp(X %*% beta2))

    # Define the root-finding function: F(t) - u = 0
    cdf_solver <- function(t, g1, l1, g2, l2, u) {
        H1 <- l1 * t^g1 # Cumulative hazard for cause 1
        H2 <- l2 * t^g2 # Cumulative hazard for cause 2
        return((1 - exp(-(H1 + H2))) - u)
    }

    # Generate a uniform random variable for each subject
    u <- stats::runif(n)

    # For each subject, find the event time 't' by solving cdf_solver for 0
    times <- sapply(1:n, function(i) {
        stats::uniroot(
            cdf_solver,
            interval = c(0, max_time * 2),
            extendInt = "upX",
            g1 = gamma1, l1 = lambda1_i[i],
            g2 = gamma2, l2 = lambda2_i[i],
            u = u[i]
        )$root
    })

    # At the generated event time, determine the cause based on relative hazards
    hazard1 <- gamma1 * lambda1_i * times^(gamma1 - 1)
    hazard2 <- gamma2 * lambda2_i * times^(gamma2 - 1)
    prob_cause1 <- hazard1 / (hazard1 + hazard2)

    # Handle cases where total hazard is zero
    prob_cause1[is.nan(prob_cause1)] <- 0

    event_type <- stats::rbinom(n = n, size = 1, prob = prob_cause1)
    c.ind <- ifelse(event_type == 1, 1, 2)

    # Generate censoring times from an exponential distribution
    cens_times <- stats::rexp(n = n, rate = rate_cens)

    # Apply censoring: if censoring time is earlier, status is 0
    c.ind[cens_times < times] <- 0
    times <- pmin(times, cens_times)

    # Apply administrative censoring and winsorize time
    c.ind[times >= max_time] <- 0
    times <- pmin(times, max_time)
    times[times < min_time] <- min_time

    sim.data <- data.frame(fstatus = c.ind, ftime = times)
    X_df <- as.data.frame(X)
    colnames(X_df) <- paste0("X", seq_len(p))
    sim.data <- cbind(sim.data, X_df)

    return(sim.data)
}

#' Simulate Competing Risks Data from a Mixture Model
#'
#' @description
#' This function generates competing risks survival data from a mixture model framework.
#' A subject is first assigned a latent cause of failure, and the event time is
#' then drawn from a cause-specific Weibull distribution.
#'
#' **Note:** This method is distinct from and does **not** necessarily produce data
#' that follows a proportional sub-distribution hazards (Fine & Gray) model.
#'
#' @param n Integer, number of subjects to simulate.
#' @param p Integer, total number of covariates.
#' @param beta1 Numeric vector of length `p`, coefficients for cause 1.
#' @param beta2 Numeric vector of length `p`, coefficients for cause 2.
#' @param num.true Integer, number of non-zero ("true") covariates.
#' @param mix_p Numeric (0-1), base probability for the mixture assignment.
#' @param cor_vals Numeric vector, correlation for each block in block-diagonal structure.
#' @param noise_cor Numeric, the correlation for noise variables.
#' @param nblocks Integer, number of blocks for block-diagonal correlation.
#' @param lambda1 Numeric, the baseline rate parameter for the Weibull distribution of cause 1.
#' @param rho1 Numeric, the baseline shape parameter for the Weibull distribution of cause 1.
#' @param lambda2 Numeric, the baseline rate parameter for the Weibull distribution of cause 2.
#' @param rho2 Numeric, the baseline shape parameter for the Weibull distribution of cause 2.
#' @param cens_max Numeric, the maximum time for the uniform censoring distribution.
#' @param max_time Numeric, the maximum follow-up time (administrative censoring).
#' @param min_time Numeric, the minimum possible event time.
#' @param exchangeable Logical, if TRUE, use an exchangeable correlation structure.
#'
#' @return A data.frame with `n` rows and `p+2` columns ('fstatus', 'ftime',
#'   and covariates X1...Xp).
#'
cause_subdist_sim <- function(n, p, beta1, beta2, num.true = 20, mix_p = 0.5,
                              cor_vals = c(0.7, 0.4, 0.6, 0.5), noise_cor = 0.1,
                              nblocks = 4, lambda1 = 1, rho1 = 4,
                              lambda2 = 0.8, rho2 = 10, cens_max = 1.5,
                              max_time = 1.5, min_time = 1e-4, exchangeable = FALSE) {
    if (length(beta1) != p || length(beta2) != p) stop("Length of beta1 and beta2 must match p.")

    correlation_matrix <- sim_cor_matrix(
        p, num.true,
        exchangeable = exchangeable, nblocks = nblocks,
        cor_vals = cor_vals, noise_cor = noise_cor
    )
    X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = correlation_matrix)

    eta1_prob <- X %*% beta1
    prob_not_cause1 <- (1 - mix_p)^exp(eta1_prob)
    prob_cause1 <- 1 - prob_not_cause1
    c.ind <- 1 + stats::rbinom(n, 1, prob = prob_cause1) # 1 = cause 2, 2 = cause 1

    # To match description: beta1 affects event 1, beta2 affects event 2
    c.ind <- ifelse(c.ind == 1, 2, 1)


    ftime <- numeric(n)

    # Subjects assigned to cause 1
    is_cause1 <- which(c.ind == 1)
    n1 <- length(is_cause1)
    if (n1 > 0) {
        eta1_time <- X[is_cause1, ] %*% beta1
        u1 <- stats::runif(n1)
        t1 <- (-log(u1) / (lambda1 * exp(eta1_time)))^(1 / rho1)
        ftime[is_cause1] <- t1
    }

    # Subjects assigned to cause 2
    is_cause2 <- which(c.ind == 2)
    n2 <- length(is_cause2)
    if (n2 > 0) {
        eta2_time <- X[is_cause2, ] %*% beta2
        u2 <- stats::runif(n2)
        t2 <- (-log(u2) / (lambda2 * exp(eta2_time)))^(1 / rho2)
        ftime[is_cause2] <- t2
    }

    cens_times <- stats::runif(n, min = 0, max = cens_max)

    # Apply censoring
    fstatus <- c.ind # Start with original cause
    fstatus[cens_times < ftime] <- 0
    ftime <- pmin(ftime, cens_times)

    # Apply administrative censoring and winsorize
    fstatus[ftime >= max_time] <- 0
    ftime <- pmin(ftime, max_time)
    ftime[ftime < min_time] <- min_time

    sim.data <- data.frame(fstatus = fstatus, ftime = ftime)
    X_df <- as.data.frame(X)
    colnames(X_df) <- paste0("X", seq_len(p))
    sim.data <- cbind(sim.data, X_df)

    return(sim.data)
}

#' Generate Competing Risks Survival Data for Simulation Studies
#'
#' @description
#' This function generates complex competing risks data based on five distinct
#' settings described in the simulation study. It handles the creation of
#' coefficient vectors, covariate correlation structures, and calls the appropriate
#' underlying simulation engine (either Cause-Specific Hazards or a Mixture Model).
#'
#' The five settings are:
#' 1.  **CSH: Single effects on endpoint 1.**
#' 2.  **CSH: Single effects on both endpoints (block structure).**
#' 3.  **CSH: Opposing effects.**
#' 4.  **CSH: Mixture of single and opposing effects.**
#' 5.  **Mixture Model: Opposing effects (violates CSH proportionality).**
#'
#' @param n_train Integer, number of subjects in training set.
#' @param n_test Integer, number of subjects in test set.
#' @param p Integer, total number of covariates.
#' @param num_true Integer, number of non-zero ("true") covariates.
#' @param setting Integer (1-5), the simulation setting to use.
#' @param iter Integer, the seed for the simulation run for reproducibility.
#' @param sims Integer, optional, the total number of simulations for display purposes.
#' @param signal_sd Numeric, the target standard deviation of the linear
#'   predictor `X %*% beta` for each cause. The setting's coefficient pattern is
#'   rescaled to hit this value so that signal strength stays comparable across
#'   `p` and `num_true`. Set to `NULL` to use the raw pattern.
#'
#' @return A list containing:
#' \item{train}{A data.frame for the training set (size n_train).}
#' \item{test}{A data.frame for the test set (size n_test).}
#' \item{beta1}{The true coefficient vector for cause 1.}
#' \item{beta2}{The true coefficient vector for cause 2.}
#' \item{call}{The function call.}
#' \item{cen.prop}{The proportion of observations for each status (0=censored).}
#' @export
gen_data <- function(n_train = 300, n_test = 100, p = 300,
                     num_true = 20, setting = 1,
                     iter = stats::runif(1, 0, 9e5), sims = NULL,
                     signal_sd = 2) {
    set.seed(iter)
    seed <- as.integer(paste(sample.int(9, 5, replace = TRUE), collapse = ""))
    set.seed(seed)
    cli::cli_alert_info("Setting: {setting} | Iteration {iter}/{sims} | seed = {seed} | p = {p} | k = {num_true}")

    beta1 <- rep(0, p)
    beta2 <- rep(0, p)
    nu_ind <- seq_len(num_true)
    k <- num_true

    # Define coefficient patterns based on the setting
    if (setting == 1) {
        beta1[nu_ind] <- 1
        beta2[nu_ind] <- 0
    } else if (setting == 2) {
        beta1[nu_ind] <- rep(c(1, 0, 1, 0), each = k / 4)
        beta2[nu_ind] <- rep(c(0, 1, 0, 1), each = k / 4)
    } else if (setting == 3) {
        beta1[nu_ind] <- rep(c(0.5, -0.5), times = k / 2)
        beta2[nu_ind] <- rep(c(-0.5, 0.5), times = k / 2)
    } else if (setting == 4) {
        beta1_true <- c(
            rep(1, k / 4),
            rep(c(0.5, -0.5), times = k / 8),
            rep(1, k / 4),
            rep(0, k / 4)
        )
        beta2_true <- c(
            rep(0, k / 4),
            rep(c(-0.5, 0.5), times = k / 8),
            rep(0, k / 4),
            rep(1, k / 4)
        )
        beta1[nu_ind] <- beta1_true
        beta2[nu_ind] <- beta2_true
    } else if (setting == 5) {
        beta1[nu_ind] <- 1
        beta2[nu_ind] <- -1
    } else {
        cli::cli_abort("'setting' must be an integer between 1 and 5.")
    }

    # Hold the signal strength fixed as p and k grow. 
    if (!is.null(signal_sd)) {
        cor_mat <- sim_cor_matrix(p, k, exchangeable = setting %in% c(1, 5))
        beta1 <- scale_signal(beta1, cor_mat, signal_sd)
        beta2 <- scale_signal(beta2, cor_mat, signal_sd)
    }

    # Data Simulation
    n_total <- n_train + n_test

    # Correctly choose simulation function and correlation structure based on
    if (setting %in% c(1, 2, 3, 4)) {
        # CSH framework for settings 1-4
        sim.data <- cause_hazards_sim(
            n = n_total, p = p,
            beta1 = beta1, beta2 = beta2,
            num.true = k,
            exchangeable = (setting == 1), # Exchangeable for setting 1
            lambda01 = 0.55, lambda02 = 0.35,
            gamma1 = 1.5, gamma2 = 1.5
        )
    } else if (setting == 5) {
        # Mixture Model framework for setting 5
        sim.data <- cause_subdist_sim(
            n = n_total, p = p,
            beta1 = beta1, beta2 = beta2,
            num.true = k,
            exchangeable = TRUE, # Exchangeable for setting 5
            cens_max = 1.5
        )
    }

    # Train-Test Split
    # Use the first n_train for training and the rest for testing to ensure exact sizes
    # Since data is generated i.id., this simple split is valid random split.

    train <- sim.data[1:n_train, ]
    test <- sim.data[(n_train + 1):n_total, ]

    return(list(
        train = train,
        test = test,
        beta1 = beta1,
        beta2 = beta2,
        call = match.call(),
        cen.prop = prop.table(table(factor(sim.data$fstatus, levels = 0:2)))
    ))
}
