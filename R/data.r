
weibull_hazard <- Vectorize(function(gamma, lambda, t) {
    return(gamma * lambda * t^(gamma - 1))
})

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
                              rate_cens = 0.05, min_time = 1/365, exchangeable = FALSE) {
    
    
    if(length(beta1) != p || length(beta2) != p) stop("Length of beta1 and beta2 must match p.")
    if(!exchangeable && nblocks != length(cor_vals)) stop("Length of cor_vals must match nblocks.")
    
    # Covariate Generation
    if(isTRUE(exchangeable)) {
        # Exchangeable correlation structure
        mat <- matrix(noise_cor, nrow = p, ncol = p)
        cor_exchangeable <- 0.5
        mat[1:num.true, 1:num.true] <- cor_exchangeable
        diag(mat) <- 1
        X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = mat)
    } else {
        # Block-diagonal correlation structure
        vpb <- num.true / nblocks
        correlation_matrix <- matrix(noise_cor, nrow = p, ncol = p)
        for (i in 1:nblocks) {
            start_index <- (i - 1) * vpb + 1
            end_index <- i * vpb
            correlation_matrix[start_index:end_index, start_index:end_index] <- cor_vals[i]
        }
        diag(correlation_matrix) <- 1
        X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = correlation_matrix)
    }
    
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
                              max_time = 1.5, min_time = 1/365, exchangeable = FALSE) {
    
    if(length(beta1) != p || length(beta2) != p) stop("Length of beta1 and beta2 must match p.")
    
    if(isTRUE(exchangeable)) {
        # Exchangeable correlation structure
        mat <- matrix(noise_cor, nrow = p, ncol = p)
        cor_exchangeable <- 0.5
        mat[1:num.true, 1:num.true] <- cor_exchangeable
        diag(mat) <- 1
        X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = mat)
    } else {
        # Block-diagonal correlation structure
        vpb <- num.true / nblocks
        correlation_matrix <- matrix(noise_cor, nrow = p, ncol = p)
        for (i in 1:nblocks) {
            start_index <- (i - 1) * vpb + 1
            end_index <- i * vpb
            correlation_matrix[start_index:end_index, start_index:end_index] <- cor_vals[i]
        }
        diag(correlation_matrix) <- 1
        X <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = correlation_matrix)
    }
    
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
        u2 <- runif(n2)
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
#' @param n Integer, total number of subjects to simulate.
#' @param p Integer, total number of covariates.
#' @param num_true Integer, number of non-zero ("true") covariates.
#' @param setting Integer (1-5), the simulation setting to use.
#' @param iter Integer, the seed for the simulation run for reproducibility.
#' @param sims Integer, optional, the total number of simulations for display purposes.
#'
#' @return A list containing:
#' \item{train}{A data.frame for the training set (75% of data).}
#' \item{test}{A data.frame for the test set (25% of data).}
#' \item{beta1}{The true coefficient vector for cause 1.}
#' \item{beta2}{The true coefficient vector for cause 2.}
#' \item{call}{The function call.}
#' \item{cen.prop}{The proportion of observations for each status (0=censored).}
#'
gen_data <- function(n = 400, p = 300,
                     num_true = 20, setting = 1,
                     iter = runif(1, 0, 9e5), sims = NULL) {
    
    cli::cli_alert_info("Setting: {setting} | Iteration {i}/{sims} | p = {p} | k = {num_true}", i = iter)
    set.seed(iter)
    # set.seed(sample.int(5))
    
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
        beta1_true <- c(rep(1, k / 4),
                        rep(c(0.5, -0.5), times = k / 8),
                        rep(1, k / 4),
                        rep(0, k / 4))
        beta2_true <- c(rep(0, k / 4),
                        rep(c(-0.5, 0.5), times = k / 8),
                        rep(0, k / 4),
                        rep(1, k / 4))
        beta1[nu_ind] <- beta1_true
        beta2[nu_ind] <- beta2_true
    } else if (setting == 5) {
        beta1[nu_ind] <- 1
        beta2[nu_ind] <- -1
    } else {
        stop("'setting' must be an integer between 1 and 5.")
    }
    
    # Data Simulation 
    # Correctly choose simulation function and correlation structure based on 
    if (setting %in% c(1, 2, 3, 4)) {
        # CSH framework for settings 1-4
        sim.data <- cause_hazards_sim(
            n = n, p = p,
            beta1 = beta1, beta2 = beta2,
            num.true = k,
            exchangeable = (setting == 1), # Exchangeable for setting 1
            lambda01 = 0.55, lambda02 = 0.35,
            gamma1 = 1.5, gamma2 = 1.5
        )
    } else if (setting == 5) {
        # Mixture Model framework for setting 5
        sim.data <- cause_subdist_sim(
            n = n, p = p,
            beta1 = beta1, beta2 = beta2,
            num.true = k,
            exchangeable = TRUE, # Exchangeable for setting 5
            cens_max = 1.5
        )
    }
    
    # Train-Test Split 
    train.index <- caret::createDataPartition(sim.data$fstatus, p = 0.75, list = FALSE)
    train <- sim.data[train.index, ]
    test <- sim.data[-train.index, ]
    
    return(list(
        train = train,
        test = test,
        beta1 = beta1,
        beta2 = beta2,
        call = match.call(),
        cen.prop = prop.table(table(factor(sim.data$fstatus, levels = 0:2)))
    ))
}