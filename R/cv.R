#' Prepare Penalty Parameters for the Fitting Function
#'
#' A helper function to calculate the lambda parameters required by the mtool
#' fitting function based on the chosen regularization method.
#'
#' @param regularization A string, either 'elastic-net' or 'SCAD'.
#' @param lambda The primary shrinkage parameter.
#' @param alpha The mixing parameter for elastic-net or shape parameter for SCAD.
#' @return A list containing the calculated lambda1 and lambda2 values.
prepare_penalty_params <- function(regularization, lambda, alpha) {
    # Ensure regularization is a character string
    if (!is.character(regularization) || length(regularization) != 1) {
        cli::cli_abort("Argument 'regularization' must be a single string, like 'elastic-net' or 'SCAD'.")
    }
    switch(
        regularization,
        `elastic-net` = {
            # Default alpha for elastic-net is 0.5
            if (is.null(alpha)) {
                cli::cli_inform("Using default alpha = 0.5 for elastic-net.")
                alpha <- 0.5
            }
            # Validate alpha range
            if (alpha < 0 || alpha > 1) {
                cli::cli_abort("'alpha' for elastic-net must be between 0 and 1.")
            }
            
            list(lambda1 = lambda * alpha, 
                 lambda2 = 0.5 * lambda * (1 - alpha))
        },
        
        
        SCAD = {
            # Default 'a' parameter for SCAD is 3.7
            if (is.null(alpha) || alpha < 2) {
                cli::cli_inform("Using default alpha (shape parameter) = 3.7 for SCAD.")
                alpha <- 3.7 
            }
            # Return lambda and the shape parameter
            list(lambda1 = lambda, lambda2 = alpha)
        },
        
        # Default case: executed if 'regularization' does not match any named cases
        cli::cli_abort("Invalid regularization specified. Use 'elastic-net' or 'SCAD'.")
    )
}


#' Create a Case-Base Dataset for Competing Risks
#'
#' Transforms a survival dataset into the case-base format required for fitting
#' a multinomial logistic model for competing risks.
#'
#' @param formula A survival formula, e.g., `Surv(time, event) ~ cov1 + cov2`.
#' @param data A data frame containing the variables in the formula.
#' @param ratio Numeric. The ratio of base samples to case samples.
#' @param ratio_event A numeric vector specifying which event categories to use
#'   for determining the size of the base series. Defaults to "all", which uses
#'   the total number of all non-censored events.
#'
#' @return A list containing the components of the case-base dataset: `time`,
#'   `event`, `covariates`, and `offset`.
#' @export
create_cb_data <- function(formula, data, ratio = 20, ratio_event = 1) {
    
    response_vars <- all.vars(formula[[2]])
    if (length(response_vars) != 2) {
        cli::cli_abort("LHS of formula must be Surv(time, event).")
    }
    time_var <- response_vars[1]
    status_var <- response_vars[2]
    
    if (!all(c(time_var, status_var) %in% names(data))) {
        cli::cli_abort("Time and/or status variables from formula not found in data.")
    }
    
    time <- data[[time_var]]
    status <- data[[status_var]]
    
    cov_matrix <- as.matrix(data[!(names(data) %in% c(status_var, time_var))])
    
    # Determine which events to use for the base series ratio
    all_event_types <- unique(status[status != 0])
    if (identical(ratio_event, "all")) {
        case_count <- sum(status != 0)
    } else if (is.numeric(ratio_event) && all(ratio_event %in% all_event_types)) {
        case_count <- sum(status %in% ratio_event)
    } else {
        cli::cli_abort("'ratio_event' must be 'all' or a numeric vector of valid event types.")
    }
    if (case_count == 0) cli::cli_abort("No events found for the specified 'ratio_event'.")
    
    # Create Base Series
    n <- nrow(data)
    B <- sum(time)
    b_size <- ratio * case_count
    offset <- log(B / b_size)
    
    cli::cli_alert_info("Created base series with {b_size} samples based on {case_count} event(s).")
    
    prob_select <- time / B
    sampled_indices <- sample(n, size = b_size, replace = TRUE, prob = prob_select)
    
    time_bseries <- runif(b_size) * time[sampled_indices]
    cov_bseries <- cov_matrix[sampled_indices, , drop = FALSE]
    event_bseries <- rep(0, b_size) # Event is always 0 for base series
    
    # Extract Case Series (all non-censored events)
    case_indices <- which(status != 0)
    time_cseries <- time[case_indices]
    cov_cseries <- cov_matrix[case_indices, , drop = FALSE]
    event_cseries <- status[case_indices]
    
    # Combine and return
    list(
        time = c(time_bseries, time_cseries),
        event = c(event_bseries, event_cseries),
        covariates = rbind(cov_bseries, cov_cseries),
        offset = rep(offset, length(time_bseries) + length(time_cseries))
    )
}

#' Fit a Penalized Multinomial Model on Case-Base Data
#'
#' A wrapper function to fit a penalized model (Elastic-Net or SCAD) using
#' an optimizer from the `mtool` package.
#'
#' @param cb_data A case-base dataset from `create_cb_data`.
#' @param regularization A string, either 'elastic-net' or 'SCAD'.
#' @param lambda The primary shrinkage parameter.
#' @param alpha The mixing/shape parameter. See `prepare_penalty_params`.
#' @param unpen_cov Integer. The number of leading covariates to leave unpenalized.
#' @param fit_fun The fitting function from `mtool` to use.
#' @param param_start Optional starting values for the coefficients.
#' @param standardize Logical. If TRUE, covariates are scaled to have mean 0 and SD 1.
#' @return The fitted model object from the specified `fit_fun`.
#' @export
fit_cb_model <- function(cb_data, regularization = c('elastic-net', 'SCAD'), lambda, alpha = NULL,
                         fit_fun = mtool::mtool.MNlogistic,
                         param_start = NULL,
                         standardize = TRUE, # Default to TRUE as it's best practice
                         all_event_levels = NULL) {
    
    regularization <- rlang::arg_match(regularization)
    penalty_params <- prepare_penalty_params(regularization, lambda, alpha)
    
    if (is.null(all_event_levels)) all_event_levels <- sort(unique(cb_data$event))
    Y_factor <- factor(cb_data$event, levels = all_event_levels)
    
    # --- Standardization Logic ---
    penalized_covs <- cb_data$covariates
    scaler <- NULL
    if (standardize) {
        scaler_obj <- scale(penalized_covs)
        penalized_covs <- as.matrix(scaler_obj)
        scaler <- list(
            center = attr(scaler_obj, "scaled:center"),
            scale = attr(scaler_obj, "scaled:scale")
        )
        
        # Prevent division by zero for constant variables
        if (any(scaler$scale == 0, na.rm = TRUE)) {
            cli::cli_warn("Some covariates have zero variance and were not scaled.")
            scaler$scale[scaler$scale == 0] <- 1
            
            penalized_covs <- scale(cb_data$covariates, 
                                    center = scaler$center,
                                    scale = scaler$scale)
        }
    }
    
    # Build design matrix
    X <- as.matrix(cbind(penalized_covs, time = log(cb_data$time), 1))
    
    opt_args <- list(X = X, Y = Y_factor, offset = cb_data$offset,
                     N_covariates = 2, # Hardcoded to 2 for intercept and log(time)
                     regularization = regularization,
                     transpose = FALSE, lambda1 = penalty_params$lambda1,
                     lambda2 = penalty_params$lambda2, lambda3 = 0)
    
    if (!is.null(param_start)) {
        opt_args$param_start <- as.matrix(param_start)
    }
    
    fit <- do.call(fit_fun, opt_args)
    
    # --- Coefficient Re-scaling ---
    if (standardize) {
        cli::cli_alert_info("Re-scaling coefficients back to the original data scale.")
        
        # Re-scale both dense and sparse coefficient matrices
        for (coef_type in c("coefficients", "coefficients_sparse")) {
            if (!is.null(fit[[coef_type]])) {
                coefs_scaled <- fit[[coef_type]]
                p_penalized <- ncol(penalized_covs)
                
                # Separate penalized from unpenalized coefficients
                beta_penalized_scaled <- coefs_scaled[1:p_penalized, , drop = FALSE]
                
                # 1. Re-scale penalized coefficients: beta_orig = beta_scaled / scale
                beta_penalized_orig <- sweep(beta_penalized_scaled, 1, scaler$scale, FUN = "/")
                
                # 2. Adjust the intercept
                intercept_adjustment <- colSums(sweep(beta_penalized_scaled, 1, scaler$center / scaler$scale, FUN = "*"), na.rm = TRUE)
                intercept_orig <- coefs_scaled[nrow(coefs_scaled), ] - intercept_adjustment
                
                # 3. Reconstruct the full coefficient matrix on the original scale
                coefs_orig <- rbind(
                    beta_penalized_orig,
                    coefs_scaled[p_penalized + 1, , drop = FALSE], # log(time) is unchanged
                    intercept_orig
                )
                
                # Update row names to match the original structure
                rownames(coefs_orig) <- c(colnames(cb_data$covariates), "log(time)", "(Intercept)")
                
                # Replace the coefficients in the fit object
                fit[[coef_type]] <- coefs_orig
            }
        }
    }
    # Return the fit (with original-scale coefficients) and the scaler
    return(c(fit, scaler = list(scaler)))
}

#' Calculate Multinomial Deviance for a Fitted Case-Base Model
#' @param cb_data A case-base dataset from `create_cb_data`.
#' @param fit_object The output from `fit_cb_model`.
#' @param all_event_levels A vector of all possible event levels to ensure
#'   dimensional consistency.
#' @return The multinomial deviance value.
calc_multinom_deviance <- function(cb_data, fit_object, all_event_levels) {
    # Reconstruct the design matrix exactly as in fit_cb_model
    X <- as.matrix(cbind(cb_data$covariates, time = log(cb_data$time), 1))
    
    fitted_vals <- X %*% fit_object$coefficients
    if (is.vector(fitted_vals)) fitted_vals <- as.matrix(fitted_vals)
    
    pred_mat <- VGAM::multilogitlink(fitted_vals, inverse = TRUE)
    
    Y_fct <- factor(cb_data$event, levels = all_event_levels)
    
    Y_mat <- matrix(0, ncol = length(all_event_levels), nrow = nrow(X))
    
    valid_indices <- !is.na(Y_fct)
    Y_mat[cbind(which(valid_indices), as.integer(Y_fct)[valid_indices])] <- 1
    
    VGAM::multinomial()@deviance(mu = pred_mat, y = Y_mat, w = rep(1, nrow(X)))
}


# --- SECTION 2: CROSS-VALIDATION FUNCTIONS ---

#' Find the Maximum Lambda (lambda_max)
#' @param ... Other arguments passed to find_lambda_max.
#' @return The estimated lambda_max value.
find_lambda_max <- function(cb_data, n_unpenalized, 
                            n_event_types, ncores, 
                            fit_fun, 
                            ...) {
    cli::cli_alert_info("Searching for optimal lambda_max (first round)...")
    null_model_coefs <- n_unpenalized * (n_event_types - 1)
    search_grid <- c(2, 1.5, 1.1, 0.8, 0.5, 0.25, 0.1, 0.08, 0.07, 0.05, 0.01, 0.005)
    search_grid <-exp(seq(log(2.5), log(0.005), length.out = 10))
    
    fit_results <-  pbmcapply::pbmclapply(search_grid, 
                                          function(lambda_val) {
                                              # We only care about the fit object here
                                              fit_cb_model(cb_data, lambda = lambda_val,
                                                           fit_fun = fit_fun,
                                                           ...)$no_non_zero
                                          }, mc.cores = ncores)
    
    
    non_zero_counts <- unlist(fit_results)
    
    upper_idx <- which(non_zero_counts <= null_model_coefs)
    lower_idx <- which(non_zero_counts > null_model_coefs)
    
    if (length(upper_idx) == 0 || length(lower_idx) == 0) {
        cli::cli_warn("Could not bracket lambda_max with the initial grid. Using largest value.")
        return(max(search_grid))
    }
    
    upper_bound <- min(search_grid[upper_idx])
    lower_bound <- max(search_grid[lower_idx])
    
    fine_grid <- seq(lower_bound, upper_bound, length.out = 10)
    
    cli::cli_alert_info("Searching for optimal lambda_max (second round)...")
    
    fine_results <- pbmcapply::pbmclapply(fine_grid, function(lambda_val) {
        fit_cb_model(cb_data, lambda = lambda_val,
                     fit_fun = fit_fun,
                     ...)$no_non_zero
    }, mc.cores = ncores)
    
    lambda_max <- fine_grid[which.min(unlist(fine_results))]
    cli::cli_alert_success("Found lambda_max: {round(lambda_max, 4)}")
    return(lambda_max)
}

#' Generate a Lambda Grid for Regularization
#' @return A numeric vector of lambda values.
create_lambda_grid <- function(lambda_max, epsilon, nlambda, regularization) {
    if (regularization == "elastic-net") {
        rev(exp(seq(log(lambda_max * epsilon), log(lambda_max), length.out = nlambda)))
    } else {
        rev(seq(lambda_max * epsilon, lambda_max, length.out = nlambda))
    }
}

#' Run a Single Fold of Cross-Validation
#' @return A vector of multinomial deviances for the fold.
run_cv_fold <- function(fold_indices, cb_data, lambdagrid, all_event_levels, ...) {
    # 1. Split data into training and validation folds (on their original scale)
    train_cv_data <- lapply(cb_data, function(x) if(is.matrix(x)) x[-fold_indices, , drop = FALSE] else x[-fold_indices])
    test_cv_data  <- lapply(cb_data, function(x) if(is.matrix(x)) x[fold_indices, , drop = FALSE] else x[fold_indices])
    
    # Iterate over lambda grid
    sapply(lambdagrid, function(lambda) {
        # Fit model on the training fold. Internal standardization occurs, but the
        # returned coefficients in model_info are on the original scale.
        model_info <- fit_cb_model(
            train_cv_data,
            lambda = lambda,
            all_event_levels = all_event_levels,
            ...
        )
        
        # Calculate deviance on the original, unscaled test fold.
        # This works because the coefficients have been re-scaled to match.
        calc_multinom_deviance(
            test_cv_data,
            model_info,
            all_event_levels = all_event_levels
        )
    })
}


#' Cross-Validation for Penalized Multinomial Case-Base Models
#' @return An object of class `cb.cv` containing the results.
#' @export
cv_cb_model <- function(formula, data, regularization = 'elastic-net', alpha = 0.5,
                        nfold = 5, nlambda = 50, ncores = parallel::detectCores() / 2,
                        fit_fun = mtool::mtool.MNlogistic, 
                        ratio = 20, ratio_event = 1,
                        ...) {
    
    cb_data <- create_cb_data(formula, data, ratio = ratio,
                              ratio_event = ratio_event)
    
    all_event_levels <- sort(unique(cb_data$event))
    n_event_types <- length(all_event_levels)
    n_unpenalized <- 2 # Intercept and log(time)
    
    lambda_max <- find_lambda_max(cb_data, 
                                  regularization = regularization, alpha = alpha,
                                  n_unpenalized = n_unpenalized, n_event_types = n_event_types,
                                  ncores = ncores,
                                  fit_fun = fit_fun,
                                  ...)
    
    lambdagrid <- create_lambda_grid(lambda_max, epsilon = 0.001, nlambda = nlambda, regularization)
    
    folds <- caret::createFolds(factor(cb_data$event), k = nfold, list = TRUE)
    cli::cli_alert_info("Starting {nfold}-fold cross-validation...")
    
    deviance_matrix <- pbmcapply::pbmclapply(folds, run_cv_fold,
                                             cb_data = cb_data,
                                             lambdagrid = lambdagrid,
                                             regularization = regularization,
                                             alpha = alpha,
                                             all_event_levels = all_event_levels,
                                             fit_fun = fit_fun,
                                             ...,
                                             mc.cores = ncores
    )
    
    deviance_matrix <- do.call(cbind, deviance_matrix)
    
    mean_dev <- rowMeans(deviance_matrix)
    se_dev <- apply(deviance_matrix, 1, sd) / sqrt(nfold)
    
    lambda.min <- lambdagrid[which.min(mean_dev)]
    
    min_dev_upper_bound <- min(mean_dev) + se_dev[which.min(mean_dev)]
    lambda.1se <- max(lambdagrid[mean_dev <= min_dev_upper_bound])
    
    fit.min <- fit_cb_model(
        cb_data,
        regularization = regularization,
        lambda = lambda.min,
        alpha = alpha,
        fit_fun = fit_fun,
        ...
    )
    
    result <- list(
        fit.min = fit.min,
        lambdagrid = lambdagrid,
        deviance_mean = mean_dev,
        deviance_se = se_dev,
        lambda.min = lambda.min,
        lambda.1se = lambda.1se,
        call = match.call()
    )
    
    class(result) <- "cb.cv"
    cli::cli_alert_success("Cross-validation complete.")
    return(result)
}

#' Plot Cross-Validation Results
#' @return A ggplot object.
#' @export
plot.cb.cv <- function(x, ...) {
    plot_data <- data.frame(
        lambda = x$lambdagrid,
        mean_dev = x$deviance_mean,
        upper = x$deviance_mean + x$deviance_se,
        lower = x$deviance_mean - x$deviance_se
    )
    
    ggplot(plot_data, aes(x = lambda, y = mean_dev)) +
        geom_errorbar(aes(ymin = lower, ymax = upper), 
                      width = 0.05, 
                      color = "grey80") +
        geom_point(color = "red") +
        geom_vline(xintercept = x$lambda.min, linetype = "dashed",
                   color = "blue") +
        geom_vline(xintercept = x$lambda.1se, linetype = "dotted",
                   color = "purple") +
        labs(
            x = "Lambda",
            y = "Multinomial Deviance",
            title = "Cross-Validation Performance",
            subtitle = "Blue: Lambda.min | Purple: Lambda.1se"
        ) +
        scale_x_log10() +
        theme_minimal()
}


#' Fit a Penalized Multinomial Model over a Regularization Path (Updated)
#'
#' Fits the case-base model for a sequence of lambda values and returns the
#' path of coefficients. Includes a progress bar for single-core execution.
#'
#' @param formula A survival formula, e.g., `Surv(time, event) ~ .`.
#' @param data The training dataframe.
#' @param regularization Penalty type ('elastic-net' or 'SCAD').
#' @param alpha The elastic-net mixing parameter.
#' @param nlambda The number of lambda values to generate for the path.
#' @param ncores The number of CPU cores to use for parallel processing.
#' @param ... Additional arguments passed to `fit_cb_model` and `create_cb_data`.
#'
#' @return An object of class `cb.path` containing the coefficient paths.
#' @export
path_cb_model <- function(formula, data, regularization = 'elastic-net', alpha = 0.5,
                          nlambda = 100, ncores = 1,
                          ...) {
    
    # 1. Create the full case-base dataset once
    cli::cli_alert_info("Creating case-base dataset...")
    cb_data <- create_cb_data(formula, data, ...)
    
    all_event_levels <- sort(unique(cb_data$event))
    n_event_types <- length(all_event_levels)
    n_unpenalized <- 2 # Intercept and log(time)
    
    # 2. Determine the lambda grid for the path
    lambda_max <- find_lambda_max(cb_data, regularization = regularization, alpha = alpha,
                                  n_unpenalized = n_unpenalized, n_event_types = n_event_types,
                                  ncores = ncores, all_event_levels = all_event_levels, ...)
    
    lambdagrid <- create_lambda_grid(lambda_max, epsilon = 0.005, nlambda = nlambda, regularization)
    
    # 3. Fit the model for each lambda value
    cli::cli_alert_info("Fitting model for {nlambda} lambda values...")
    
    fit_lambda <- function(lambda) {
        fit_cb_model(
            cb_data,
            regularization = regularization,
            lambda = lambda,
            alpha = alpha,
            all_event_levels = all_event_levels,
            standardize = TRUE,
            ...
        )
    }
    
    if (ncores > 1) {
        cli::cli_alert_info("Running on {ncores} cores. Progress bar is disabled.")
        path_fits <- mclapply(lambdagrid, fit_lambda, mc.cores = ncores)
    } else {
        path_fits <- vector("list", nlambda)
        cli::cli_progress_bar("Fitting Path", total = nlambda)
        for (i in seq_along(lambdagrid)) {
            path_fits[[i]] <- fit_lambda(lambdagrid[[i]])
            cli::cli_progress_update()
        }
    }
    
    # 4. Extract the re-scaled coefficients from each fit
    coefficients <- map(path_fits, ~.x$fit$coefficients)
    coefficients_sparse <- map(path_fits, ~.x$fit$coefficients_sparse)
    
    result <- list(
        coefficients = coefficients,
        coefficients_sparse = coefficients_sparse,
        lambdagrid = lambdagrid,
        call = match.call()
    )
    
    class(result) <- "cb.path"
    cli::cli_alert_success("Path fitting complete.")
    return(result)
}



#' Plot Coefficient Paths from a cb.path Object
#'
#' S3 method to plot the regularization path of coefficients.
#'
#' @param x An object of class `cb.path`.
#' @param plot_intercept Logical. Whether to include the intercept in the plot.
#' @param ... Not used.
#'
#' @return A ggplot object.
#' @export
plot.cb.path <- function(x, plot_intercept = FALSE, ...) {
    
    # Wrangle the list of coefficient matrices into a long-format tibble
    plot_data <- imap_dfr(x$coefficients, ~{
        .x %>%
            as.data.frame() %>%
            tibble::rownames_to_column("variable") %>%
            mutate(lambda = x$lambdagrid[.y])
    }) %>%
        pivot_longer(
            cols = -c(variable, lambda),
            names_to = "event_type",
            values_to = "coefficient"
        )
    
    if (!plot_intercept) {
        plot_data <- filter(plot_data, variable != "(Intercept)")
    }
    
    ggplot(plot_data, aes(x = log(lambda), y = coefficient, group = variable, color = variable)) +
        geom_line(alpha = 0.8) +
        facet_wrap(~event_type, scales = "free_y") +
        theme_minimal() +
        guides(color = "none") + # Hide legend for clarity if many variables
        labs(
            x = "log(Lambda)",
            y = "Coefficient Value",
            title = "Coefficient Regularization Paths",
            subtitle = "Each line represents a variable's coefficient as penalty increases"
        )
}