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
    }
    switch(
        regularization,
        `elastic-net` = {
            # Default alpha for elastic-net is 0.5
            if (is.null(alpha)) {
                alpha <- 0.5
            }
            
            list(lambda1 = lambda * alpha, 
                 lambda2 = 0.5 * lambda * (1 - alpha))
        },
        
        
        SCAD = {
            # Default 'a' parameter for SCAD is 3.7
            if (is.null(alpha))  {
                alpha <- 3.7 
                
            } else if (alpha < 2)  {
                alpha <- 3.7 
                
            }
            # Return lambda and the shape parameter
            list(lambda1 = lambda, lambda2 = alpha)
        },
        
        
    )
}

penalty_params_notif <- function(regularization, alpha) {
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
            } else if (alpha < 0 || alpha > 1) {
                cli::cli_abort("'alpha' for elastic-net must be between 0 and 1.")
            }
            
        },
        
        
        SCAD = {
            if (is.null(alpha))  {
                cli::cli_inform("Using default alpha (shape parameter) = 3.7 for SCAD.")
                
            } else if (alpha < 2)  {
                cli::cli_inform("Using default alpha (shape parameter) = 3.7 for SCAD.")
                
            }
        },
        
        
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
    case_indices <- which(status != 0)
    
    # Combine and return
    n_b <- length(sampled_indices)
    n_c <- length(case_indices)
    total_rows <- n_b + n_c
    
    # 1. Pre-allocate the final objects with the correct size and type
    final_time <- numeric(total_rows)
    final_event <- numeric(total_rows)
    final_covs <- matrix(0, nrow = total_rows, ncol = ncol(cov_matrix))
    colnames(final_covs) <- colnames(cov_matrix)
    
    # 2. Fill the first part (base series)
    final_time[1:n_b] <- runif(n_b) * time[sampled_indices]
    final_event[1:n_b] <- 0 # Event is 0 for base series
    final_covs[1:n_b, ] <- cov_matrix[sampled_indices, , drop = FALSE]
    
    # 3. Fill the second part (case series)
    final_time[(n_b + 1):total_rows] <- time[case_indices]
    final_event[(n_b + 1):total_rows] <- status[case_indices]
    final_covs[(n_b + 1):total_rows, ] <- cov_matrix[case_indices, , drop = FALSE]
    
    # 4. Return the pre-allocated and filled list
    list(
        time = final_time,
        event = final_event,
        covariates = final_covs,
        offset = rep(offset, total_rows)
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
fit_cb_model <- function(cb_data,
                         regularization = c('elastic-net', 'SCAD'), 
                         lambda, alpha = NULL,
                         fit_fun = mtool::mtool.MNlogistic,
                         param_start = NULL,
                         n_unpenalized = 2,
                         standardize = TRUE, # Default to TRUE as it's best practice
                         all_event_levels = NULL,
                         ...) {
    
    regularization <- rlang::arg_match(regularization)
    penalty_params <- prepare_penalty_params(regularization, lambda, alpha)
    
    if (is.null(all_event_levels)) all_event_levels <- sort(unique(cb_data$event))
    Y_factor <- factor(cb_data$event, levels = all_event_levels)
    
    # --- Standardization Logic ---
    cb_data$time
    penalized_covs <- cb_data$covariates
    scaler <- NULL
    if (standardize) {
        # Initial calculation of center and scale
        center <- colMeans(penalized_covs, na.rm = TRUE)
        scale <- apply(penalized_covs, 2, sd, na.rm = TRUE)
        
        # Check for zero variance
        if (any(scale == 0, na.rm = TRUE)) {
            cli::cli_warn("Some covariates have zero variance and were not scaled.")
            scale[scale == 0] <- 1
        }
        penalized_covs <- scale(penalized_covs, center = center, scale = scale)
        scaler <- list(center = center, scale = scale)
        
        if(!is.null(param_start)){
            
            original_cov_coefs <- param_start[1:ncol(penalized_covs), , drop = FALSE]
            
            intercept_adjustment <- crossprod(scaler$center, original_cov_coefs)
            param_start[ncol(penalized_covs) + 1, ] <- param_start[ncol(penalized_covs) + 1, ] + intercept_adjustment
            
            param_start_scaled <- sweep(original_cov_coefs, 1, scaler$scale, FUN = "*")
            
            param_start[1:ncol(penalized_covs), ] <- param_start_scaled
        }
        
    }
    
    X <- model.matrix(~., 
                      data = data.frame(cbind(penalized_covs, 
                                              time = log(cb_data$time))))
    
    X <- cbind(X[,-1], X[,1])
    
    # Build design matrix
    # X <- as.matrix(cbind(penalized_covs, time = log(cb_data$time), 1))
    
    opt_args <- list(X = X, Y = Y_factor, offset = cb_data$offset,
                     N_covariates = n_unpenalized, # Hardcoded to 2 for intercept and log(time)
                     regularization = regularization,
                     transpose = FALSE, lambda1 = penalty_params$lambda1,
                     lambda2 = penalty_params$lambda2, lambda3 = 0,
                     param_start = param_start,
                     ...)
    
    fit <- do.call(fit_fun, opt_args)
    
    # --- Coefficient Re-scaling ---
    if (standardize) {
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
                fit[[coef_type]] <- round(coefs_orig, 8)
            }
        }
        # Return the fit (with original-scale coefficients) and the scaler
        return(c(fit, scaler = list(scaler)))
    }
    return(fit)
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
find_lambda_max <- function(cb_data,
                            n_unpenalized,
                            alpha,
                            regularization,
                            ...) {
    
    
    # The number of non-zero coefficients in a null model (intercepts only).
    n_event_types <- length(unique(cb_data$event))
    null_model_coefs <- n_unpenalized * (n_event_types - 1)
    # A wide grid to search for the general location of lambda_max.
    search_grid <- round(exp(seq(log(10), log(0.005), length.out = 10)), 6)
    
    # Main Logic with Progress Bar
    progressr::handlers("cli")
    
    with_progress({
        fine_grid_size <- 5
        
        p <- progressr::progressor(steps = length(search_grid) + fine_grid_size)
        
        # Helper function to fit the model and advance the progress bar.
        # This avoids repeating the fit_cb_model call.
        fit_and_get_count <- function(lambda_val) {
            fit_model <- fit_cb_model(
                cb_data,
                lambda = lambda_val,
                regularization = regularization,
                alpha = alpha,
                n_unpenalized = n_unpenalized,
                ...
            )
            result <- sum(!same(fit_model$coefficients_sparse, 0))
            
            p()
            
            return(result)
        }
        
        # --- First, Coarse Search ---
        cli::cli_alert_info("Searching for lambda_max (coarse grid)...")
        coarse_results <- furrr::future_map_dbl(
            .x = search_grid,
            .f = fit_and_get_count,
            .options = furrr::furrr_options(seed = TRUE)
        )
        
        # --- Bracket the optimal lambda ---
        # Find lambdas that result in a null model (or simpler).
        upper_idx <- which(coarse_results <= null_model_coefs)
        # Find lambdas that result in a more complex model.
        lower_idx <- which(coarse_results > null_model_coefs)
        
        # Handle cases where the grid is too narrow.
        if (length(upper_idx) == 0 || length(lower_idx) == 0) {
            cli::cli_warn("Could not bracket lambda_max with the initial grid. Using largest value.")
            
            p(steps = fine_grid_size)
            
            return(max(search_grid))
        }
        
        # Define the bounds for the finer search.
        upper_bound <- min(search_grid[upper_idx])
        lower_bound <- max(search_grid[lower_idx])
        
        # Finer Search ---
        fine_grid <- round(seq(lower_bound, upper_bound, length.out = fine_grid_size), 6)
        cli::cli_alert_info("Searching for lambda_max (fine grid)...")
        fine_results <- furrr::future_map_dbl(
            .x = fine_grid,
            .f = fit_and_get_count, # Use the same helper function
            .options = furrr::furrr_options(seed = TRUE)
        )
        
        # Identify the first lambda in the fine grid that produces a null model.
        # This is our best estimate for lambda_max.
        first_null_model_idx <- which(fine_results <= null_model_coefs)[1]
        lambda_max <- fine_grid[first_null_model_idx]
        
        # Fallback if the fine search fails for some reason.
        if (is.na(lambda_max)) {
            cli::cli_warn("Fine grid search failed to find a suitable lambda_max. Returning upper bound.")
            return(upper_bound)
        }
    }) # End of with_progress block.
    lambda_max<- lambda_max*1.2
    cli::cli_alert_success("Found lambda_max: {round(lambda_max, 4)}")
    return(lambda_max)
}

#' Generate a Lambda Grid for Regularization
#' @return A numeric vector of lambda values.
create_lambda_grid <- function(cb_data,
                               lambda,
                               nlambda,
                               lambda_max,
                               lambda.min.ratio,
                               regularization,
                               alpha,
                               n_unpenalized,
                               ...){
    
    if(is.null(lambda)){
        
        penalty_params_notif(regularization = regularization,
                             alpha = alpha)
        
        if(is.null(lambda_max)){
            lambda_max <- find_lambda_max(
                cb_data,
                regularization = regularization,
                alpha = alpha,
                n_unpenalized = n_unpenalized,
                ...)
            
            grid <- rev(exp(seq(log(lambda_max * lambda.min.ratio), 
                                log(lambda_max), length.out = nlambda)))
            
        }} else {
            
            grid <- sort(unique(lambda[lambda >= 0]), decreasing = TRUE) 
            
            if(length(grid) == 0) cli::cli_abort("Provided lambda values invalid.")
            
            
        }
    
    cli::cli_alert_info("Using {length(grid)} lambdas. Range: {signif(min(grid), 3)} to {signif(max(grid), 3)}")
    
    return(round(grid, 5))
}

#' Run a Single Fold of Cross-Validation
#' @return A vector of multinomial deviances for the fold.
run_cv_fold <- function(fold_indices, cb_data, 
                        lambdagrid, 
                        all_event_levels,
                        regularization,
                        alpha,
                        n_unpenalized = 2,
                        warm_start = T,
                        update_f = NULL,
                        ...) {
    # 1. Split data into training and validation folds
    train_cv_data <- lapply(cb_data, function(x) if(is.matrix(x)) x[-fold_indices, , drop = FALSE] else x[-fold_indices])
    test_cv_data  <- lapply(cb_data, function(x) if(is.matrix(x)) x[fold_indices, , drop = FALSE] else x[fold_indices])
    
    # 2. Initialize for loop
    deviances <- numeric(length(lambdagrid))
    non_zero <- numeric(length(lambdagrid))
    
    param_start <- NULL # Cold start for the first lambda
    # 3. Iterate over lambda grid with a for loop to enable warm starts
    for (i in seq_along(lambdagrid)) {
        model_info <- fit_cb_model(
            train_cv_data,
            lambda = lambdagrid[i],
            all_event_levels = all_event_levels,
            regularization = regularization,
            alpha = alpha,
            param_start = param_start, # Pass warm start
            n_unpenalized = n_unpenalized,
            ...
        )
        
        # Calculate deviance on the original, unscaled test fold
        deviances[i] <- calc_multinom_deviance(
            test_cv_data,
            model_info, # Contains re-scaled coefficients
            all_event_levels = all_event_levels
        )
        
        non_zero[i] <- sum(!same(model_info$coefficients_sparse, 0))
        
        update_f()
        
        # Update the warm start for the next iteration
        if(warm_start) param_start <- model_info$coefficients
    }
    
    return(list(deviances = deviances,
                non_zero = non_zero))
}


#' Cross-Validation for Penalized Multinomial Case-Base Models
#' @return An object of class `cb.cv` containing the results.
#' @export
cv_cbSCRIP <- function(formula, data, regularization = 'elastic-net',
                       cb_data = NULL,
                       alpha = NULL,
                       lambda = NULL,
                       nfold = 5, nlambda = 50, 
                       ncores = parallel::detectCores() / 2,
                       n_unpenalized = 2,
                       ratio = 50, ratio_event = 1,
                       lambda_max = NULL,
                       lambda.min.ratio = ifelse(nobs < nvars, 0.01, 1e-03),
                       warm_start = T,
                       ...) {
    
    if(is.null(cb_data)){
        cb_data <- create_cb_data(formula, data, ratio = ratio,
                                  ratio_event = ratio_event)
    }
    
    all_event_levels <- sort(unique(cb_data$event))
    
    n_event_types <- length(all_event_levels)
    nobs <- length(cb_data$event)
    nvars <- ncol(cb_data$covariates)
    
    if (ncores > 1) {
        future::plan(multisession, workers = ncores)
    } else {
        future::plan(sequential)
    }
    
    on.exit(future::plan(future::sequential), add = TRUE)
    
    lambdagrid <- create_lambda_grid(cb_data = cb_data,
                                     lambda = lambda,
                                     nlambda = nlambda,
                                     lambda_max = lambda_max,
                                     lambda.min.ratio = lambda.min.ratio,
                                     regularization = regularization,
                                     alpha = alpha,
                                     n_unpenalized = n_unpenalized,
                                     ...)
    
    folds <- caret::createFolds(factor(cb_data$event), k = nfold, list = TRUE)
    cli::cli_alert_info("Starting {nfold}-fold cross-validation...")
    
    # progressr::handlers(global = TRUE)
    progressr::handlers("cli")
    # on.exit(progressr::handlers(global = FALSE), add = TRUE)
    
    with_progress({
        p <- progressr::progressor(steps = nfold * nlambda)
        
        fold_list <- furrr::future_map(
            .x = folds, 
            .f = function(fold) {
                res <- run_cv_fold(
                    fold_indices = fold,
                    cb_data = cb_data,
                    lambdagrid = lambdagrid,
                    n_unpenalized = n_unpenalized,
                    regularization = regularization,
                    alpha = alpha,
                    all_event_levels = all_event_levels,
                    update_f = p,
                    ...
                )
                return(res)
            },
            .options = furrr::furrr_options(seed = TRUE)
        )
    })
    
    deviance_matrix <- lapply(fold_list, function(.x){.x$deviances})
    
    coeffs_list <- lapply(fold_list, function(.x){.x$coeffs})
    
    deviance_matrix <- do.call(cbind, deviance_matrix)
    
    # rownames(deviance_matrix) <- lambdagrid
    
    non_zero_matrix <- lapply(fold_list, function(.x){.x$non_zero})
    
    non_zero_matrix <- do.call(cbind, non_zero_matrix)
    
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
        n_unpenalized = n_unpenalized,
        ...
    )
    
    result <- list(
        fit.min = fit.min,
        lambdagrid = lambdagrid,
        deviance_matrix = deviance_matrix,
        non_zero_matrix = non_zero_matrix,
        deviance_mean = mean_dev,
        deviance_se = se_dev,
        lambda.min = lambda.min,
        lambda.1se = lambda.1se,
        cb_data = cb_data,
        call = match.call()
    )
    
    class(result) <- "cbSCRIP.cv"
    cli::cli_alert_success("Cross-validation complete.")
    return(result)
}


#' Fit a Penalized Multinomial Model over a Regularization Path
#'
#' Fits the case-base model for a sequence of lambda values using warm starts,
#' returning the path of coefficients. Includes a progress bar.
#'
#' @inheritParams cv_cb_model
#' @return An object of class `cb.path` containing coefficient paths.
#' @export
cbSCRIP <- function(formula, data, regularization = 'elastic-net', 
                    cb_data = NULL,
                    alpha = NULL,
                    lambda = NULL,
                    nlambda = 100, n_unpenalized = 2,
                    ratio = 50, ratio_event = 1,
                    lambda_max = NULL,
                    lambda.min.ratio = ifelse(nobs < nvars, 0.01, 1e-03),
                    warm_start = T,
                    ...) {
    
    # 1. Create the full case-base dataset once
    cli::cli_alert_info("Creating case-base dataset...")
    
    if(is.null(cb_data)){
        cb_data <- create_cb_data(formula, data, ratio = ratio,
                                  ratio_event = ratio_event)
    }
    
    all_event_levels <- sort(unique(cb_data$event))
    
    n_event_types <- length(all_event_levels)
    
    nobs <- length(cb_data$event)
    
    nvars <- ncol(cb_data$covariates)
    
    lambdagrid <- create_lambda_grid(cb_data = cb_data,
                                     lambda = lambda,
                                     nlambda = nlambda,
                                     lambda_max = lambda_max,
                                     lambda.min.ratio = lambda.min.ratio,
                                     regularization = regularization,
                                     alpha = alpha,
                                     n_unpenalized = n_unpenalized,
                                     ...)
    nlambda <- length(lambdagrid)
    
    # 3. Fit the model for each lambda value using warm starts
    cli::cli_alert_info("Fitting model path for {nlambda} lambda values...")
    
    path_fits <- vector("list", nlambda)
    
    param_start <- NULL # Cold start for the first lambda
    
    cli::cli_progress_bar("Fitting Path", total = nlambda)
    for (i in seq_along(lambdagrid)) {
        
        model_info <- fit_cb_model(
            cb_data = cb_data,
            lambda = lambdagrid[i],
            regularization = regularization,
            alpha = alpha,
            n_unpenalized = n_unpenalized,
            param_start = param_start, # Pass warm start
            ...
        )
        path_fits[[i]] <- model_info
        
        # Update the warm start for the next iteration
        if(warm_start) param_start <- model_info$coefficients
        
        cli::cli_progress_update()
    }
    # 4. Extract the re-scaled coefficients from each fit
    coefficients <- purrr::map(path_fits, ~.x$coefficients)
    
    names(coefficients) <- lambdagrid
    
    result <- list(
        coefficients = coefficients,
        lambdagrid = lambdagrid,
        models_info = path_fits,
        cb_data = cb_data,
        call = match.call()
    )
    
    class(result) <- "cbSCRIP"
    cli::cli_alert_success("Path fitting complete.")
    return(result)
}