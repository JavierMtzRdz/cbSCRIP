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



#' Find the Maximum Lambda (lambda_max)
#' @param ... Other arguments passed to find_lambda_max.
#' @return The estimated lambda_max value.
find_lambda_max <- function(cb_data,
                            n_unpenalized,
                            alpha,
                            regularization,
                            ...) {
    
    n_event_types <- length(unique(cb_data$event))
    null_model_coefs <- n_unpenalized * (n_event_types - 1)
    search_grid <- round(exp(seq(log(7), log(0.005), length.out = 5)), 6)
    
    progressr::handlers("cli")
    
    progressr::with_progress({
        fine_grid_size <- 5
        
        p <- progressr::progressor(steps = length(search_grid) + fine_grid_size)
        
        # Helper function to fit the model and advance the progress bar
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
        
        # Coarse Search
        cli::cli_alert_info("Searching for lambda_max (coarse grid)...")
        coarse_results <- furrr::future_map_dbl(
            .x = search_grid,
            .f = fit_and_get_count,
            .options = furrr::furrr_options(
                globals = TRUE,
                seed = TRUE
            )
        )
        
        upper_idx <- which(coarse_results <= null_model_coefs)
        lower_idx <- which(coarse_results > null_model_coefs)
        
        # grid is too narrow.
        if (length(upper_idx) == 0 || length(lower_idx) == 0) {
            cli::cli_warn("Could not bracket lambda_max with the initial grid. Using largest value.")
            
            p(steps = fine_grid_size)
            
            return(max(search_grid))
        }
        
        #  bounds for the finer search.
        upper_bound <- min(search_grid[upper_idx])
        lower_bound <- max(search_grid[lower_idx])
        
        # Finer Search
        fine_grid <- round(seq(lower_bound, upper_bound, length.out = fine_grid_size), 6)
        cli::cli_alert_info("Searching for lambda_max (fine grid)...")
        fine_results <- furrr::future_map_dbl(
            .x = fine_grid,
            .f = fit_and_get_count,
            .options = furrr::furrr_options(
                globals = TRUE,
                seed = TRUE
            )
        )
        
        
        # This is our best estimate for lambda_max.
        first_null_model_idx <- which(fine_results <= null_model_coefs)[1]
        lambda_max <- fine_grid[first_null_model_idx]
        
        # Fallback if the fine search fails for some reason.
        if (is.na(lambda_max)) {
            cli::cli_warn("Fine grid search failed to find a suitable lambda_max. Returning upper bound.")
            return(upper_bound)
        }
    }) 
    
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
    
    nobs <- length(cb_data$event)
    
    nvars <- ncol(cb_data$covariates)
    
    if (is.null(lambda.min.ratio)) lambda.min.ratio <- ifelse(nobs < nvars, 0.01, 5e-04)
    
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
            }
            
            grid <- rev(exp(seq(log(lambda_max * lambda.min.ratio), 
                                log(lambda_max), length.out = nlambda)))
            
        } else {
            
            grid <- sort(unique(lambda[lambda >= 0]), decreasing = TRUE) 
            
            if(length(grid) == 0) cli::cli_abort("Provided lambda values invalid.")
            
            
        }
    
    cli::cli_alert_info("Using {length(grid)} lambdas. Range: {signif(min(grid), 3)} to {signif(max(grid), 3)}")
    
    return(round(grid, 5))
}

#' Run a Single Fold of Cross-Validation
#' @return A vector of multinomial deviances for the fold.
#' @export
run_cv_fold <- function(fold_indices, cb_data, 
                        lambdagrid, 
                        all_event_levels,
                        regularization,
                        alpha,
                        n_unpenalized = 2,
                        warm_start = T,
                        update_f = NULL,
                        ...) {
    # Split data into training and validation folds
    train_cv_data <- lapply(cb_data, function(x) if(is.matrix(x)) x[-fold_indices, , drop = FALSE] else x[-fold_indices])
    test_cv_data  <- lapply(cb_data, function(x) if(is.matrix(x)) x[fold_indices, , drop = FALSE] else x[fold_indices])
    
    # Initialize for loop
    deviances <- numeric(length(lambdagrid))
    non_zero <- numeric(length(lambdagrid))
    
    param_start <- NULL 
    
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
                       ratio = 25, ratio_event = 1,
                       lambda_max = NULL,
                       lambda.min.ratio = NULL,
                       warm_start = T,
                       ...) {
    
    if(is.null(cb_data)){
        cb_data <- create_cb_data(formula, data, ratio = ratio,
                                  ratio_event = ratio_event)
    }
    
    all_event_levels <- sort(unique(cb_data$event))
    
    n_event_types <- length(all_event_levels)
    
    
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
    
    progressr::handlers("cli")
    
    progressr::with_progress({
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
            .options = furrr::furrr_options(
                globals = TRUE,
                seed = TRUE
            )
        )
    })
    
    deviance_matrix <- lapply(fold_list, function(.x){.x$deviances})
    
    coeffs_list <- lapply(fold_list, function(.x){.x$coeffs})
    
    deviance_matrix <- do.call(cbind, deviance_matrix)
    
    # rownames(deviance_matrix) <- lambdagrid
    
    non_zero_matrix <- lapply(fold_list, function(.x){.x$non_zero})
    
    non_zero_matrix <- do.call(cbind, non_zero_matrix)
    
    mean_dev <- rowMeans(deviance_matrix)
    
    se_dev <- apply(deviance_matrix, 1, stats::sd) / sqrt(nfold)
    
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
#' @inheritParams cv_cbSCRIP
#' @return An object of class `cb.path` containing coefficient paths.
#' @export
cbSCRIP <- function(formula, data, regularization = 'elastic-net', 
                    cb_data = NULL,
                    alpha = NULL,
                    lambda = NULL,
                    nlambda = 100, n_unpenalized = 2,
                    ratio = 25, ratio_event = 1,
                    lambda_max = NULL,
                    lambda.min.ratio = NULL,
                    warm_start = T,
                    ...) {
    
    # Create the full case-base dataset 
    cli::cli_alert_info("Creating case-base dataset...")
    
    if(is.null(cb_data)){
        
        cb_data <- create_cb_data(formula, data, ratio = ratio,
                                  ratio_event = ratio_event)
    }
    
    all_event_levels <- sort(unique(cb_data$event))
    
    n_event_types <- length(all_event_levels)
    
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
    
    
    cli::cli_alert_info("Fitting model path for {nlambda} lambda values...")
    
    path_fits <- vector("list", nlambda)
    
    param_start <- NULL #
    
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
    # Extract the re-scaled coefficients 
    coefficients <- purrr::map(path_fits, ~.x$coefficients)
    
    non_zero <- purrr::map(path_fits, ~sum(!same(.x$coefficients, 0)))
    
    result <- list(
        coefficients = coefficients,
        non_zero = non_zero,
        lambdagrid = lambdagrid,
        models_info = path_fits,
        cb_data = cb_data,
        call = match.call()
    )
    
    class(result) <- "cbSCRIP"
    cli::cli_alert_success("Path fitting complete.")
    return(result)
}