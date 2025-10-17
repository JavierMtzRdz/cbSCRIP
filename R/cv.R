#' Calculate Multinomial Deviance for a Fitted Case-Base Model
#' @param cb_data A case-base dataset from `create_cb_data`.
#' @param fit_object The output from `fit_cb_model`.
#' @param all_event_levels A vector of all possible event levels to ensure
#'   dimensional consistency.
#' @return The multinomial deviance value.
calc_multinom_deviance <- function(cb_data, fit_object, all_event_levels) {
    # 1. Reconstruct the design matrix (ensure this matches the fitting process)
    X <- as.matrix(cbind(cb_data$covariates, time = log(cb_data$time), 1))
    
    # 2. Get the essential offset term
    offset <- cb_data$offset
    if (is.null(offset)) {
        stop("`offset` not found in cb_data. It is essential for deviance calculation.")
    }
    
    # 3. Calculate scores for event classes using the GLM offset convention
    fitted_vals <- X %*% fit_object$coefficients
    scores_events <- fitted_vals + offset
    
    # 4. Manually compute the softmax probabilities, including the baseline
    # The baseline score is 0, so its exponential is exp(0) = 1
    exp_scores_events <- exp(scores_events)
    denominator <- 1 + rowSums(exp_scores_events)
    
    # Probabilities of the event classes
    prob_events <- exp_scores_events / denominator
    # Probability of the baseline class (event = 0)
    prob_baseline <- 1 / denominator
    
    # 5. Combine into the final predicted probability matrix `mu`
    # Ensure the column order matches `all_event_levels`
    pred_mat <- cbind(prob_baseline, prob_events)
    # Assuming event levels are 0, 1, 2...
    colnames(pred_mat) <- c(0, 1:ncol(prob_events))
    pred_mat <- pred_mat[, as.character(all_event_levels)]
    
    # 6. Create the observed outcome matrix (Y_mat)
    Y_fct <- factor(cb_data$event, levels = all_event_levels)
    Y_mat <- model.matrix(~ Y_fct - 1)
    
    # 7. Calculate deviance with the CORRECT predicted probabilities
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
    search_grid <- round(exp(seq(log(7), log(0.005), length.out = 5)), 8)
    
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
            result <- sum(!same(fit_model$coefficients, 0))
            
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
        fine_grid <- round(seq(lower_bound, upper_bound, length.out = fine_grid_size), 8)
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
#' @export
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
    
    if(length(grid) > 1) cli::cli_alert_info("Using {length(grid)} lambdas. Range: {signif(min(grid), 3)} to {signif(max(grid), 3)}")
    
    return(round(grid, 8))
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
        
        non_zero[i] <- sum(!same(model_info$coefficients, 0))
        
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

#' Print a cbSCRIP.cv object
#'
#' @param x An object of class `cbSCRIP.cv`, the result of `cv_cbSCRIP`.
#' @param ... Additional arguments passed to other methods.
#'
#' @return The original object `x`, invisibly.
#' @export
print.cbSCRIP.cv <- function(x, ...) {
    # Header for the output
    cat("--- Cross-Validated Case-Base Competing Risks Model ---\n\n")
    
    # Print the function call
    cat("Call:\n")
    print(x$call)
    cat("\n")
    
    # Determine number of folds and lambda values
    n_folds <- ncol(x$deviance_matrix)
    n_lambda <- length(x$lambdagrid)
    cat(paste0("Performed ", n_folds, "-fold cross-validation over ",
               n_lambda, " lambda values.\n\n"))
    
    # Get the index of the minimum deviance
    idx_min <- which.min(x$deviance_mean)
    
    # Report the optimal lambda values
    cat("Optimal Lambda Values:\n")
    cat(sprintf("  Lambda with minimum deviance (lambda.min): %.4f\n", x$lambda.min))
    cat(sprintf("  Largest lambda within 1 SE of min (lambda.1se): %.4f\n\n", x$lambda.1se))
    
    # Report the number of non-zero coefficients for the final fitted model
    n_nonzero <- sum(x$fit.min$coefficients != 0)
    cat(paste0("The final model (fit.min) was fit using lambda.min and has ",
               n_nonzero, " non-zero coefficients.\n"))
    
    # Return the original object invisibly
    invisible(x)
}

#' Print a cbSCRIP.path object
#'
#' @param x An object of class `cbSCRIP.path`, the result of `path_cbSCRIP`.
#' @param ... Additional arguments passed to other methods.
#'
#' @return The original object `x`, invisibly.
#' @export
print.cbSCRIP.path <- function(x, ...) {
    # Header for the output
    cat("--- Case-Base Competing Risks Model Path ---\n\n")
    
    # Print the function call
    cat("Call:\n")
    print(x$call)
    cat("\n")
    
    # Summary of the regularization path
    n_lambda <- length(x$lambdagrid)
    cat(paste0("Regularization path fit for ", n_lambda, " lambda values.\n"))
    
    # Get the number of non-zero coefficients at the first and last lambda
    n_nonzero_start <- x$non_zero[[1]]
    n_nonzero_end <- x$non_zero[[n_lambda]]
    
    cat(paste0("Number of non-zero coefficients: from ",
               n_nonzero_start, " to ", n_nonzero_end, ".\n"))
    
    # Return the original object invisibly
    invisible(x)
}

#' Refit Models Along a Regularization Path
#'
#' For each lambda in a `cbSCRIP.path` object, this function identifies the
#' selected variables and refits a standard (unpenalized) `casebase` model
#' using only that subset.
#'
#' @param object A fitted object of class
#' @param coeffs A character string specifying the type of coefficients to return.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `cbSCRIP.path`, where the `coefficients` list
#'   contains the refitted coefficients. A new element, `refitted_models`, is
#'   also added, containing the full model objects from each refit.
#'
#' @importFrom stats as.formula coef
#' @importFrom purrr map map2
#' @importFrom cli cli_alert_info cli_progress_bar cli_progress_update
#' @export
refit_cbSCRIP <- function(object, ...) {
    
    cli::cli_alert_info("Refitting models for each lambda using only selected variables...")
    # Extract necessary components from the original object
    cb_data <- object$cb_data
    original_call <- object$call
    
    # Get the original ratio used for fitting
    ratio <- ifelse(!is.null(original_call$ratio), eval(original_call$ratio), 50)
    
    # Set up a progress bar
    cli::cli_progress_bar("Refitting Path", total = length(object$lambdagrid))
    
    # Iterate over each set of coefficients in the path to refit the models
    progressr::with_progress({
        # Define the progressor
        p <- progressr::progressor(steps = length(object$lambdagrid))
        refitted_models <- purrr::map(object$coefficients, function(coef_matrix) {
            
            # Identify variables with non-zero coefficients for any cause
            selected_vars_logical <- (rowSums(abs(coef_matrix)) > 1e-10) & (rownames(coef_matrix) %in% colnames(cb_data$covariates))
            selected_vars_names <- rownames(coef_matrix)[selected_vars_logical]
            
            # Handle the case where no variables are selected (intercept-only model)
            if (length(selected_vars_names) == 0) {
                refit_formula_str <- "status ~ 1 + log(time)"
                # Data still needs time and status columns for fitSmoothHazard
                refit_data <- data.frame(time = cb_data$time, status = cb_data$event)
            } else {
                # Prepare data with only the selected covariates
                refit_data <- as.data.frame(cb_data$covariates[, selected_vars_names, drop = FALSE])
                refit_data[["time"]] <- cb_data$time
                refit_data[["status"]] <- cb_data$event
                refit_data[["offset"]] <- cb_data$offset
                
                class(refit_data) <- c("cbData", class(refit_data))
                
                # Create the new formula for refitting
                refit_formula_str <- paste("status ~",
                                           paste(make.names(selected_vars_names), collapse = " + "),
                                           "+ log(time)"
                )
            }
            
            # Refit the model using casebase::fitSmoothHazard
            refitted_model <- tryCatch({
                casebase::fitSmoothHazard(
                    formula = as.formula(refit_formula_str),
                    data = refit_data,
                    time = "time",
                    ratio = ratio
                )
            }, error = function(e) {
                warning(paste("Refitting failed for one lambda value:", e$message))
                return(NULL)
            })
            
            return(refitted_model)
        })
    })
    
    # Reconstruct the coefficient list in the original format
    # Get dimensions and names from the first original coefficient matrix
    orig_coef_template <- object$coefficients[[1]]
    p_orig <- nrow(orig_coef_template)
    K_orig <- ncol(orig_coef_template)
    
    refitted_coefficients <- purrr::map(refitted_models, function(model) {
        
        # Create an empty matrix with the original dimensions and names
        full_coef_mat <- matrix(0, nrow = p_orig, ncol = K_orig,
                                dimnames = list(rownames(orig_coef_template),
                                                colnames(orig_coef_template)))
        if (!is.null(model)) {
            refit_coefs <- coef(model) # This is the named vector
            
            # Iterate through the named vector to parse names and place coefficients
            for (i in seq_along(refit_coefs)) {
                coef_name <- names(refit_coefs)[i]
                coef_val <- refit_coefs[i]
                
                # Split "X1:1" into "X1" and "1"
                parts <- strsplit(coef_name, ":")[[1]]
                var_name <- parts[1]
                cause_idx <- as.integer(parts[2])
                
                # Check if the variable and cause exist in our template matrix
                if (var_name %in% rownames(full_coef_mat) && cause_idx <= K_orig) {
                    full_coef_mat[var_name, cause_idx] <- coef_val
                }
            }
        }
        return(full_coef_mat)
    })
    
    # Update the object with the new, adjusted coefficients and the refitted models
    object$coefficients <- refitted_coefficients
    object$refitted_models <- refitted_models
    
    return(object)
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
                    coeffs = c("adjusted", "original"),
                    ...) {
    
    # Create the full case-base dataset 
    
    if(is.null(cb_data)){
        
        cli::cli_alert_info("Creating case-base dataset...")
        
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
    
    if (nlambda > 1) {
        
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
        
        coeffs <- rlang::arg_match(coeffs)
        
        if (coeffs == "adjusted") {
            result <- refit_cbSCRIP(result)
        }
        
        class(result) <- "cbSCRIP.path"
        cli::cli_alert_success("Path fitting complete.")
        
    } else {
        result <- fit_cb_model(
            cb_data = cb_data,
            lambda = lambdagrid,
            regularization = regularization,
            alpha = alpha,
            n_unpenalized = n_unpenalized,
            ...
        )
        
        result$coefficients <- list(result$coefficients)
        
        result$cb_data <- cb_data
        
        result <- refit_cbSCRIP(result)
        
        result$coefficients <- result$coefficients[[1]]
        
        result$refitted_models <- result$refitted_models[[1]]
        
    }
    
    return(result)
}