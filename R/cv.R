#' @importFrom VGAM multinomial vglm vglm.control
#' @importFrom stats model.matrix sd as.formula
#' @importFrom cli cli_warn cli_alert_success cli_abort cli_alert_info cli_progress_bar cli_progress_update
#' @importFrom future plan multisession sequential
#' @importFrom caret createFolds
#' @importFrom progressr handlers with_progress progressor
#' @importFrom furrr future_map furrr_options
#' @importFrom purrr map
#' @importFrom rlang arg_match
#' @importFrom methods new
NULL

#' Calculate Multinomial Deviance for a Fitted Case-Base Model
#' @param cb_data A case-base dataset from `create_cb_data`.
#' @param fit_object The output from `fit_cb_model`.
#' @param all_event_levels A vector of all possible event levels to ensure
#'   dimensional consistency.
#' @return The multinomial deviance value.
#' @keywords internal
calc_multinom_deviance <- function(cb_data, fit_object, all_event_levels) {
    # Reconstruct design matrix
    X <- as.matrix(cbind(cb_data$covariates, "log(time)" = log(cb_data$time), "(Intercept)" = 1))
    
    # Get the offset term
    offset <- cb_data$offset
    if (is.null(offset)) {
        stop("`offset` not found in cb_data.")
    }
    
    # Calculate scores for all classes
    scores_events <- (X %*% fit_object$coefficients) + offset
    scores_baseline <- rep(0, nrow(X))
    all_scores <- cbind(scores_baseline, scores_events)
    
    # Log-Sum-Exp Trick
    max_scores <- apply(all_scores, 1, max, na.rm = TRUE)
    
    # Subtract the max score before exponentiating to prevent overflow
    stabilized_scores <- all_scores - max_scores
    exp_scores <- exp(stabilized_scores)
    
    # Calculate probabilities using the stabilized values
    denominator <- rowSums(exp_scores)
    pred_mat <- exp_scores / denominator
    
    # Ensure column order and names are correct
    colnames(pred_mat) <- c(0, 1:(ncol(pred_mat)-1))
    pred_mat <- pred_mat[, as.character(all_event_levels)]
    
    # Create the observed outcome matrix (Y_mat)
    Y_fct <- factor(cb_data$event, levels = all_event_levels)
    Y_mat <- stats::model.matrix(~ Y_fct - 1)
    
    # Calculate deviance with stable probabilities
    VGAM::multinomial()@deviance(mu = pred_mat, y = Y_mat, w = rep(1, nrow(X)))
}



#' Find the Maximum Lambda (lambda_max)
#' @param cb_data Case-base data list.
#' @param n_unpenalized Number of unpenalized predictors.
#' @param alpha The alpha value for elastic net.
#' @return The estimated lambda_max value.
#' @keywords internal
find_lambda_max <- function(cb_data, n_unpenalized, alpha) {
    
    # Handle alpha = 0 (Ridge)
    if (alpha <= 1e-10) { # Use tolerance for floating point
        cli::cli_warn("lambda_max is undefined or infinite for alpha = 0 (Ridge).")
    }
    
    # Data Components
    Y_factor <- as.factor(cb_data$event)
    offsets <- cb_data$offset
    penalized_covs_raw <- as.matrix(cb_data$covariates)
    
    n <- nrow(penalized_covs_raw) # Number of observations
    K_total <- nlevels(Y_factor) # Total number of classes (e.g., 3 for 0, 1, 2)
    K_params <- K_total - 1     # Number of parameter columns (non-baseline)
    
    # Calculate Null Probabilities
    
    eta_null_params <- matrix(0, nrow = n, ncol = K_params)
    # Add offset
    eta_null_params[] <- offsets
    
    # Softmax calculation
    exp_eta_null <- exp(eta_null_params)
    # Add baseline class contribution (exp(0) = 1)
    denom <- 1 + rowSums(exp_eta_null)
    
    P_null <- matrix(0, nrow = n, ncol = K_total)
    P_null[, 1] <- 1 / denom # Baseline probability (assuming class 0 or first factor level)
    if (K_params > 0) {
        for(k in 1:K_params) {
            P_null[, k + 1] <- exp_eta_null[, k] / denom
        }
    }
    
    # Calculate Null Model Residuals
    Y_matrix <- stats::model.matrix(~ 0 + Y_factor) # n x K_total one-hot matrix
    Residuals <- Y_matrix - P_null          # n x K_total residual matrix
    
    # Build the Standardized Design Matrix
    
    # Standardize the original covariates
    center <- colMeans(penalized_covs_raw, na.rm = TRUE)
    scale <- apply(penalized_covs_raw, 2, stats::sd, na.rm = TRUE)
    
    # Check for zero variance and replace scale with 1 to avoid NaN/Inf
    zero_var_idx <- which(scale < 1e-10)
    if (length(zero_var_idx) > 0) {
        cli::cli_warn("Some covariates have near-zero variance and were not scaled.")
        scale[zero_var_idx] <- 1
    }
    
    penalized_covs_scaled <- scale(penalized_covs_raw, center = center, scale = scale)
    
    # Build the full design matrix using model.matrix, exactly like the wrapper
    design_data <- data.frame(penalized_covs_scaled,
                              time = log(cb_data$time))
    X_intermediate <- stats::model.matrix(~., data = design_data)
    
    # Re-order to match wrapper: Intercept column goes last
    X_full_design <- cbind(X_intermediate[, -1, drop = FALSE], X_intermediate[, 1, drop = FALSE])
    colnames(X_full_design)[ncol(X_full_design)] <- "(Intercept)" # Rename intercept column
    
    
    # Calculate Gradient for PENALIZED Variables
    
    total_predictors <- ncol(X_full_design)
    n_penalized <- total_predictors - n_unpenalized
    
    if (n_penalized <= 0) {
        cli::cli_warn("No variables are penalized based on n_unpenalized = {n_unpenalized}. lambda_max is 0.")
        return(0)
    }
    
    # The penalized variables are the FIRST n_penalized columns of X_full_design
    penalized_idx <- 1:n_penalized
    X_penalized <- X_full_design[, penalized_idx, drop = FALSE]
    
    # Gradient = X_penalized^T * Residuals
    # We only need the gradient w.r.t the K_params (non-baseline) parameter sets
    # Residuals[, -1] selects columns for classes 1, 2, ... K_params
    Gradient_matrix <- t(X_penalized) %*% Residuals[, -1, drop = FALSE]
    
    # Find Max Absolute Gradient Component and Scale
    
    # Find the largest absolute value across all elements of the gradient matrix
    max_abs_grad <- max(abs(Gradient_matrix))
    
    # Calculate final lambda_max using the standard formula
    # Note: glmnet uses n=nrow(X), which matches 'n' here.
    lambda_max <- max_abs_grad / (alpha * n)
    
    cli::cli_alert_success("Calculated lambda_max: {round(lambda_max, 4)}")
    
    return(lambda_max)
}


#' Generate a Lambda Grid for Regularization
#' @param cb_data Case-base data.
#' @param lambda A user-supplied lambda sequence.
#' @param nlambda Number of lambda values.
#' @param lambda_max The maximum lambda value.
#' @param lambda.min.ratio Ratio for smallest lambda.
#' @param regularization Type of regularization.
#' @param alpha The alpha value.
#' @param n_unpenalized Number of unpenalized predictors.
#' @param ... Additional arguments.
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
    
    if (is.null(lambda.min.ratio)) lambda.min.ratio <- ifelse(nobs < nvars, 0.01, 1e-03)
    
    if(is.null(lambda)){
        
        # penalty_params_notif(regularization = regularization,
        #                      alpha = alpha)
        
        if (is.null(alpha)) {
            alpha <- 0.5
        }
        
        if(is.null(lambda_max)){
            lambda_max <- find_lambda_max(
                cb_data,
                # regularization = regularization,
                alpha = alpha,
                n_unpenalized = n_unpenalized
                # ...
            )
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
#' @param fold_indices Indices for the validation fold.
#' @param cb_data Case-base data.
#' @param lambdagrid Vector of lambda values.
#' @param all_event_levels Vector of event levels.
#' @param regularization Type of regularization.
#' @param alpha The alpha value.
#' @param n_unpenalized Number of unpenalized predictors.
#' @param warm_start Logical, whether to use warm starts.
#' @param update_f A progress update function.
#' @param ... Additional arguments for `fit_cb_model`.
#' @return A list of multinomial deviances and non-zero counts for the fold.
#' @export
run_cv_fold <- function(fold_indices, cb_data,
                        lambdagrid,
                        all_event_levels,
                        regularization,
                        alpha,
                        n_unpenalized = 2,
                        warm_start = TRUE,
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
        
        opt_args <- list(train_cv_data,
                         lambda = lambdagrid[i],
                         all_event_levels = all_event_levels,
                         regularization = regularization,
                         alpha = alpha,
                         param_start = param_start, # Pass warm start
                         n_unpenalized = n_unpenalized,
                         ...)
        
        # if(is.null(opt_args$lr_adj)) opt_args$lr_adj <- 100
        
        model_info <- do.call(fit_cb_model, opt_args)
        
        # Calculate deviance on the original, unscaled test fold
        deviances[i] <- calc_multinom_deviance(
            test_cv_data,
            model_info, # Contains re-scaled coefficients
            all_event_levels = all_event_levels
        )
        
        non_zero[i] <- sum(!same(rowSums(abs(model_info$coefficients)), 0))
        
        if (!is.null(update_f)) update_f()
        
        # Update the warm start for the next iteration
        if(warm_start) param_start <- model_info$coefficients
    }
    
    return(list(deviances = deviances,
                non_zero = non_zero))
}


#' Cross-Validation for Penalized Multinomial Case-Base Models
#' @param formula A formula object.
#' @param data The input data.frame.
#' @param regularization Type of regularization.
#' @param cb_data A case-base data list.
#' @param alpha The alpha value.
#' @param lambda A vector of lambda values.
#' @param nfold Number of CV folds.
#' @param nlambda Number of lambda values.
#' @param ncores Number of cores for parallel execution.
#' @param n_unpenalized Number of unpenalized predictors.
#' @param ratio Sampling ratio.
#' @param ratio_event Event ratio.
#' @param lambda_max Max lambda.
#' @param lambda.min.ratio Min lambda ratio.
#' @param warm_start Logical, whether to use warm starts.
#' @param ... Additional arguments.
#' @return An object of class `cb.cv` containing the results.
#' @export
cv_cbSCRIP <- function(formula, data, regularization = 'elastic-net',
                       cb_data = NULL,
                       alpha = NULL,
                       lambda = NULL,
                       nfold = 5, nlambda = 50,
                       ncores = parallel::detectCores() / 2,
                       n_unpenalized = 2,
                       ratio = 50, ratio_event = "all",
                       lambda_max = NULL,
                       lambda.min.ratio = NULL,
                       warm_start = TRUE,
                       ...) {
    
    if(is.null(cb_data)){
        cb_data <- create_cb_data(formula, data, ratio = ratio,
                                  ratio_event = ratio_event)
    }
    
    all_event_levels <- sort(unique(cb_data$event))
    
    n_event_types <- length(all_event_levels)
    
    
    old_plan <- future::plan()
    if (ncores > 1) {
        future::plan(future::multisession, workers = ncores)
    } else {
        future::plan(future::sequential)
    }
    
    on.exit(future::plan(old_plan), add = TRUE)
    
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
#' @param object A fitted object of class `cbSCRIP.path`.
#' @param ... Additional arguments (currently unused).
#'
#' @return An object of class `cbSCRIP.path`, where the `coefficients` list
#'   contains the refitted coefficients. A new element, `refitted_models`, is
#'   also added, containing the full model objects from each refit.
#'
#' @importFrom stats as.formula coef
#' @importFrom purrr map
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
                                           "+ log(time) + offset(offset)"
                )
            }
            
            # This line seems problematic as it modifies cb_data in a loop
            # cb_data$covariates <- cb_data$covariates[, selected_vars_names, drop = FALSE]
            
            # Refit the model using VGAM::vglm
            refitted_model <- tryCatch({
                
                model <- VGAM::vglm(stats::as.formula(refit_formula_str),
                                    data = refit_data,
                                    family = VGAM::multinomial(refLevel = 1),
                                    control = VGAM::vglm.control(step = 0.5,
                                                                 maxit = 40))
                
                typeEvents <- sort(unique(refit_data[["status"]]))
                
                methods::new("CompRisk", model,
                             originalData = data.frame(),
                             typeEvents = typeEvents,
                             timeVar = "time",
                             eventVar = "status"
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
            refit_coefs <- stats::coef(model) # This is the named vector
            
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
    
    object$adjusted <- TRUE
    
    return(object)
}

#' Fit a Penalized Multinomial Model over a Regularization Path
#'
#' Fits the case-base model for a sequence of lambda values using warm starts,
#' returning the path of coefficients. Includes a progress bar.
#'
#' @inheritParams cv_cbSCRIP
#' @param formula A formula object.
#' @param data The input data.frame.
#' @param cb_data A case-base data list.
#' @param lambda A vector of lambda values.
#' @param alpha The alpha value.
#' @param nlambda Number of lambda values.
#' @param n_unpenalized Number of unpenalized predictors.
#' @param ratio Sampling ratio.
#' @param ratio_event Event ratio.
#' @param lambda_max Max lambda.
#' @param lambda.min.ratio Min lambda ratio.
#' @param warm_start Logical, whether to use warm starts.
#' @param coeffs Character, whether to return "adjusted" (refitted) or "original" coefficients.
#' @param ... Additional arguments.
#' @return An object of class `cbSCRIP.path` or `cbSCRIP`.
#' @export
cbSCRIP <- function(formula, data, regularization = 'elastic-net',
                    cb_data = NULL,
                    lambda = NULL,
                    alpha = NULL,
                    nlambda = 50, n_unpenalized = 2,
                    ratio = 50, ratio_event = "all",
                    lambda_max = NULL,
                    lambda.min.ratio = NULL,
                    warm_start = TRUE,
                    coeffs = c("adjusted", "original"),
                    ...) {
    
    coeffs <- rlang::arg_match(coeffs)
    
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
        
        non_zero <- purrr::map(path_fits, ~sum(!same(rowSums(abs(.x$coefficients)), 0)))
        
        result <- list(
            coefficients = coefficients,
            non_zero = non_zero,
            lambdagrid = lambdagrid,
            models_info = path_fits,
            cb_data = cb_data,
            call = match.call()
        )
        
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
        
        result$cb_data <- cb_data
        
        if (coeffs == "adjusted") {
            
            result$coefficients <- list(result$coefficients)
            
            result <- refit_cbSCRIP(result)
            
            result$coefficients <- result$coefficients[[1]]
            
            result$refitted_models <- result$refitted_models[[1]]
            
            result$adjusted <- TRUE
            
            
        }
        
        
        result$call <- match.call()
        
        class(result) <- "cbSCRIP"
    }
    
    return(result)
}
