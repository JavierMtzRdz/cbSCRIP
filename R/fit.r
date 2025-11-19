#' Prepare Penalty Parameters for the Fitting Function
#'
#' A helper function to calculate the lambda parameters required by the cbSCRIP
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
                alpha <- 0.5
            }
            
            list(lambda1 = lambda * alpha, 
                 lambda2 = 0.5 * lambda * (1 - alpha),
                 alpha = alpha)
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

#' Notifying lambda path
#' @return Notification
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
create_cb_data <- function(formula, data, ratio = 20, ratio_event = "all") {
    
    data <- data.frame(data)
    
    response_vars <- all.vars(formula[[2]])
    
    if (length(response_vars) != 2) {
        cli::cli_abort("LHS of formula must be Surv(time, event).")
    }
    time_var <- response_vars[1]
    status_var <- response_vars[2]
    
    if (!all(c(time_var, status_var) %in% names(data))) {
        cli::cli_abort("Time and/or status variables from formula not found in data.")
    }
    
    # time <- data[[time_var]]
    # status <- data[[status_var]]
    time <- data[,time_var]
    
    status <- data[,status_var]
    
    cov_matrix <- stats::model.matrix(stats::as.formula(formula[-2]),
                                      data[!(names(data) %in% 
                                                 c(status_var,
                                                   time_var))])[, -1, 
                                                                drop = FALSE]
    
    # Determine which events to use for the base series ratio
    
    all_event_types <- unique(status[status != 0])
    if (identical(ratio_event, "all")) {
        event_types_for_cases <- unique(status[status != 0])
    } else if (is.numeric(ratio_event) && all(ratio_event %in% all_event_types)) {
        event_types_for_cases <- ratio_event
        
    } else {
        cli::cli_abort("'ratio_event' must be 'all' or a numeric vector of valid event types.")
    }
    
    case_count <- length(which(status %in% event_types_for_cases))
    
    if (case_count == 0) cli::cli_abort("No events found for the specified 'ratio_event'.")
    
    # Create Base Series
    n <- nrow(data)
    B <- sum(time)
    b_size <- ratio * case_count
    offset <- log(B / b_size)
    
    cli::cli_alert_info("Created base series with {b_size} samples based on {case_count} event(s).")
    
    prob_select <- time / B
    sampled_indices <- sample(n, size = b_size, 
                              replace = TRUE, prob = prob_select)
    
    
    case_indices <- which(status %in% all_event_types)
    
    # Combine and return
    n_b <- length(sampled_indices)
    n_c <- length(case_indices)
    total_rows <- n_b + n_c
    
    # Pre-allocate the final objects 
    final_time <- numeric(total_rows)
    final_event <- numeric(total_rows)
    final_covs <- matrix(0, nrow = total_rows, ncol = ncol(cov_matrix))
    colnames(final_covs) <- colnames(cov_matrix)
    
    # base series
    final_time[1:n_b] <- runif(n_b) * time[sampled_indices]
    final_event[1:n_b] <- 0 
    final_covs[1:n_b, ] <- cov_matrix[sampled_indices, , drop = FALSE]
    
    # case series
    final_time[(n_b + 1):total_rows] <- time[case_indices]
    final_event[(n_b + 1):total_rows] <- status[case_indices]
    final_covs[(n_b + 1):total_rows, ] <- cov_matrix[case_indices, , drop = FALSE]
    
    # Pre-allocated and filled list
    list(
        time = final_time,
        event = final_event,
        covariates = final_covs,
        offset = rep(offset, total_rows)
    )
}

#' Internal Function to Fit a Regularization Path
#'
#' Standardizes data once and fits the model for a sequence of lambdas using warm starts.
#'
#' @param cb_data Case-base data list.
#' @param lambdas Vector of lambda values.
#' @param regularization Regularization type.
#' @param alpha Alpha value.
#' @param n_unpenalized Number of unpenalized covariates.
#' @param fit_fun Fitting function.
#' @param param_start Initial parameters.
#' @param standardize Whether to standardize.
#' @param all_event_levels Event levels.
#' @param ... Additional arguments.
#' @return A list of model fits.
#' @keywords internal
#' Standardize Case-Base Data
#'
#' Centers and scales the covariates in a case-base dataset.
#'
#' @param cb_data A case-base data list.
#' @return A list containing the standardized `cb_data` and the `scaler` object.
#' @export
standardize_cb_data <- function(cb_data) {
    penalized_covs <- as.matrix(cb_data$covariates)
    p_covs <- ncol(penalized_covs)
    
    center <- colMeans(penalized_covs, na.rm = TRUE)
    n <- nrow(penalized_covs)
    scale <- apply(penalized_covs, 2, sd, na.rm = TRUE) * sqrt((n - 1) / n)
    
    zero_var <- scale < .Machine$double.eps
    if (any(zero_var)) {
        scale[zero_var] <- 1
    }
    
    penalized_covs_scaled <- scale(penalized_covs, center = center, scale = scale)
    
    cb_data$covariates <- penalized_covs_scaled
    
    scaler <- list(center = center, scale = scale)
    
    return(list(cb_data = cb_data, scaler = scaler))
}

#' Unstandardize Coefficients
#'
#' Transforms coefficients from the standardized scale back to the original scale.
#'
#' @param coefficients A matrix or vector of standardized coefficients.
#' @param scaler A list with `center` and `scale` components.
#' @param p_covs Number of penalized covariates.
#' @param col_names Original column names for the covariates.
#' @return The unstandardized coefficients.
#' @export
unstandardize_coefficients <- function(coefficients, scaler, col_names) {
    if (is.null(coefficients)) return(NULL)
    
    p_covs <- length(scaler$scale)
    
    coefs_scaled <- coefficients
    
    beta_covs_scaled <- coefs_scaled[1:p_covs, , drop = FALSE]
    time_coef <- coefs_scaled[p_covs + 1, , drop = FALSE]
    intercept_scaled <- coefs_scaled[p_covs + 2, , drop = FALSE]
    
    beta_covs_orig <- beta_covs_scaled / scaler$scale
    intercept_adj <- colSums(beta_covs_scaled * (scaler$center / scaler$scale))
    intercept_orig <- intercept_scaled - intercept_adj
    
    coefs_orig <- rbind(beta_covs_orig, time_coef, intercept_orig)
    rownames(coefs_orig) <- c(col_names, "log(time)", "(Intercept)")
    
    return(coefs_orig)
}

#' Fit a Penalized Multinomial Model on Case-Base Data
#'
#' A wrapper function to fit a penalized model (Elastic-Net or SCAD) using
#' an optimizer from the `cbSCRIP` package.
#'
#' @param cb_data A case-base dataset from `create_cb_data`.
#' @param regularization A string, either 'elastic-net' or 'SCAD'.
#' @param lambda The primary shrinkage parameter.
#' @param alpha The mixing/shape parameter. See `prepare_penalty_params`.
#' @param unpen_cov Integer. The number of leading covariates to leave unpenalized.
#' @param fit_fun The fitting function from `cbSCRIP` to use.
#' @param param_start Optional starting values for the coefficients.
#' @param standardize Logical. If TRUE, covariates are scaled to have mean 0 and SD 1.
#' @return The fitted model object from the specified `fit_fun`.
#' @export
fit_cb_model <- function(cb_data,
                         regularization = c('elastic-net', 'SCAD'), 
                         lambda, alpha = NULL,
                         fit_fun = MNlogisticSAGAN,
                         param_start = NULL,
                         n_unpenalized = 2,
                         standardize = TRUE, 
                         all_event_levels = NULL,
                         ...) {
    
    # Load libraries efficiently
    if (!requireNamespace("Matrix", quietly = TRUE)) cli::cli_abort("Matrix package required")
    
    regularization <- rlang::arg_match(regularization)
    penalty_params <- prepare_penalty_params(regularization, lambda, alpha)
    
    if (is.null(all_event_levels)) all_event_levels <- sort(unique(cb_data$event))
    
    Y_factor <- factor(cb_data$event, levels = all_event_levels)
    K_params <- nlevels(factor(cb_data$event)) - 1
    
    penalized_covs <- as.matrix(cb_data$covariates)
    p_covs <- ncol(penalized_covs)
    
    scaler <- list(center = rep(0, p_covs), scale = rep(1, p_covs))
    
    names(scaler$center) <- colnames(penalized_covs)
    names(scaler$scale) <- colnames(penalized_covs)
    
    if (standardize) {
        # Use helper function to standardize
        std_res <- standardize_cb_data(cb_data)
        cb_data <- std_res$cb_data # Use standardized data
        scaler <- std_res$scaler
        penalized_covs <- cb_data$covariates # Update local reference
        
        if(!is.null(param_start)){
            
            beta_raw <- param_start[1:p_covs, , drop = FALSE]
            
            intercept_raw <- param_start[p_covs + 2, , drop = FALSE] 
            
            # Scale beta
            beta_scaled <- beta_raw * scaler$scale
            
            # Adjust intercept
            intercept_adj <- colSums(beta_raw * scaler$center)
            intercept_scaled <- intercept_raw + intercept_adj
            
            # Update param_start with scaled values
            param_start[1:p_covs, ] <- beta_scaled
            
            param_start[p_covs + 2, ] <- intercept_scaled
            
        }
        
    }
    
    # X <- stats::model.matrix(~., 
    #                   data = data.frame(cbind(penalized_covs, 
    #                                           time = log(cb_data$time))))
    
    X <- cbind(penalized_covs, "log(time)" = log(cb_data$time), "(Intercept)" = 1)
    
    # design matrix
    # X <- as.matrix(cbind(penalized_covs, time = log(cb_data$time), 1))
    
    opt_args <- list(X = X, Y = Y_factor, offset = cb_data$offset,
                     N_covariates = n_unpenalized, 
                     regularization = regularization,
                     lambda1 = penalty_params$lambda1,
                     lambda2 = penalty_params$lambda2,
                     param_start = param_start,
                     ...)
    
    fit <- do.call(fit_fun, opt_args)
    
    # Re-scaling
    if (standardize && !is.null(fit$coefficients)) {
        fit$coefficients <- unstandardize_coefficients(fit$coefficients, scaler, colnames(cb_data$covariates))
    }
    
    call <-  match.call()
    call$lambda <- lambda
    call$alpha <- penalty_params$alpha
    call$n_unpenalized <- n_unpenalized
    
    fit <- c(fit, 
             scaler = list(scaler),
             adjusted = F,
             call = call,
             lambda = lambda,
             alpha = penalty_params$alpha)
    class(fit) <- "cbSCRIP"
    return(fit)
}

#' Print a cbSCRIP object
#'
#' Provides a concise and informative summary of a fitted `cbSCRIP` model,
#' highlighting the regularization parameters, convergence status, and the
#' specific coefficients selected by the penalty.
#'
#' @param x An object of class `cbSCRIP`.
#' @param ... Additional arguments (currently unused).
#' @param print_limit The maximum number of non-zero coefficients to display.
#'   If more are selected, a summary message is shown.
#'
#' @return The original object `x`, invisibly.
#' @importFrom cli cat_rule cat_bullet style_bold style_italic symbol
#' @importFrom crayon green red
#' @export
print.cbSCRIP <- function(x, ..., print_limit = 10) {
    # --- Header ---
    cli::cat_rule(cli::style_bold("Case-Base Competing Risks Model (cbSCRIP)"), col = "blue")
    cat("\n")
    
    # --- Model & Penalty Details ---
    lambda_call <- x$lambda
    alpha_call <- x$alpha
    
    if (!is.null(lambda_call)) {
        # Evaluate the expression to get its numeric value
        cli::cat_bullet("Lambda: ", sprintf("%.4f", lambda_call), bullet = "info")
    }
    if (!is.null(alpha_call)) {
        # Also evaluate alpha for robustness
        cli::cat_bullet("Alpha (for elastic-net): ", alpha_call, bullet = "info")
    }
    
    # --- Selected Coefficients Details ---
    cli::cat_rule("Selected Coefficients", col = "blue")
    
    coefs <- x$coefficients
    if (is.null(coefs) || !is.matrix(coefs)) {
        cat("No coefficient matrix found.\n")
        invisible(x)
        return()
    }
    
    # Convert the wide coefficient matrix
    long_coefs <- as.data.frame(as.table(coefs, base = list(NUMBERS)))
    long_coefs$Var2 <- as.numeric(as.factor(long_coefs$Var2))
    colnames(long_coefs) <- c("Variable", "Cause", "Coefficient")
    
    # Filter for non-zero coefficients
    selected_coefs <- long_coefs[abs(long_coefs$Coefficient) > 1e-10, ]
    n_selected <- nrow(selected_coefs)
    
    if (n_selected == 0) {
        cat("No variables selected at this lambda value.\n")
    } else {
        cat("Found", cli::style_bold(n_selected), "non-zero coefficients:\n\n")
        
        # Format for printing
        selected_coefs$Variable <- as.character(selected_coefs$Variable)
        selected_coefs$Cause <- as.character(selected_coefs$Cause)
        selected_coefs$Coefficient <- sprintf("% .5f", selected_coefs$Coefficient)
        
        # Print all if under the limit, otherwise print a summary
        if (n_selected <= print_limit) {
            print(as.data.frame(selected_coefs), row.names = FALSE, right = FALSE)
        } else {
            print(utils::head(as.data.frame(selected_coefs), n = print_limit), row.names = FALSE, right = FALSE)
            cat(cli::style_italic(paste0("\n... and ", n_selected - print_limit, " more non-zero coefficients.")))
        }
    }
    
    # Return object invisibly, as is standard for print methods
    invisible(x)
}