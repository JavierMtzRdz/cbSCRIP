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
                         fit_fun = MNlogisticSAGA,
                         param_start = NULL,
                         n_unpenalized = 2,
                         standardize = TRUE, 
                         all_event_levels = NULL,
                         ...) {
    
    library(Matrix)
    library(cbSCRIP)
    
    regularization <- rlang::arg_match(regularization)
    penalty_params <- prepare_penalty_params(regularization, lambda, alpha)
    
    if (is.null(all_event_levels)) all_event_levels <- sort(unique(cb_data$event))
    Y_factor <- factor(cb_data$event, levels = all_event_levels)
    
    penalized_covs <- cb_data$covariates
    scaler <- NULL
    if (standardize) {
        #  center and scale
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
    
    X <- stats::model.matrix(~., 
                      data = data.frame(cbind(penalized_covs, 
                                              time = log(cb_data$time))))
    
    X <- cbind(X[,-1], X[,1])
    
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
    if (standardize) {
        # Re-scale both matrices
        for (coef_type in c("coefficients")) {
            if (!is.null(fit[[coef_type]])) {
                coefs_scaled <- fit[[coef_type]]
                p_penalized <- ncol(penalized_covs)
                
                # Separate penalized from unpenalized coefficients
                beta_penalized_scaled <- coefs_scaled[1:p_penalized, , drop = FALSE]
                
                # Re-scale coefficients
                beta_penalized_orig <- sweep(beta_penalized_scaled, 1, scaler$scale, FUN = "/")
                
                # Adjust the intercept
                intercept_adjustment <- colSums(as.matrix(sweep(beta_penalized_scaled, 1, scaler$center / scaler$scale, FUN = "*")), na.rm = TRUE)
                intercept_orig <- coefs_scaled[nrow(coefs_scaled), ] - intercept_adjustment
                
                # full coefficient matrix on the original scale
                coefs_orig <- rbind(
                    beta_penalized_orig,
                    coefs_scaled[p_penalized + 1, , drop = FALSE], 
                    intercept_orig
                )
                
                # Update row names to match the original structure
                rownames(coefs_orig) <- c(colnames(cb_data$covariates), 
                                          "log(time)", "(Intercept)")
                
                # Replace the coefficients in the fit object
                fit[[coef_type]] <- round(coefs_orig, 8)
            }
        }
        fit <- c(fit, scaler = list(scaler))
    }
    fit <- c(fit, 
             scaler = list(scaler),
             adjusted = F,
             call = match.call())
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
    # Use rlang::`%||%` to handle cases where lambda might be named differently
    lambda_expr <- x$call$lambda
    alpha_expr <- x$call$alpha
    
    if (!is.null(lambda_expr)) {
        # Evaluate the expression to get its numeric value
        lambda_val <- eval(lambda_expr, envir = parent.frame())
        cli::cat_bullet("Lambda: ", sprintf("%.4f", lambda_val), bullet = "info")
    }
    if (!is.null(alpha_expr)) {
        # Also evaluate alpha for robustness
        alpha_val <- eval(alpha_expr, envir = parent.frame())
        cli::cat_bullet("Alpha (for elastic-net): ", alpha_val, bullet = "info")
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