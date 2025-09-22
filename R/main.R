
MNlogistic <- function(X, Y, offset, N_covariates,
                       regularization = 'l1', transpose = F,
                       lambda1, lambda2 = 0, lambda3 = 0,
                       learning_rate = 1e-4, tolerance = 1e-4,
                       niter_inner_mtplyr = 7, maxit = 100, 
                       ncores = -1,
                       group_id = NULL, group_weights = NULL, 
                       groups = NULL, groups_var = NULL, 
                       own_variables = NULL, N_own_variables = NULL,
                       param_start = NULL, verbose = FALSE,
                       save_history = FALSE) {
    ## Dimensions and checks
    nx <- nrow(X)

    if (!is.vector(Y)) {Y <- as.vector(Y)}
    ny <- length(Y)

    if (!is.vector(offset)) {offset <- as.vector(offset)}
    noff <- length(offset)

    if (nx == ny & nx == noff) {
        n <- nx
    } else {
        stop('X, Y and offset have different number of observations.')
    }

    p <- ncol(X)

    K <- length(unique(Y)) - 1

    ## regularization
    
    
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared", "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf", "sparse-group-lasso-l2",
              "sparse-group-lasso-linf", "l1l2", "l1linf", "l1l2+l1", "l1linf+l1",
              "l1linf-row-column", "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    pen4 <- c("SCAD")
    penalty_code <- 0
    if (regularization %in% pen1) { penalty_code <- 1 }
    if (regularization %in% pen2) { penalty_code <- 2 }
    if (regularization %in% pen3) { penalty_code <- 3 }
    if (regularization %in% pen4) { penalty_code <- 4 }
    if (penalty_code == 0 && regularization != "none") { 
        stop('The provided regularization is not supported.')
    }
    if(regularization == "none" && penalty_code == 0) penalty_code <- 1 
    
    
    if (is.null(group_id)) group_id <- rep(0L, p) # Default if missing
    if (is.null(group_weights)) group_weights <- vector(mode = 'double')
    if (is.null(groups)) groups <- matrix(NA_real_, nrow=0, ncol=0) # Empty matrix
    if (is.null(groups_var)) groups_var <- matrix(NA_real_, nrow=0, ncol=0)
    if (is.null(own_variables)) own_variables <- vector(mode = 'integer')
    if (is.null(N_own_variables)) N_own_variables <- vector(mode = 'integer')
    
    if (penalty_code == 1) {
        
        if(!is.matrix(groups) || nrow(groups)==0) groups <- matrix(NA_real_, nrow=1, ncol=1) 
        if(!is.matrix(groups_var) || nrow(groups_var)==0) groups_var <- matrix(NA_real_, nrow=1, ncol=1)
    } else if (penalty_code == 2) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=2.')
        if (is.null(groups_var) || nrow(groups_var)==0) stop('Required input `groups_var` is missing for penalty=2.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=2.')
    } else if (penalty_code == 3) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=3.')
        if (is.null(own_variables) || length(own_variables)==0) stop('Required input `own_variables` is missing for penalty=3.')
        if (is.null(N_own_variables) || length(N_own_variables)==0) stop('Required input `N_own_variables` is missing for penalty=3.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=3.')
    }

    ## call 
    
    result <- MultinomLogistic(X = as.matrix(X), 
                                Y = as.numeric(Y), 
                                offset = as.numeric(offset), K = K,
                                reg_p = as.integer(p - N_covariates),
                                penalty = as.integer(penalty_code),
                                regul = regularization,
                                transpose = transpose,
                                grp_id = group_id,
                                etaG = group_weights, 
                                grp = groups,
                                grpV = groups_var,
                                own_var = own_variables,
                                N_own_var = N_own_variables,
                                lam1 = as.double(lambda1),
                                lam2 = as.double(lambda2),
                                lam3 = as.double(lambda3),
                                learning_rate = as.double(learning_rate),
                                tolerance = as.double(tolerance),
                                niter_inner = as.integer(niter_inner_mtplyr * nx),
                                maxit = as.integer(maxit),
                                ncores = as.integer(ncores),
                                param_start = param_start,
                                verbose = as.logical(verbose),
                                save_history = as.logical(save_history))  
    
    nzc <- length(result$`Sparse Estimates`@i)
 
    return(list(
        coefficients = result$Estimates,
        coefficients_sparse = result$`Sparse Estimates`,
        coefficients_history = result$History, # Renamed for clarity
        converged = result$Converged,
        convergence_pass = result$`Convergence Iteration`,
        no_non_zero = nzc
    ))
}


#' @export
MNlogisticAcc <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = FALSE,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                # learning_rate = 1e-4,
                                # momentum_gamma = 0.9, #  momentum
                                c_factor = 100,
                                v_factor = 0,
                                pos = FALSE,          #  Positivity constraint
                                tolerance = 1e-4,
                                niter_inner_mtplyr = 2,
                                maxit = 100, ncores = -1,
                                group_id = NULL, group_weights = NULL, # etaG
                                groups = NULL, groups_var = NULL,     # grp, grpV
                                own_variables = NULL, N_own_variables = NULL,
                                param_start = NULL, verbose = TRUE,
                                save_history = FALSE) {
    
    nx <- nrow(X)
    if (!is.matrix(X)) X <- as.matrix(X) # Ensure X is a matrix
    if (!is.vector(Y)) Y <- as.vector(Y)
    if (!is.vector(offset)) offset <- as.vector(offset)
    if (nx != length(Y) || nx != length(offset)) {
        stop('X, Y and offset have different number of observations.')
    }
    n <- nx
    p <- ncol(X)
    
    valid_Y_values <- Y[!is.na(Y) & Y > 0]
    if (length(valid_Y_values) == 0) stop("Y does not contain any valid positive class labels.")
    unique_classes <- unique(valid_Y_values)
    K_val <- length(unique_classes)
    
    if (K_val > 0 && !all(sort(unique_classes) == 1:K_val)) {
        warning(paste("Classes in Y are:", paste(sort(unique_classes), collapse=", "),
                      ". K is set to", K_val, 
                      ". Ensure C++ handles these actual Y values (e.g., if they are not 1 to K_val consecutive)."))
    }
    
    
    
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared", "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf", "sparse-group-lasso-l2",
              "sparse-group-lasso-linf", "l1l2", "l1linf", "l1l2+l1", "l1linf+l1",
              "l1linf-row-column", "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    pen4 <- c("SCAD")
    penalty_code <- 0
    if (regularization %in% pen1) { penalty_code <- 1 }
    if (regularization %in% pen2) { penalty_code <- 2 }
    if (regularization %in% pen3) { penalty_code <- 3 }
    if (regularization %in% pen4) { penalty_code <- 4 }
    if (penalty_code == 0 && regularization != "none") { 
        stop('The provided regularization is not supported.')
    }
    if(regularization == "none" && penalty_code == 0) penalty_code <- 1 
    

    if (is.null(group_id)) group_id <- rep(0L, p) # Default if missing
    if (is.null(group_weights)) group_weights <- vector(mode = 'double')
    if (is.null(groups)) groups <- matrix(NA_real_, nrow=0, ncol=0) # Empty matrix
    if (is.null(groups_var)) groups_var <- matrix(NA_real_, nrow=0, ncol=0)
    if (is.null(own_variables)) own_variables <- vector(mode = 'integer')
    if (is.null(N_own_variables)) N_own_variables <- vector(mode = 'integer')
    
    if (penalty_code == 1) {
    
        if(!is.matrix(groups) || nrow(groups)==0) groups <- matrix(NA_real_, nrow=1, ncol=1) 
        if(!is.matrix(groups_var) || nrow(groups_var)==0) groups_var <- matrix(NA_real_, nrow=1, ncol=1)
    } else if (penalty_code == 2) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=2.')
        if (is.null(groups_var) || nrow(groups_var)==0) stop('Required input `groups_var` is missing for penalty=2.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=2.')
    } else if (penalty_code == 3) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=3.')
        if (is.null(own_variables) || length(own_variables)==0) stop('Required input `own_variables` is missing for penalty=3.')
        if (is.null(N_own_variables) || length(N_own_variables)==0) stop('Required input `N_own_variables` is missing for penalty=3.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=3.')
    }
    
    
    X <- as.matrix(X)
    Y <- as.integer(Y)
    offset <- as.double(offset)
    group_id <- as.integer(group_id)
    group_weights <- as.double(group_weights)
    groups <- as.matrix(groups)
    groups_var <- as.matrix(groups_var)
    own_variables <- as.integer(own_variables)
    N_own_variables <- as.integer(N_own_variables)
    
    
    result <- MultinomLogisticAcc(X = X, Y = Y, offset = offset, K = K_val,
                                  reg_p = as.integer(p - N_covariates),
                                  penalty = as.integer(penalty_code),
                                  regul = regularization,
                                  transpose = transpose,
                                  grp_id = group_id,
                                  etaG = group_weights, 
                                  grp = groups,
                                  grpV = groups_var,
                                  own_var = own_variables,
                                  N_own_var = N_own_variables,
                                  lam1 = as.double(lambda1),
                                  lam2 = as.double(lambda2),
                                  lam3 = as.double(lambda3),
                                  c_factor = c_factor,
                                  v_factor = v_factor,
                                  # learning_rate = as.double(learning_rate),
                                  # momentum_gamma = as.double(momentum_gamma), 
                                  tolerance = as.double(tolerance),
                                  niter_inner = as.integer(niter_inner_mtplyr * nx),
                                  maxit = as.integer(maxit),
                                  ncores = as.integer(ncores),
                                  pos = as.logical(pos), 
                                  param_start = param_start,
                                  verbose = verbose,
                                  save_history = as.logical(save_history))    
    
    if (inherits(result$`Sparse Estimates`, "sparseMatrix")) {
        nzc <- Matrix::nnzero(result$`Sparse Estimates`)
    } else {
        nzc <- sum(result$`Sparse Estimates` != 0)
    }
    
    return(list(
        coefficients = result$Estimates,
        coefficients_sparse = result$`Sparse Estimates`,
        coefficients_history = result$History, # Renamed for clarity
        converged = result$Converged,
        convergence_pass = result$`Convergence Iteration`,
        no_non_zero = nzc
    ))
}

#' @export
mtool.MNlogisticSAGA <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = FALSE,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                # learning_rate = 1e-4,
                                # momentum_gamma = 0.9, #  momentum
                                c_factor = 100,
                                v_factor = 0,
                                pos = FALSE,          #  Positivity constraint
                                tolerance = 1e-4,
                                # niter_inner_mtplyr = 2,
                                maxit = 100, ncores = -1,
                                group_id = NULL, group_weights = NULL, # etaG
                                groups = NULL, groups_var = NULL,     # grp, grpV
                                own_variables = NULL, N_own_variables = NULL,
                                param_start = NULL, verbose = TRUE,
                                save_history = FALSE) {
    
    nx <- nrow(X)
    if (!is.matrix(X)) X <- as.matrix(X) # Ensure X is a matrix
    if (!is.vector(Y)) Y <- as.vector(Y)
    if (!is.vector(offset)) offset <- as.vector(offset)
    if (nx != length(Y) || nx != length(offset)) {
        stop('X, Y and offset have different number of observations.')
    }
    n <- nx
    p <- ncol(X)
    
    valid_Y_values <- Y[!is.na(Y) & Y > 0]
    if (length(valid_Y_values) == 0) stop("Y does not contain any valid positive class labels.")
    unique_classes <- unique(valid_Y_values)
    K_val <- length(unique_classes)
    
    if (K_val > 0 && !all(sort(unique_classes) == 1:K_val)) {
        warning(paste("Classes in Y are:", paste(sort(unique_classes), collapse=", "),
                      ". K is set to", K_val, 
                      ". Ensure C++ handles these actual Y values (e.g., if they are not 1 to K_val consecutive)."))
    }
    
    
    
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared", "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf", "sparse-group-lasso-l2",
              "sparse-group-lasso-linf", "l1l2", "l1linf", "l1l2+l1", "l1linf+l1",
              "l1linf-row-column", "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    pen4 <- c("SCAD")
    penalty_code <- 0
    if (regularization %in% pen1) { penalty_code <- 1 }
    if (regularization %in% pen2) { penalty_code <- 2 }
    if (regularization %in% pen3) { penalty_code <- 3 }
    if (regularization %in% pen4) { penalty_code <- 4 }
    if (penalty_code == 0 && regularization != "none") { 
        stop('The provided regularization is not supported.')
    }
    if(regularization == "none" && penalty_code == 0) penalty_code <- 1 
    
    
    if (is.null(group_id)) group_id <- rep(0L, p) # Default if missing
    if (is.null(group_weights)) group_weights <- vector(mode = 'double')
    if (is.null(groups)) groups <- matrix(NA_real_, nrow=0, ncol=0) # Empty matrix
    if (is.null(groups_var)) groups_var <- matrix(NA_real_, nrow=0, ncol=0)
    if (is.null(own_variables)) own_variables <- vector(mode = 'integer')
    if (is.null(N_own_variables)) N_own_variables <- vector(mode = 'integer')
    
    if (penalty_code == 1) {
        
        if(!is.matrix(groups) || nrow(groups)==0) groups <- matrix(NA_real_, nrow=1, ncol=1) 
        if(!is.matrix(groups_var) || nrow(groups_var)==0) groups_var <- matrix(NA_real_, nrow=1, ncol=1)
    } else if (penalty_code == 2) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=2.')
        if (is.null(groups_var) || nrow(groups_var)==0) stop('Required input `groups_var` is missing for penalty=2.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=2.')
    } else if (penalty_code == 3) {
        if (is.null(groups) || nrow(groups)==0) stop('Required input `groups` is missing for penalty=3.')
        if (is.null(own_variables) || length(own_variables)==0) stop('Required input `own_variables` is missing for penalty=3.')
        if (is.null(N_own_variables) || length(N_own_variables)==0) stop('Required input `N_own_variables` is missing for penalty=3.')
        if (is.null(group_weights) || length(group_weights)==0) stop('Required input `group_weights` is missing for penalty=3.')
    }
    
    
    X <- as.matrix(X)
    Y <- as.integer(Y)
    offset <- as.double(offset)
    group_id <- as.integer(group_id)
    group_weights <- as.double(group_weights)
    groups <- as.matrix(groups)
    groups_var <- as.matrix(groups_var)
    own_variables <- as.integer(own_variables)
    N_own_variables <- as.integer(N_own_variables)
    
    
    result <- MultinomLogisticSAGA(X = X, Y = Y, offset = offset, K = K_val,
                                  reg_p = as.integer(p - N_covariates),
                                  penalty = as.integer(penalty_code),
                                  regul = regularization,
                                  transpose = transpose,
                                  grp_id = group_id,
                                  etaG = group_weights, 
                                  grp = groups,
                                  grpV = groups_var,
                                  own_var = own_variables,
                                  N_own_var = N_own_variables,
                                  lam1 = as.double(lambda1),
                                  lam2 = as.double(lambda2),
                                  lam3 = as.double(lambda3),
                                  c_factor = c_factor,
                                  v_factor = v_factor,
                                  # learning_rate = as.double(learning_rate),
                                  # momentum_gamma = as.double(momentum_gamma), 
                                  tolerance = as.double(tolerance),
                                  maxit = as.integer(maxit),
                                  ncores = as.integer(ncores),
                                  pos = as.logical(pos), 
                                  param_start = param_start,
                                  verbose = verbose,
                                  save_history = as.logical(save_history))    
    
    if (inherits(result$`Sparse Estimates`, "sparseMatrix")) {
        nzc <- Matrix::nnzero(result$`Sparse Estimates`)
    } else {
        nzc <- sum(result$`Sparse Estimates` != 0)
    }
    
    return(list(
        coefficients = result$Estimates,
        coefficients_sparse = result$`Sparse Estimates`,
        coefficients_history = result$History, # Renamed for clarity
        converged = result$Converged,
        convergence_pass = result$`Convergence Iteration`,
        no_non_zero = nzc
    ))
}
