#################### R wrapper of mtool ####################

# mtool.estimate

mtool.estimate <- function(X, Y, weight, N_covariates,
                           loss = 'ls', regularization = 'l1',
                           transpose = F,
                           lambda1, lambda2 = 0, lambda3 = 0,
                           learning_rate = 1e-4, tolerance = 1e-3,
                           niter_inner_mtplyr = 5, maxit = 100, ncores = -1,
                           group_id, group_weights,
                           groups, groups_var,
                           own_variables, N_own_variables) {
    ## Dimensions and checks
    nx <- nrow(X)
    ny <- nrow(Y)
    if (nx == ny) {
        n <- nx
    } else {
        stop('X and Y have different number of observations.')
    }
    
    if (missing(weight)) {
        weight = rep(1, nx)
    }
    
    if (!length(weight) == nx) {
        stop(" 'weight' should be a vector with length equal to the sample size.")
    }
    
    p <- ncol(X)
    
    K <- ncol(Y)
    
    ## loss
    lossfns <- c('ls', 'logistic', 'cox')
    
    if (loss %in% lossfns) {
        lossfunc <- which(lossfns == loss)
    } else {
        stop('The provided loss function is not supported.')
    }
    
    ## regularization
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }
    
    ### check regularization-specific inputs
    #### penalty = 1, call proximal(Flat), requires `group_id` in integer vector
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 2, call proximalGraph
    #### requires `groups` and `groups_var` in integer matrices and `group_weights` in double vector
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 3, call proximalGraph
    #### requires `own_variables` and `N_own_variables` in integer vectors, `group_weights` in double vector
    #### and `groups` in integer matrix
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }
    
    ## extract number of observations per task and indices of the observations
    obs.flag <- !is.na(Y)
    n_k <- apply(obs.flag, MARGIN = 2, FUN = sum)
    ### extract the indices of non-missing observations for each task
    task_rowid <- apply(obs.flag, MARGIN = 2, FUN = which)
    #### if missingness pattern is identical across tasks, task_rowid will be a matrix instead of a list
    if (is.list(task_rowid)) {
        task_rowid <- unlist(task_rowid)
    } else {
        if (is.matrix(task_rowid)) {
            task_rowid <- as.vector(task_rowid)
        }
    }
    
    ## call mtool main function
    result <- mtool(X = X, Y = Y, wt = weight, K = K, nk_vec = n_k,
                    task_rowid = task_rowid, reg_p = p - N_covariates,
                    loss = lossfunc, penalty = penalty, regul = regularization,
                    transpose = transpose,
                    grp_id = group_id, etaG = group_weights,
                    grp = groups, grpV = groups_var,
                    own_var = own_variables, N_own_var = N_own_variables,
                    lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                    learning_rate = learning_rate, tolerance = tolerance,
                    niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                    ncores = ncores)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                no_non_zero = nzc))
}






########## include a given random fold id assignment

# cross-validation
mtool.cv_error <- function(X, Y, weight, nfold, fold_seed,
                           CV_criterion,
                           N_covariates,
                           loss = 'ls', regularization = 'l1',
                           transpose = F,
                           lambda1, lambda2 = 0, lambda3 = 0,
                           learning_rate = 1e-4, tolerance = 1e-3,
                           niter_inner_mtplyr = 5, maxit = 100, ncores = 1,
                           group_id, group_weights,
                           groups, groups_var,
                           own_variables, N_own_variables) {
    ## Dimensions and checks
    nx <- nrow(X)
    ny <- nrow(Y)
    if (nx == ny) {
        n <- nx
    } else {
        stop('X and Y have different number of observations.')
    }
    
    if (ncores != 1) {
        cat('More than 1 core is being used for the computation of the proximal operator. Be careful if the cross validation procedure is running on parallel.')
    }
    
    if (missing(weight)) {
        weight = rep(1, nx)
    }
    
    if (!length(weight) == nx) {
        stop(" 'weight' should be a vector with length equal to the sample size.")
    }
    
    p <- ncol(X)
    
    K <- ncol(Y)
    
    ## loss
    lossfns <- c('ls', 'logistic', 'cox')
    
    if (loss %in% lossfns) {
        lossfunc <- which(lossfns == loss)
    } else {
        stop('The provided loss function is not supported.')
    }
    
    ## regularization
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }
    
    ### check regularization-specific inputs
    #### penalty = 1, call proximal(Flat), requires `group_id` in integer vector
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 2, call proximalGraph
    #### requires `groups` and `groups_var` in integer matrices and `group_weights` in double vector
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 3, call proximalGraph
    #### requires `own_variables` and `N_own_variables` in integer vectors, `group_weights` in double vector
    #### and `groups` in integer matrix
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }
    
    ## extract number of observations per task and indices of the observations
    obs.flag <- !is.na(Y)
    n_k <- apply(obs.flag, MARGIN = 2, FUN = sum)
    task_rowid <- apply(obs.flag, MARGIN = 2, FUN = which)
    #### if missingness pattern is identical across tasks, task_rowid will be a matrix instead of a list
    if (is.matrix(task_rowid)) {
        task_rowid <- as.list(as.data.frame(task_rowid))
    }
    set.seed(fold_seed)
    foldid <- lapply(task_rowid, FUN = function(x) {sample(rep(seq(nfold), length = length(x)))})
    set.seed(NULL)
    
    # cat("fold_ids", utils::head(foldid[[5]],10), "random", rnorm(1))
    
    ## cross validation
    ntrain <- nx * (1 - 1/nfold)
    pred.mat <- matrix(NA, nrow = nx, ncol = K)
    err.fold <- numeric(nfold)
    
    for (i in seq(nfold)) {
        cat("\n fold", i)
        
        ## mapply() works on lists and arrays (vectors, matrices, etc.)
        task_rowid_train <- mapply(task_rowid, foldid, FUN = function(x, y) {x[y != i ]}, SIMPLIFY = F)
        task_rowid_test <- mapply(task_rowid, foldid, FUN = function(x, y) {x[y == i ]}, SIMPLIFY = F)
        
        n_k <- unlist(lapply(task_rowid_train, FUN = length))
        
        fit <-  mtool(X = X, Y = Y, wt = weight, K = K, nk_vec = n_k,
                      task_rowid = unlist(task_rowid_train), reg_p = p - N_covariates,
                      loss = lossfunc, penalty = penalty, regul = regularization,
                      transpose = transpose,
                      grp_id = group_id, etaG = group_weights,
                      grp = groups, grpV = groups_var,
                      own_var = own_variables, N_own_var = N_own_variables,
                      lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                      learning_rate = learning_rate, tolerance = tolerance,
                      niter_inner = niter_inner_mtplyr * ntrain,
                      maxit = maxit, ncores = ncores)
        beta.list <- split(fit$`Sparse Estimates`, f = rep(1:K, each = p))
        
        pred <- mapply(task_rowid_test, beta.list, FUN = function(x, y) {X[x,] %*% y}, SIMPLIFY = F)
        
        pred.mat.fold <- matrix(NA, nrow = nx, ncol = K)
        
        for (k in seq(K)) {
            pred.mat[task_rowid_test[[k]], k] <- pred[[k]]
            pred.mat.fold[task_rowid_test[[k]], k] <- pred[[k]]
        }
        
        ### cv error for each fold to calculate var(cv.error)
        if (CV_criterion == "MSE") {
            error.mat.fold <- (pred.mat.fold - Y)^2
            err.fold[i] <- sum(apply(error.mat.fold, MARGIN = 2, FUN = mean, na.rm = T))
        }
        
        if (CV_criterion == "Pearson") {
            err.fold[i] <- mean(diag(stats::cor(pred.mat, Y, use = "pairwise.complete.obs", method = "pearson")))
        }
    }
    
    # confirm missingness patterns in pred.mat and y are the same
    ## Somehow there can be false alarms.
    missing.check <- identical(is.na(Y), is.na(pred.mat))
    if (!missing.check) {
        warning("Something may be wrong with cross-validation. Missingness in CV prediction and in original Y may be different.")
    }
    
    if (CV_criterion == "MSE") {
        error.mat <- (pred.mat - Y)^2
        # print(sum(apply(error.mat, MARGIN = 2, FUN = mean, na.rm = T)))
        error <- sum(apply(error.mat, MARGIN = 2, FUN = mean, na.rm = T))
    }
    
    if (CV_criterion == "Pearson") {
        error <- mean(diag(stats::cor(pred.mat, Y, use = "pairwise.complete.obs", method = "pearson")))
    }
    
    var.error <- 1/K * stats::var(err.fold)
    
    return(list(CV_prediction = pred.mat,
                CV_error = error,
                CV_error_var = var.error))
}


# Automatic tree builder with manual input
## input: a list of tree nodes
## equally spaced node height
## output: a list of own_variables, N_own_variables, groups, eta_g, as required by mtool.estimate and others.


tree_builder <- function(tree) {
    n.nodes <- length(tree)
    
    tree <- lapply(tree, sort)
    
    reorder1 <- order(sapply(tree, FUN = length), decreasing = T)
    
    tree <- tree[reorder1]
    
    reorder2 <- order(sapply(tree, FUN = utils::head, n = 1))
    
    tree <- tree[reorder2]
    
    # pre-allocate own_variable, N_own_variable, group weights, groups matrix
    n_own_var <- integer(n.nodes)
    eta_g <- numeric(n.nodes)
    g <- matrix(0L, nrow = n.nodes, ncol = n.nodes)
    
    # own_variables
    own_var <- sapply(tree, FUN = utils::head, n = 1) - 1L
    
    # N_own_variables
    for (i in seq(n.nodes)) {
        nodes.set <- tree[-(1:i)]
        node.tmp <- tree[[i]]
        nov.tmp <- 0L
        for (j in seq(length(node.tmp))) {
            if (!node.tmp[j] %in% unlist(nodes.set)) {
                nov.tmp <- nov.tmp + 1L
            }
        }
        n_own_var[i] <- nov.tmp
    }
    
    # groups
    for (i in 2:n.nodes) {
        for (j in (i-1):1) {
            if ( setequal(intersect(tree[[i]], tree[[j]]),  tree[[i]]) ) {
                g[i, j] <- 1L
                break
            }
        }
    }
    
    # group weights: eta_g
    
    w <- pass.down <- numeric(n.nodes)
    pass.down[1] <- 1
    w[1] <- 0
    
    sub.tree.old <- tree
    
    for (i in 2:n.nodes) {
        node <- tree[[i]]
        
        ancester <- which(g[i,] == 1L)
        
        sub.tree <- tree[sapply(tree, FUN = function(x) {setequal(x, intersect(x, node))})]
        sub.tree.ancester <- sub.tree
        sub.tree.ancester[[length(sub.tree.ancester) + 1]] <- tree[[ancester]]
        
        ancester.height <- max(table(unlist(sub.tree.ancester)))
        node.height <- max(table(unlist(sub.tree)))
        pass.down[i] <- (node.height - 1) / (ancester.height - 1) * pass.down[ancester]
        
        if (ancester == 1) {
            w[i] <- 1 - pass.down[i]
        } else {
            w[i] <- w[ancester] / (1 - pass.down[ancester]) * pass.down[ancester] * (1 - pass.down[i])
        }
    }
    return(list(own_variables = own_var,
                N_own_variables = n_own_var,
                groups = g,
                eta_g = w))
    
}








# Automatic tree builder with hclust() input
## input: an hclust object
## height generated by hclust()
## output: a list of own_variables, N_own_variables, groups, eta_g, as required by mtool.estimate and others.

tree_builder_hclust <- function(cluster) {
    merge <- cluster$merge
    n.nonleafnodes <- nrow(merge)
    n.leafnodes <- - min(merge)
    n.nodes <- n.nonleafnodes + n.leafnodes
    
    tree <- vector(mode = 'list', length = n.nodes)
    height <- numeric(n.nodes)
    
    for (i in seq(n.nonleafnodes)) {
        node <- merge[i,]
        if (node[1] < 0 & node[2] < 0) {
            tree[[i]] <- - node
        } else {
            if (node[1] <0 | node[2] < 0) {
                leaf <- - min(node)
                prev.node <- max(node)
                tree[[i]] <- c(leaf, tree[[prev.node]])
            } else {
                tree[[i]] <- c(tree[[ node[1] ]], tree[[ node[2] ]])
            }
        }
    }
    
    leaves.indices <- n.nonleafnodes + 1:n.leafnodes
    
    for (i in leaves.indices) {
        tree[[i]] <- i - n.nonleafnodes
    }
    
    height <- c(cluster$height, rep(0, n.leafnodes))
    
    tree <- lapply(tree, sort)
    
    reorder1 <- order(sapply(tree, FUN = length), decreasing = T)
    
    tree <- tree[reorder1]
    height <- height[reorder1]
    
    reorder2 <- order(sapply(tree, FUN = utils::head, n = 1))
    
    tree <- tree[reorder2]
    height <- height[reorder2]
    height <- height / max(height)
    
    # pre-allocate own_variable, N_own_variable, group weights, groups matrix
    n_own_var <- integer(n.nodes)
    eta_g <- numeric(n.nodes)
    g <- matrix(0L, nrow = n.nodes, ncol = n.nodes)
    
    # own_variables
    own_var <- sapply(tree, FUN = utils::head, n = 1) - 1L
    
    # N_own_variables
    for (i in seq(n.nodes)) {
        nodes.set <- tree[-(1:i)]
        node.tmp <- tree[[i]]
        nov.tmp <- 0L
        for (j in seq(length(node.tmp))) {
            if (!node.tmp[j] %in% unlist(nodes.set)) {
                nov.tmp <- nov.tmp + 1L
            }
        }
        n_own_var[i] <- nov.tmp
    }
    
    # groups
    for (i in 2:n.nodes) {
        for (j in (i-1):1) {
            if ( setequal(intersect(tree[[i]], tree[[j]]),  tree[[i]]) ) {
                g[i, j] <- 1L
                break
            }
        }
    }
    
    # group weights: eta_g
    
    w <- numeric(n.nodes)
    w[1] <- 0
    
    sub.tree.old <- tree
    
    for (i in 2:n.nodes) {
        
        ancestor <- which(g[i,] == 1L)
        
        # sub.tree <- nodes[sapply(nodes, FUN = function(x) {setequal(x, intersect(x, node))})]
        # sub.tree.ancestor <- sub.tree
        # sub.tree.ancestor[[length(sub.tree.ancestor) + 1]] <- nodes[[ancestor]]
        # 
        # ancestor.height <- max(table(unlist(sub.tree.ancestor)))
        # node.height <- max(table(unlist(sub.tree)))
        # pass.down[i] <- (node.height - 1) / (ancestor.height - 1) * pass.down[ancestor]
        
        if (ancestor == 1) {
            w[i] <- 1 - height[i]
        } else {
            w[i] <- w[ancestor] / (1 - height[ancestor]) * height[ancestor] * (1 - height[i])
        }
    }
    
    return(list(own_variables = own_var,
                N_own_variables = n_own_var,
                groups = g,
                eta_g = w,
                tree = tree))
}






# Penalized Multinomial Logistic Regression
mtool.MNlogistic <- function(X, Y, offset, N_covariates,
                             regularization = 'l1', transpose = F,
                             lambda1, lambda2 = 0, lambda3 = 0,
                             learning_rate = 1e-4, tolerance = 1e-4,
                             niter_inner_mtplyr = 7, maxit = 100, ncores = -1,
                             group_id, group_weights,
                             groups, groups_var,
                             own_variables, N_own_variables) {
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
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")

    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }

    ### check regularization-specific inputs
    #### penalty = 1, call proximal(Flat), requires `group_id` in integer vector
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }

    #### penalty = 2, call proximalGraph
    #### requires `groups` and `groups_var` in integer matrices and `group_weights` in double vector
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }

    #### penalty = 3, call proximalGraph
    #### requires `own_variables` and `N_own_variables` in integer vectors, `group_weights` in double vector
    #### and `groups` in integer matrix
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }

    ## call mtool main function
    result <- MultinomLogistic(X = X, Y = Y, offset = offset, K = K, reg_p = p - N_covariates,
                               penalty = penalty, regul = regularization, transpose = transpose,
                               grp_id = group_id, etaG = group_weights,
                               grp = groups, grpV = groups_var,
                               own_var = own_variables, N_own_var = N_own_variables,
                               lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                               learning_rate = learning_rate, tolerance = tolerance,
                               niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                               ncores = ncores)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                no_non_zero = nzc))
}



mtool.MNlogistic2 <- function(X, Y, offset, N_covariates,
                             regularization = 'l1', transpose = F,
                             lambda1, lambda2 = 0, lambda3 = 0,
                             learning_rate = 1e-4, tolerance = 1e-4,
                             niter_inner_mtplyr = 7, maxit = 100, ncores = -1,
                             group_id, group_weights,
                             groups, groups_var,
                             own_variables, N_own_variables,
                             param_start = NULL,
                             verbose = FALSE) {
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
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")

    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }

    ### check regularization-specific inputs
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }

    #### penalty = 2, call proximalGraph
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }

    #### penalty = 3, call proximalGraph
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }

    ## call mtool main function
    result <- MultinomLogistic2(X = X, Y = Y, offset = offset, K = K, reg_p = p - N_covariates,
                               penalty = penalty, regul = regularization, transpose = transpose,
                               grp_id = group_id, etaG = group_weights,
                               grp = groups, grpV = groups_var,
                               own_var = own_variables, N_own_var = N_own_variables,
                               lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                               learning_rate = learning_rate, tolerance = tolerance,
                               niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                               ncores = ncores,
                               param_start = param_start,
                               verbose = verbose)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                no_non_zero = nzc))
}




mtool.MNlogistic3 <- function(X, Y, offset, N_covariates,
                              regularization = 'l1', transpose = F,
                              lambda1, lambda2 = 0, lambda3 = 0,
                              learning_rate = 1e-4, tolerance = 1e-4,
                              niter_inner_mtplyr = 7, maxit = 100, ncores = -1,
                              group_id, group_weights,
                              groups, groups_var,
                              own_variables, N_own_variables) {
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
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }
    
    ### check regularization-specific inputs
    #### penalty = 1, call proximal(Flat), requires `group_id` in integer vector
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 2, call proximalGraph
    #### requires `groups` and `groups_var` in integer matrices and `group_weights` in double vector
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 3, call proximalGraph
    #### requires `own_variables` and `N_own_variables` in integer vectors, `group_weights` in double vector
    #### and `groups` in integer matrix
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }
    
    ## call mtool main function
    result <- MultinomLogistic3(X = X, Y = Y, offset = offset, K = K, reg_p = p - N_covariates,
                                penalty = penalty, regul = regularization, transpose = transpose,
                                grp_id = group_id, etaG = group_weights,
                                grp = groups, grpV = groups_var,
                                own_var = own_variables, N_own_var = N_own_variables,
                                lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                                learning_rate = learning_rate, tolerance = tolerance,
                                niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                                ncores = ncores)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                coefficientshist = result$CoefficientHistory,
                no_non_zero = nzc))
}


mtool.MNlogisticExp <- function(X, Y, offset, N_covariates,
                              regularization = 'l1', transpose = F,
                              lambda1, lambda2 = 0, lambda3 = 0,
                              learning_rate = 1e-4, tolerance = 1e-4,
                              niter_inner_mtplyr = 7, maxit = 100, ncores = -1,
                              group_id, group_weights,
                              groups, groups_var,
                              own_variables, N_own_variables) {
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
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }
    
    ### check regularization-specific inputs
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 2, call proximalGraph
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 3, call proximalGraph
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }
    
    ## call mtool main function
    result <- MultinomLogisticExp(X = X, Y = Y, offset = offset, K = K, reg_p = p - N_covariates,
                                penalty = penalty, regul = regularization, transpose = transpose,
                                grp_id = group_id, etaG = group_weights,
                                grp = groups, grpV = groups_var,
                                own_var = own_variables, N_own_var = N_own_variables,
                                lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                                learning_rate = learning_rate, tolerance = tolerance,
                                niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                                ncores = ncores)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                coefficientshist = result$CoefficientHistory,
                no_non_zero = nzc))
}


mtool.MNlogisticAcc <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = FALSE,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                learning_rate = 1e-4,
                                momentum_gamma = 0.9, #  momentum
                                pos = FALSE,          #  Positivity constraint
                                tolerance = 1e-4,
                                niter_inner_mtplyr = 2,
                                maxit = 100, ncores = -1,
                                group_id = NULL, group_weights = NULL, # etaG
                                groups = NULL, groups_var = NULL,     # grp, grpV
                                own_variables = NULL, N_own_variables = NULL,
                                param_start = NULL, verbose = TRUE,
                                intercept = TRUE) {
    
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
                                  learning_rate = as.double(learning_rate),
                                  momentum_gamma = as.double(momentum_gamma), 
                                  tolerance = as.double(tolerance),
                                  niter_inner = as.integer(niter_inner_mtplyr * nx),
                                  maxit = as.integer(maxit),
                                  ncores = as.integer(ncores),
                                  pos = as.logical(pos), 
                                  param_start = param_start,
                                  verbose = verbose,
                                  add_intercept = intercept)    
    
    if (inherits(result$`Sparse Estimates`, "sparseMatrix")) {
        nzc <- Matrix::nnzero(result$`Sparse Estimates`)
    } else {
        nzc <- sum(result$`Sparse Estimates` != 0)
    }
    
    return(list(coefficients = result$`Estimates`, 
                coefficients_sparse = result$`Sparse Estimates`, 
                coefficientshist = result$CoefficientHistory,
                no_non_zero = nzc))
}



mtool.MNlogisticSAHRA <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = F,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                learning_rate = 1e-4, tolerance = 1e-4,
                                niter_inner_mtplyr = 7, maxit = 100, ncores = -1,
                                group_id, group_weights,
                                groups, groups_var,
                                own_variables, N_own_variables) {
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
    pen1 <- c("l0", "l1", "l2", "linf", "l2-not-squared",
              "elastic-net", "fused-lasso",
              "group-lasso-l2", "group-lasso-linf",
              "sparse-group-lasso-l2", "sparse-group-lasso-linf",
              "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
              "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2 <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3 <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    if (regularization %in% pen1) { penalty <- 1 }
    if (regularization %in% pen2) { penalty <- 2 }
    if (regularization %in% pen3) { penalty <- 3 }
    if (! regularization %in% c(pen1, pen2, pen3)) {
        stop('The provided regularization is not supported.')
    }
    
    ### check regularization-specific inputs
    #### penalty = 1, call proximal(Flat), requires `group_id` in integer vector
    if (penalty == 1) {
        if (missing(group_id)) { group_id <- rep(0L, p) }
        group_weights <- vector(mode = 'double')
        groups <- matrix(NA)
        groups_var <- matrix(NA)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 2, call proximalGraph
    if (penalty == 2) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(groups_var)) { stop('Required input `groups_var` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        own_variables <- vector(mode = 'integer')
        N_own_variables <- vector(mode = 'integer')
    }
    
    #### penalty = 3, call proximalGraph
    if (penalty == 3) {
        if (missing(groups)) { stop('Required input `groups` is missing.') }
        if (missing(own_variables)) { stop('Required input `own_variables` is missing.') }
        if (missing(N_own_variables)) { stop('Required input `N_own_variables` is missing.') }
        if (missing(group_weights)) { stop('Required input `group_weights` is missing.') }
        group_id <- rep(0L, p)
        groups_var <- matrix(NA)
    }
    
    ## call mtool main function
    result <- MultinomLogisticSARAH(X = X, Y = Y, offset = offset, K = K, reg_p = p - N_covariates,
                                  penalty = penalty, regul = regularization, transpose = transpose,
                                  grp_id = group_id, etaG = group_weights,
                                  grp = groups, grpV = groups_var,
                                  own_var = own_variables, N_own_var = N_own_variables,
                                  lam1 = lambda1, lam2 = lambda2, lam3 = lambda3,
                                  learning_rate = learning_rate, tolerance = tolerance,
                                  niter_inner = niter_inner_mtplyr * nx, maxit = maxit,
                                  ncores = ncores)
    nzc <- length(result$`Sparse Estimates`@i)
    return(list(coefficients = result$`Sparse Estimates`,
                coefficientshist = result$CoefficientHistory,
                no_non_zero = nzc))
}





mtool.MNlogisticPCD <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = FALSE,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                learning_rate = 1e-4, tolerance = 1e-4,
                                maxit = 100, ncores = -1, pos = FALSE, # Removed niter_inner_mtplyr, added pos
                                group_id = NULL, group_weights = NULL,
                                groups = NULL, groups_var = NULL,
                                own_variables = NULL, N_own_variables = NULL) {
    
    ## --- Dimensions and checks ---
    if (!is.matrix(X)) X <- as.matrix(X)
    nx <- nrow(X)
    
    if (!is.vector(Y)) {Y <- as.vector(Y)}
    ny <- length(Y)
    
    if (!is.vector(offset)) {offset <- as.vector(offset)}
    noff <- length(offset)
    
    if (nx == ny && nx == noff) {
        n_obs <- nx # Use n_obs to avoid conflict with ncores if passed directly
    } else {
        stop('X, Y and offset have different number of observations.')
    }
    
    p_total <- ncol(X) # Use p_total for clarity
    
    # Corrected K Calculation
    valid_Y_values <- Y[!is.na(Y) & Y > 0] # Exclude 0s or other non-class values
    if (length(valid_Y_values) == 0) {
        stop("Y does not contain any valid positive class labels (expected 1 to K).")
    }
    unique_classes <- unique(valid_Y_values)
    K_val <- length(unique_classes)
    if (K_val > 0 && !all(sort(unique_classes) == 1:K_val)) {
        warning(paste("Classes in Y are:", paste(sort(unique_classes), collapse=", "),
                      ". K is set to", K_val,
                      ". Ensure C++ handles these actual Y values (e.g., if they are not 1 to K_val consecutive and Y values map to 0..K-1 indices)."))
    }
    if (K_val == 0) stop("Could not determine number of classes K from Y.")
    
    
    ## --- Regularization setup ---
    pen1_types <- c("l0", "l1", "l2", "linf", "l2-not-squared",
                    "elastic-net", "fused-lasso",
                    "group-lasso-l2", "group-lasso-linf",
                    "sparse-group-lasso-l2", "sparse-group-lasso-linf",
                    "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
                    "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2_types <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3_types <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    penalty_code_val <- 0 # Use new var name
    if (regularization %in% pen1_types) { penalty_code_val <- 1 }
    else if (regularization %in% pen2_types) { penalty_code_val <- 2 }
    else if (regularization %in% pen3_types) { penalty_code_val <- 3 }
    
    if (penalty_code_val == 0 && regularization != "none") {
        stop('The provided regularization is not supported by this mapping.')
    }
    if (regularization == "none" && penalty_code_val == 0) penalty_code_val <- 1 # Default to flat penalty context
    
    ## --- Grooming penalty-specific inputs ---
    # Provide sensible defaults if arguments are NULL (missing() is for formal args)
    if (is.null(group_id)) { group_id <- rep(0L, p_total) }
    if (is.null(group_weights)) { group_weights <- vector(mode = 'double') }
    if (is.null(groups)) { groups <- matrix(NA_real_, nrow = 0, ncol = 0) } # Empty matrix
    if (is.null(groups_var)) { groups_var <- matrix(NA_real_, nrow = 0, ncol = 0) }
    if (is.null(own_variables)) { own_variables <- vector(mode = 'integer') }
    if (is.null(N_own_variables)) { N_own_variables <- vector(mode = 'integer') }
    
    if (penalty_code_val == 1) {
        # group_id already defaulted. Ensure others are valid empty types if C++ expects them.
        # For matrix arguments, ensure they are actual matrices even if empty.
        if (!is.matrix(groups) || ncol(groups) == 0) groups <- matrix(NA_real_, nrow = 1, ncol = 0) # Or some valid empty matrix
        if (!is.matrix(groups_var) || ncol(groups_var) == 0) groups_var <- matrix(NA_real_, nrow = 1, ncol = 0)
    } else if (penalty_code_val == 2) {
        if (is.null(groups) || nrow(groups) == 0) stop('Required input `groups` is missing for penalty_code=2.')
        if (is.null(groups_var) || nrow(groups_var) == 0) stop('Required input `groups_var` is missing for penalty_code=2.')
        if (is.null(group_weights) || length(group_weights) == 0) stop('Required input `group_weights` (etaG) is missing for penalty_code=2.')
    } else if (penalty_code_val == 3) {
        if (is.null(groups) || nrow(groups) == 0) stop('Required input `groups` is missing for penalty_code=3.')
        if (is.null(own_variables) || length(own_variables) == 0) stop('Required input `own_variables` is missing for penalty_code=3.')
        if (is.null(N_own_variables) || length(N_own_variables) == 0) stop('Required input `N_own_variables` is missing for penalty_code=3.')
        if (is.null(group_weights) || length(group_weights) == 0) stop('Required input `group_weights` (etaG) is missing for penalty_code=3.')
    }
    
    reg_p_val <- as.integer(p_total - N_covariates)
    if(reg_p_val < 0) stop("N_covariates cannot be greater than total number of predictors.")
    
    ## Call C++ PCD function (Ensure C++ function is named MultinomLogisticExp_PCD_BlockRow)
    result <- MultinomLogisticPCD( # Match C++ exported function name
        X = X, Y = as.integer(Y), offset_vec = as.double(offset),
        K = as.integer(K_val), reg_p = reg_p_val,
        penalty_code = as.integer(penalty_code_val),
        regul = as.character(regularization),
        transpose = as.logical(transpose),
        grp_id = as.integer(group_id),
        etaG = as.double(group_weights), # R: group_weights -> C++: etaG
        grp_mat = as.matrix(groups),      # R: groups -> C++: grp_mat
        grpV_mat = as.matrix(groups_var), # R: groups_var -> C++: grpV_mat
        own_var = as.integer(own_variables),
        N_own_var = as.integer(N_own_variables),
        lam1 = as.double(lambda1),
        lam2 = as.double(lambda2),
        lam3 = as.double(lambda3),
        learning_rate = as.double(learning_rate),
        tolerance = as.double(tolerance),
        maxit = as.integer(maxit),
        ncores = as.integer(ncores),
        pos = as.logical(pos)
    )
    
    # Assuming result$`Sparse Estimates` is an Armadillo sp_mat converted by Rcpp to Matrix::dgCMatrix
    if (inherits(result$`Sparse Estimates`, "sparseMatrix")) {
        nzc <- Matrix::nnzero(result$`Sparse Estimates`)
    } else if (is.matrix(result$`Sparse Estimates`)) { # Fallback if it's a dense arma::mat
        nzc <- sum(result$`Sparse Estimates` != 0)
    } else {
        nzc <- NA # Cannot determine
    }
    
    return(list(coefficients = result$`Estimates`, # Keep returning dense for primary
                coefficients_sparse = result$`Sparse Estimates`,
                coefficientshist = result$CoefficientHistory,
                no_non_zero = nzc))
}


mtool.MNlogisticCCD <- function(X, Y, offset, N_covariates,
                                regularization = 'l1', transpose = FALSE,
                                lambda1, lambda2 = 0, lambda3 = 0,
                                learning_rate_scale = 1.0, # Not used by glmnet-style CCD
                                tolerance = 1e-5,
                                maxit = 100, ncores = 1, pos = FALSE,
                                kkt_abs_check_tol = 1e-6,
                                max_ccd_passes_active_set = 10,
                                group_id = NULL, group_weights = NULL,
                                groups = NULL, groups_var = NULL,
                                own_variables = NULL, N_own_variables = NULL) {
    
    ## --- Dimensions and checks ---
    if (!is.matrix(X)) X <- as.matrix(X)
    if (!is.vector(Y)) Y <- as.vector(Y)
    if (!is.vector(offset)) offset <- as.vector(offset)
    
    if (nrow(X) != length(Y) || nrow(X) != length(offset)) {
        stop('X, Y and offset have different number of observations.')
    }
    n_obs <- nrow(X)
    p_total <- ncol(X)
    
    ## --- Robust Handling of Class Labels (Y) ---
    # Convert Y to a factor to handle any type of label (numeric, character, etc.)
    # and map it to zero-based integers for C++.
    if (!is.factor(Y)) Y_factor <- factor(Y)
    else Y_factor <- Y
    
    Y_cpp <- as.integer(Y_factor) - 1 # C++ expects 0 to K-1
    K_classes <- nlevels(Y_factor)
    
    if (K_classes <= 1) {
        stop("The response variable Y must have at least 2 unique classes.")
    }
    
    message(paste("Detected", K_classes, "classes. Using '", levels(Y_factor)[1], "' as the reference class (level 0).", sep=""))
    
    ## --- Regularization setup ---
    # This section maps the string to a code, which is useful if the C++
    # backend handles multiple penalty families. The current CCD implements elastic net.
    pen1_types <- c("l0", "l1", "l2", "linf", "elastic-net", "fused-lasso",
                    "group-lasso-l2", "group-lasso-linf",
                    "sparse-group-lasso-l2", "sparse-group-lasso-linf",
                    "l1l2", "l1linf", "l1l2+l1", "l1linf+l1", "l1linf-row-column",
                    "trace-norm", "trace-norm-vec", "rank", "rank-vec", "none")
    pen2_types <- c("graph", "graph-ridge", "graph-l2", "multi-task-graph")
    pen3_types <- c("tree-l0", "tree-l2", "tree-linf", "multi-task-tree")
    
    penalty_code_val <- 0
    if (regularization %in% pen1_types) { penalty_code_val <- 1 }
    else if (regularization %in% pen2_types) { penalty_code_val <- 2 }
    else if (regularization %in% pen3_types) { penalty_code_val <- 3 }
    
    if (penalty_code_val == 0) {
        stop('The provided regularization type is not recognized.')
    }
    
    # Number of regularized predictors
    reg_p_val <- as.integer(p_total - N_covariates)
    if(reg_p_val < 0 || reg_p_val > p_total) {
        stop("N_covariates must be between 0 and the total number of predictors.")
    }
    
    ## --- Call C++ CCD function ---
    result <- MultinomLogisticCCD(
        X = X,
        Y = Y_cpp,
        offset_vec = as.double(offset),
        K_classes = as.integer(K_classes),
        reg_p = reg_p_val,
        penalty_code = as.integer(penalty_code_val),
        regul = as.character(regularization),
        transpose_prox_input = as.logical(transpose),
        grp_id = if (is.null(group_id)) integer(0) else as.integer(group_id),
        etaG = if (is.null(group_weights)) numeric(0) else as.double(group_weights),
        grp_mat = if (is.null(groups)) matrix(0,0,0) else as.matrix(groups),
        grpV_mat = if (is.null(groups_var)) matrix(0,0,0) else as.matrix(groups_var),
        own_var = if (is.null(own_variables)) integer(0) else as.integer(own_variables),
        N_own_var = if (is.null(N_own_variables)) integer(0) else as.integer(N_own_variables),
        lam1 = as.double(lambda1),
        lam2 = as.double(lambda2),
        lam3 = as.double(lambda3),
        learning_rate_scale = as.double(learning_rate_scale),
        tolerance = as.double(tolerance),
        kkt_abs_check_tol = as.double(kkt_abs_check_tol),
        maxit = as.integer(maxit),
        max_ccd_passes_active_set = as.integer(max_ccd_passes_active_set),
        ncores = as.integer(ncores),
        pos = as.logical(pos)
    )
    
    # --- Format and return results ---
    # Set column names for the K-1 non-reference classes
    class_names <- levels(Y_factor)[-1]
    if (ncol(result$Estimates) == length(class_names)) {
        colnames(result$Estimates) <- class_names
        if (inherits(result$`SparseEstimates`, "Matrix")) {
            colnames(result$`SparseEstimates`) <- class_names
        }
    }
    
    return(list(
        coefficients = result$Estimates,
        coefficients_sparse = result$SparseEstimates
    ))
}


mtool.proximalSCAD <- function(U,
                               p,
                               K,
                               lambda,
                               a = 3.7,
                               pos = FALSE,
                               transpose = FALSE) {
    
    proximalSCAD(U,
                 p,
                 K,
                 lambda,
                 a,
                 pos,
                 transpose)
    
}