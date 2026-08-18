test_that("non_zero count excludes intercept and log(time)", {
    # mock coefficient matrix
    coef_mat <- matrix(c(
        0.0, 0.0,
        0.5, 0.0,
        0.0, 0.3,
        0.1, 0.2,
        1.0, 1.0,
        0.5, 0.5
    ), nrow = 6, ncol = 2, byrow = TRUE)
    rownames(coef_mat) <- c("X1", "X2", "X3", "X4", "(Intercept)", "log(time)")

    # 3 variables selected (X2, X3, X4), excluding intercept and log(time)
    penalized_rows <- !rownames(coef_mat) %in% c("(Intercept)", "log(time)")
    n_selected <- sum(rowSums(abs(coef_mat[penalized_rows, , drop = FALSE])) > 1e-10)

    expect_equal(n_selected, 3)
})

test_that("cbSCRIP.path non_zero counts only penalized variables", {
    set.seed(42)
    n <- 150
    p <- 15
    data <- cbSCRIP::gen_data(n_train = n, p = p, num_true = 5, setting = 1)
    train <- data$train

    fit <- cbSCRIP(
        survival::Surv(ftime, fstatus) ~ .,
        data = train,
        nlambda = 5,
        maxit = 50,
        optimizer = "CCD",
        ratio = 20
    )

    nz_values <- unlist(fit$non_zero)
    expect_true(all(nz_values >= 0))
    expect_true(all(nz_values <= p))
})
