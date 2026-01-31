test_that("cbSCRIP runs with MNlogisticCCD", {
    set.seed(123)
    n <- 200
    p <- 20
    data <- cbSCRIP::gen_data(n_train = n, p = p, num_true = 5, setting = 1)
    train <- data$train

    # Fit model
    expect_no_error({
        fit <- cbSCRIP(
            survival::Surv(ftime, fstatus) ~ .,
            data = train,
            nlambda = 10,
            maxit = 100,
            optimizer = "CCD",
            ratio = 20
        )
    })

    expect_s3_class(fit, "cbSCRIP.path")
    expect_true(length(fit$lambdagrid) > 0)
})

test_that("cbSCRIP runs with MNlogisticSAGAN", {
    set.seed(123)
    n <- 200
    p <- 20
    data <- cbSCRIP::gen_data(n_train = n, p = p, num_true = 5, setting = 1)
    train <- data$train

    # Fit model
    expect_no_error({
        fit <- cbSCRIP(
            survival::Surv(ftime, fstatus) ~ .,
            data = train,
            nlambda = 10,
            maxit = 100,
            optimizer = "SAGA",
            ratio = 20
        )
    })

    expect_s3_class(fit, "cbSCRIP.path")
})

test_that("cbSCRIP runs with MNlogisticSVRG", {
    set.seed(123)
    n <- 200
    p <- 20
    data <- cbSCRIP::gen_data(n_train = n, p = p, num_true = 5, setting = 1)
    train <- data$train

    # Fit model
    expect_no_error({
        fit <- cbSCRIP(
            survival::Surv(ftime, fstatus) ~ .,
            data = train,
            nlambda = 10,
            maxit = 100,
            optimizer = "SVRG",
            ratio = 20
        )
    })

    expect_s3_class(fit, "cbSCRIP.path")
})
