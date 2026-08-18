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

test_that("cbSCRIP runs with MNlogisticFISTA", {
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
            optimizer = "FISTA",
            ratio = 20
        )
    })

    expect_s3_class(fit, "cbSCRIP.path")
})


test_that("CV folds keep every subject on one side", {
    set.seed(123)
    d <- cbSCRIP::gen_data(n_train = 150, p = 12, num_true = 4, setting = 2)
    cb <- create_cb_data(survival::Surv(ftime, fstatus) ~ ., d$train, ratio = 20)

    folds <- make_cv_folds(cb, 5)
    expect_equal(sort(unlist(folds, use.names = FALSE)), seq_along(cb$event))
    for (f in folds) expect_false(any(cb$id[f] %in% cb$id[-f]))
    for (k in unique(cb$event[cb$event != 0])) {
        expect_true(all(vapply(folds, function(f) sum(cb$event[f] == k), 0) > 0))
    }

    cb$id <- NULL
    expect_error(make_cv_folds(cb, 5), "no subject")
})
