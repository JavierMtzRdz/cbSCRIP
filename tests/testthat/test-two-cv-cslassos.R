test_that("two.cv.CSlassos runs and makes predictions", {
    set.seed(123)
    n <- 150
    p <- 10
    data <- cbSCRIP::gen_data(n_train = n, p = p, num_true = 3, setting = 1)
    train <- data$train

    expect_no_error({
        fit <- two.cv.CSlassos(
            data = train,
            nfold = 3,
            var_time = "ftime",
            var_status = "fstatus",
            lambda.select = "lambda.min"
        )
    })

    expect_s3_class(fit, "twoCSlassos")
    expect_true(!is.null(fit$cv_models))
    expect_length(fit$lambdas, 2)
    
    # Try prediction risk
    times_eval <- c(0.2, 0.5, 0.8)
    expect_no_error({
        pred <- predictRisk(fit, newdata = train, times = times_eval, cause = 1)
    })
    expect_equal(dim(pred), c(n, length(times_eval)))
})
