
# Test Script for cbSCRIP Optimization

library(cbSCRIP)
library(survival)

# 1. Generate Data
cat("Generating data...\n")
set.seed(123)
sim_data <- gen_data(n = 200, p = 20, num_true = 5, setting = 1)
data <- sim_data$train
formula <- Surv(ftime, fstatus) ~ .

# 2. Test cbSCRIP with Default Solver (SAGA)
cat("\nTesting cbSCRIP with MNlogisticSAGAN (Default)...\n")
start_time <- Sys.time()
fit_saga <- cbSCRIP(formula, data, nlambda = 10, n_unpenalized = 0)
end_time <- Sys.time()
cat("SAGA Time:", end_time - start_time, "\n")
print(fit_saga)

# 3. Test cbSCRIP with CCD Solver
cat("\nTesting cbSCRIP with MNlogisticCCD...\n")
start_time <- Sys.time()
fit_ccd <- cbSCRIP(formula, data, nlambda = 10, n_unpenalized = 0, fit_fun = MNlogisticCCD)
end_time <- Sys.time()
cat("CCD Time:", end_time - start_time, "\n")
print(fit_ccd)

# 4. Test Cross-Validation
cat("\nTesting cv_cbSCRIP...\n")
start_time <- Sys.time()
cv_fit <- cv_cbSCRIP(formula, data, nfold = 3, nlambda = 10, n_unpenalized = 0)
end_time <- Sys.time()
cat("CV Time:", end_time - start_time, "\n")
print(cv_fit)

cat("\nTest Complete.\n")
