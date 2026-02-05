# Optimizer Benchmark


``` r
devtools::load_all(".")

pacman::p_load(survival, bench, ggplot2, dplyr, tidyr)
```

## Overview

This benchmark compares the three optimization algorithms available in
`cbSCRIP`:

- **CCD**: Cyclical Coordinate Descent
- **SAGA**: Stochastic Average Gradient with Nesterov acceleration
- **SVRG**: Stochastic Variance Reduced Gradient

We evaluate performance across different dataset sizes measuring:

- **Execution time**
- **Memory allocation**

## Data Generation

``` r
set.seed(42)

# Define benchmark configurations
configs <- tibble::tibble(
  n = c(200, 500, 1000),
  p = c(20, 50, 100),
  config_name = paste0("n=", n, ", p=", p)
)

# Generate datasets
datasets <- purrr::map2(configs$n, configs$p, function(n, p) {
  gen_data(n_train = n, p = p, num_true = 5, setting = 1)$train
})
names(datasets) <- configs$config_name
```

## Benchmark Function

``` r
run_optimizer_benchmark <- function(train_data, optimizer, nlambda = 10) {
  cbSCRIP(
    Surv(ftime, fstatus) ~ .,
    data = train_data,
    nlambda = nlambda,
    optimizer = optimizer,
    ratio = 20,
    maxit = 100,
    coeffs = "original"
  )
}
```

## Running Benchmarks

``` r
optimizers <- c("CCD", "SAGA", "SVRG")

# Run benchmarks using bench::press to standardise output
benchmark_results <- bench::press(
  config_name = configs$config_name,
  {
    data <- datasets[[config_name]]
    bench::mark(
      CCD = run_optimizer_benchmark(data, "CCD"),
      SAGA = run_optimizer_benchmark(data, "SAGA"),
      SVRG = run_optimizer_benchmark(data, "SVRG"),
      iterations = 10,
      check = FALSE,
      memory = TRUE
    )
  }
)

# Ensure proper ordering of configurations
benchmark_results <- benchmark_results |>
  mutate(config_name = factor(config_name, levels = configs$config_name))

benchmark_results
```

    #> # A tibble: 9 × 7
    #>   expression config_name        min   median `itr/sec` mem_alloc `gc/sec`
    #>   <bch:expr> <fct>         <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
    #> 1 CCD        n=200, p=20   297.84ms 310.21ms    3.22      22.1MB   4.83  
    #> 2 SAGA       n=200, p=20      4.72s    4.89s    0.204     22.5MB   0.204 
    #> 3 SVRG       n=200, p=20      6.47s    7.32s    0.138     21.3MB   0.207 
    #> 4 CCD        n=500, p=50   884.05ms 961.92ms    0.989     92.3MB   2.18  
    #> 5 SAGA       n=500, p=50     13.01s   14.04s    0.0708      93MB   0.156 
    #> 6 SVRG       n=500, p=50     15.17s      16s    0.0622    93.1MB   0.143 
    #> 7 CCD        n=1000, p=100    5.06s    9.64s    0.112    360.8MB   0.604 
    #> 8 SAGA       n=1000, p=100    1.07m    1.24m    0.0136   361.7MB   0.0584
    #> 9 SVRG       n=1000, p=100   44.53s   47.19s    0.0205   361.6MB   0.0839

## Results

### Execution Time

``` r
autoplot(benchmark_results, type = "boxplot") +
  facet_wrap(~config_name, scales = "free_x") +
  labs(
    title = "Execution Time Distribution",
    subtitle = "Higher density / lower values are better",
    x = "Time"
  ) +
  theme(legend.position = "none")
```

<img src="optimizer_benchmark_files/figure-commonmark/time-plot-1.png"
data-fig-align="center" />

### Memory Allocation

``` r
benchmark_results |>
  mutate(
    mem_mb = as.numeric(mem_alloc) / 1e6
  ) |>
  ggplot(aes(x = config_name, y = mem_mb, fill = as.character(expression))) +
  geom_col(position = position_dodge(width = 1)) +
  labs(
    title = "Memory Allocation by Optimizer",
    x = "Setting",
    y = "Memory Allocated (MB)",
    fill = "Optimizer"
  )
```

<img src="optimizer_benchmark_files/figure-commonmark/memory-plot-1.png"
data-fig-align="center" />

### Summary Table

``` r
benchmark_results |>
  mutate(
    time_sec = round(as.numeric(median), 3),
    mem_mb = round(as.numeric(mem_alloc) / 1e6, 2)
  ) |>
  select(Setting = config_name, Optimizer = expression, 
         `Time (s)` = time_sec, `Memory (MB)` = mem_mb) |>
  knitr::kable()
```

| Setting       | Optimizer | Time (s) | Memory (MB) |
|:--------------|:----------|---------:|------------:|
| n=200, p=20   | CCD       |    0.310 |       23.21 |
| n=200, p=20   | SAGA      |    4.893 |       23.60 |
| n=200, p=20   | SVRG      |    7.318 |       22.36 |
| n=500, p=50   | CCD       |    0.962 |       96.79 |
| n=500, p=50   | SAGA      |   14.037 |       97.48 |
| n=500, p=50   | SVRG      |   15.998 |       97.63 |
| n=1000, p=100 | CCD       |    9.640 |      378.35 |
| n=1000, p=100 | SAGA      |   74.103 |      379.24 |
| n=1000, p=100 | SVRG      |   47.191 |      379.15 |

## Session Info

``` r
sessionInfo()
```

    #> R version 4.5.1 (2025-06-13)
    #> Platform: aarch64-apple-darwin20
    #> Running under: macOS Tahoe 26.2
    #> 
    #> Matrix products: default
    #> BLAS:   /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRblas.0.dylib 
    #> LAPACK: /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.1
    #> 
    #> locale:
    #> [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    #> 
    #> time zone: America/Vancouver
    #> tzcode source: internal
    #> 
    #> attached base packages:
    #> [1] stats     graphics  grDevices utils     datasets  methods   base     
    #> 
    #> other attached packages:
    #>  [1] tidyr_1.3.2         dplyr_1.1.4         bench_1.1.4        
    #>  [4] survival_3.8-3      cbSCRIP_0.1.1       testthat_3.3.0     
    #>  [7] mytidyfunctions_0.1 ggplot2_4.0.1       devtools_2.4.6     
    #> [10] usethis_3.2.1       pacman_0.5.1       
    #> 
    #> loaded via a namespace (and not attached):
    #>   [1] RColorBrewer_1.1-3        RcppArmadillo_15.2.3-1   
    #>   [3] rstudioapi_0.18.0         jsonlite_2.0.0           
    #>   [5] shape_1.4.6.1             magrittr_2.0.4           
    #>   [7] TH.data_1.1-5             farver_2.1.2             
    #>   [9] rmarkdown_2.30            fs_1.6.6                 
    #>  [11] vctrs_0.7.1.9000          memoise_2.0.1            
    #>  [13] base64enc_0.1-3           htmltools_0.5.9          
    #>  [15] polspline_1.1.25          Formula_1.2-5            
    #>  [17] pROC_1.19.0.1             caret_7.0-1              
    #>  [19] parallelly_1.46.1         htmlwidgets_1.6.4        
    #>  [21] desc_1.4.3                plyr_1.8.9               
    #>  [23] sandwich_3.1-1            zoo_1.8-15               
    #>  [25] lubridate_1.9.4           cachem_1.1.0             
    #>  [27] lifecycle_1.0.5           cmprsk_2.2-12            
    #>  [29] iterators_1.0.14          pkgconfig_2.0.3          
    #>  [31] Matrix_1.7-4              R6_2.6.1                 
    #>  [33] fastmap_1.2.0             future_1.69.0            
    #>  [35] digest_0.6.39             numDeriv_2016.8-1.1      
    #>  [37] colorspace_2.1-2          furrr_0.3.1              
    #>  [39] rprojroot_2.1.1           pkgload_1.4.1            
    #>  [41] Hmisc_5.2-5               labeling_0.4.3           
    #>  [43] progressr_0.18.0          timechange_0.4.0         
    #>  [45] riskRegression_2025.09.17 mgcv_1.9-4               
    #>  [47] compiler_4.5.1            remotes_2.5.0            
    #>  [49] withr_3.0.2               htmlTable_2.4.3          
    #>  [51] S7_0.2.1                  backports_1.5.0          
    #>  [53] pkgbuild_1.4.8            MASS_7.3-65              
    #>  [55] lava_1.8.2                quantreg_6.1             
    #>  [57] sessioninfo_1.2.3         ModelMetrics_1.2.2.2     
    #>  [59] tools_4.5.1               foreign_0.8-90           
    #>  [61] otel_0.2.0                future.apply_1.20.1      
    #>  [63] nnet_7.3-20               glue_1.8.0               
    #>  [65] mets_1.3.9                profmem_0.7.0            
    #>  [67] nlme_3.1-168              grid_4.5.1               
    #>  [69] checkmate_2.3.3           cluster_2.1.8.1          
    #>  [71] reshape2_1.4.5            generics_0.1.4           
    #>  [73] recipes_1.3.1             gtable_0.3.6             
    #>  [75] class_7.3-23              data.table_1.18.2.1      
    #>  [77] utf8_1.2.6                foreach_1.5.2            
    #>  [79] pillar_1.11.1             stringr_1.6.0            
    #>  [81] splines_4.5.1             lattice_0.22-7           
    #>  [83] SparseM_1.84-2            tidyselect_1.2.1         
    #>  [85] rms_8.1-0                 knitr_1.51               
    #>  [87] gridExtra_2.3             stats4_4.5.1             
    #>  [89] xfun_0.56                 casebase_0.10.6          
    #>  [91] hardhat_1.4.2             timeDate_4052.112        
    #>  [93] brio_1.1.5                stringi_1.8.7            
    #>  [95] VGAM_1.1-14               yaml_2.3.12              
    #>  [97] pec_2025.06.24            evaluate_1.0.5           
    #>  [99] codetools_0.2-20          tibble_3.3.1             
    #> [101] cli_3.6.5                 rpart_4.1.24             
    #> [103] Rcpp_1.1.1                globals_0.18.0           
    #> [105] parallel_4.5.1            MatrixModels_0.5-4       
    #> [107] ellipsis_0.3.2            gower_1.0.2              
    #> [109] listenv_0.10.0            glmnet_4.1-10            
    #> [111] mvtnorm_1.3-3             timereg_2.0.7            
    #> [113] ipred_0.9-15              scales_1.4.0             
    #> [115] prodlim_2025.04.28        purrr_1.2.1              
    #> [117] crayon_1.5.3              rlang_1.1.7              
    #> [119] multcomp_1.4-29
