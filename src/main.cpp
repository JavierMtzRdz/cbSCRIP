
#include "RcppArmadillo.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double scalar_scad_prox(double val, double lambda, double a = 3.7) {
  if (lambda <= 0.0)
    return val;

  double abs_val = std::abs(val);
  if (a <= 2.0)
    a = 3.7;

  if (abs_val <= lambda) {
    return 0.0;
  } else if (abs_val <= 2.0 * lambda) {
    return std::copysign(std::max(0.0, abs_val - lambda), val);
  } else if (abs_val <= a * lambda) {
    return ((a - 1.0) * val - std::copysign(a * lambda, val)) / (a - 2.0);
  } else {
    return val;
  }
}

double prox_elastic_net_scalar(double val, double lam1, double lam2) {
  if (lam1 <= 0.0 && lam2 <= 0.0)
    return val;

  double abs_val = std::abs(val);
  if (abs_val <= lam1)
    return 0.0;

  double st = std::copysign(abs_val - lam1, val);
  return st / (1.0 + lam2);
}

arma::vec prox_scad_vec(const arma::vec &vals, double lambda, double a = 3.7) {
  arma::vec result = vals;
  result.for_each([&](double &val) { val = scalar_scad_prox(val, lambda, a); });
  return result;
}

arma::vec prox_elastic_net_vec(const arma::vec &vals, double lam1,
                               double lam2) {
  arma::vec result = vals;
  result.for_each(
      [&](double &val) { val = prox_elastic_net_scalar(val, lam1, lam2); });
  return result;
}

// Helper to compute probabilities and negative log-likelihood
double compute_obj_and_probs(const arma::mat &X, const arma::vec &Y,
                             const arma::vec &offset, const arma::mat &param,
                             arma::mat &P, int K,
                             const std::vector<int> &active_set) {
  int n = X.n_rows;
  double obj = 0.0;

  // Linear predictor: Eta = X * param
  arma::mat Eta = X * param; // n x K

  // Add offset
  Eta.each_col() += offset;

  // Softmax with stability
  for (int i = 0; i < n; ++i) {
    double max_val = 0.0; // Baseline class 0 has score 0
    for (int k = 0; k < K; ++k) {
      if (Eta(i, k) > max_val)
        max_val = Eta(i, k);
    }

    double sum_exp = std::exp(0.0 - max_val); // Baseline
    for (int k = 0; k < K; ++k) {
      double e = std::exp(Eta(i, k) - max_val);
      P(i, k) = e;
      sum_exp += e;
    }

    // Normalize
    double inv_sum = 1.0 / sum_exp;
    for (int k = 0; k < K; ++k) {
      P(i, k) *= inv_sum;
    }

    // Objective: -log(p_y)
    int yi = static_cast<int>(Y(i));
    double log_sum_exp = max_val + std::log(sum_exp); // = log(sum(exp(eta)))

    if (yi == 0) {
      obj += log_sum_exp; // - (0)
    } else if (yi > 0 && yi <= K) {
      obj += log_sum_exp -
             (Eta(i, yi - 1)); // Note: Eta is 0-indexed for classes 1..K
    }
  }

  return obj / n;
}


// Helper to compute objective directly from Eta (to save X * beta operations)
double compute_obj_from_eta(const arma::mat &Eta, const arma::vec &Y, arma::mat &P, int K) {
  int n = Eta.n_rows;
  double obj = 0.0;
  for (int i = 0; i < n; ++i) {
    double max_eta = 0.0;
    for (int k = 0; k < K; ++k) {
      if (Eta(i, k) > max_eta) max_eta = Eta(i, k);
    }
    double sum_exp = std::exp(-max_eta); // For class 0
    for (int k = 0; k < K; ++k) {
      P(i, k) = std::exp(Eta(i, k) - max_eta);
      sum_exp += P(i, k);
    }
    double inv_sum = 1.0 / sum_exp;
    for (int k = 0; k < K; ++k) {
      P(i, k) *= inv_sum;
    }
    int yi = static_cast<int>(Y(i));
    double log_sum_exp = max_eta + std::log(sum_exp);
    if (yi == 0) {
      obj += log_sum_exp;
    } else if (yi > 0 && yi <= K) {
      obj += log_sum_exp - Eta(i, yi - 1);
    }
  }
  return obj / n;
}

// [[Rcpp::export]]
Rcpp::List MultinomLogisticCCD(
    const arma::mat &X, const arma::vec &Y, const arma::vec &offset, int K,
    int reg_p, int penalty = 1, double lam1 = 0.0, double lam2 = 0.0,
    double tolerance = 1e-7, int maxit = 5000,
    double lr_adj = 1.0, // Unused in Newton, kept for compatibility
    bool verbose = false, bool pos = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue,
    double lam1_prev = 0.0) {  // previous lambda1 for Sequential Strong Rules

  int n = X.n_rows;
  int p = X.n_cols;

  if (reg_p < 0)
    reg_p = 0;
  if (reg_p > p)
    reg_p = p;

  if (verbose) {
    Rcpp::Rcout << "CCD: p=" << p << ", reg_p=" << reg_p
                << ", unpenalized vars: " << (p - reg_p) << std::endl;
  }

  // Initialize parameters
  arma::mat param(p, K, arma::fill::zeros);
  if (param_start.isNotNull()) {
    param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
  }

  // 1. Initial Active Set Strategy
  std::vector<int> active_set;
  active_set.reserve(p);

  bool is_cold_start = param_start.isNull();

  // Warm start non-zeros MUST be in active set
  for (int j = 0; j < p; ++j) {
    bool is_nonzero = false;
    for (int k = 0; k < K; ++k) {
      if (std::abs(param(j, k)) > 1e-10) {
        is_nonzero = true;
        break;
      }
    }
    if (is_nonzero) {
      active_set.push_back(j);
    }
  }

  // Force unpenalized variables into active set
  for (int j = reg_p; j < p; ++j) {
    active_set.push_back(j);
  }

  // Sequential Strong Rules (SSR) for warm starts (Tibshirani et al. 2012)
  // For a warm start at lambda, a variable j that was zero at lambda_prev is
  // very unlikely to be non-zero at lambda if:
  //   |grad_j(beta(lambda_prev))| < 2 * lambda - lambda_prev
  // We apply SSR to pre-seed the active set with strong candidates.
  // This dramatically improves selection quality along the lambda path.
  if (!is_cold_start && lam1 > 0 && lam1_prev > 0) {
    // Compute residuals from the warm-start parameter
    arma::mat P_init(n, K, arma::fill::zeros);
    compute_obj_and_probs(X, Y, offset, param, P_init, K, active_set);

    arma::mat Residuals_init = P_init;
    for (int i = 0; i < n; ++i) {
      int yi = static_cast<int>(Y(i));
      if (yi > 0 && yi <= K)
        Residuals_init(i, yi - 1) -= 1.0;
    }

    // SSR threshold: 2*lambda - lambda_prev (Tibshirani et al. 2012).
    // If consecutive lambda step is large (2*lam1 <= lam1_prev), fallback to lam1
    // NEVER clamp to 0, which would include all p variables and destroy speed & sparsity.
    double ssr_threshold = 2.0 * lam1 - lam1_prev;
    if (ssr_threshold < 0.5 * lam1) {
      ssr_threshold = 0.5 * lam1;
    }

    // Build a mask of currently active variables
    std::vector<bool> is_active(p, false);
    for (int j : active_set) is_active[j] = true;

    // Batch-compute gradients for all inactive penalized variables
    std::vector<int> inactive_vars;
    inactive_vars.reserve(reg_p);
    for (int j = 0; j < reg_p; ++j)
      if (!is_active[j]) inactive_vars.push_back(j);

    if (!inactive_vars.empty()) {
      arma::uvec inactive_uvec = arma::conv_to<arma::uvec>::from(inactive_vars);
      arma::mat Grad_inactive = (X.cols(inactive_uvec).t() * Residuals_init) / n;

      for (size_t idx = 0; idx < inactive_vars.size(); ++idx) {
        // Check if any class gradient exceeds the SSR threshold
        double max_grad = arma::max(arma::abs(Grad_inactive.row(idx)));
        if (max_grad >= ssr_threshold) {
          active_set.push_back(inactive_vars[idx]);
        }
      }
    }
  } else if (is_cold_start && lam1 > 0) {
    // Exact multinomial gradient for cold-start screening:
    // Computes true class-specific gradients at beta=0 + offset
    arma::mat P_init(n, K, arma::fill::zeros);
    compute_obj_and_probs(X, Y, offset, param, P_init, K, active_set);
    arma::mat Residuals_init = P_init;
    for (int i = 0; i < n; ++i) {
      int yi = static_cast<int>(Y(i));
      if (yi > 0 && yi <= K)
        Residuals_init(i, yi - 1) -= 1.0;
    }
    if (reg_p > 0) {
      arma::mat Grad_init = (X.cols(0, reg_p - 1).t() * Residuals_init) / n;
      for (int j = 0; j < reg_p; ++j) {
        double max_grad = arma::max(arma::abs(Grad_init.row(j)));
        if (max_grad >= lam1)
          active_set.push_back(j);
      }
    }
  }

  // Ensure uniqueness and sorting
  std::sort(active_set.begin(), active_set.end());
  active_set.erase(std::unique(active_set.begin(), active_set.end()),
                   active_set.end());

  // Fallback if empty (should rarely happen with strong rules)
  if (active_set.empty()) {
    for (int j = 0; j < p; ++j)
      active_set.push_back(j);
  }

  // Probability matrix (stores P(Y=k) for k=1..K. P(Y=0) is implicit)
  arma::mat P(n, K, arma::fill::zeros);

  // Current objective
  double current_obj =
      compute_obj_and_probs(X, Y, offset, param, P, K, active_set);

  bool converged = false;
  int total_iter = 0;

  // Outer Loop: Active Set Strategy
  // 1. Optimize over Active Set until convergence
  // 2. Check KKT conditions on Inactive Set
  // 3. If violations, add to Active Set and repeat
  // 4. Else, converged.

  int max_outer_iter = 15;
  int newton_maxit = std::min(maxit, 50);
  std::vector<arma::mat> X_W_cache(K);
  for (int outer = 0; outer < max_outer_iter; ++outer) {

    // The active set only changes at the KKT check below, so build the
    // sub-design and size the work buffers once per outer pass rather than
    // reallocating them on every Newton step.
    arma::uvec active_uvec = arma::conv_to<arma::uvec>::from(active_set);
    arma::mat X_active = X.cols(active_uvec);
    for (int k = 0; k < K; ++k)
      X_W_cache[k].set_size(n, active_set.size());

    // --- Newton Loop on Active Set ---
    for (int iter = 1; iter <= newton_maxit; ++iter) {
      Rcpp::checkUserInterrupt();
      total_iter++;

      arma::mat param_old = param;
      double obj_old = current_obj;

      // 1. Compute Gradient and Hessian Diagonal (Active Set Only needed, but
      // we need P updated) We compute full residuals because P is n x K.
      arma::mat Residuals = P; // n x K
      for (int i = 0; i < n; ++i) {
        int yi = static_cast<int>(Y(i));
        if (yi > 0 && yi <= K) {
          Residuals(i, yi - 1) -= 1.0;
        }
      }

      // 2. Inner Loop: Coordinate Descent on Quadratic Surrogate
      arma::mat W = P % (1.0 - P); // n x K, element-wise

      // Refill X_W_j for all active variables to speed up DAXPY updates
      for (int k = 0; k < K; ++k) {
        for (size_t idx = 0; idx < active_set.size(); ++idx) {
           X_W_cache[k].col(idx) = X_active.col(idx) % W.col(k);
        }
      }

      // We still need the diagonal of the Hessian for all active variables.
      arma::mat H_diag(active_set.size(), K, arma::fill::zeros);
      for (size_t idx = 0; idx < active_set.size(); ++idx) {
        int j = active_set[idx];
        for (int k = 0; k < K; ++k) {
          H_diag(idx, k) = arma::dot(X_active.col(idx), X_W_cache[k].col(idx)) / n;
          
          if (penalty == 1 && j < reg_p) {
             H_diag(idx, k) += lam2;
          }
          if (H_diag(idx, k) < 1e-6) H_diag(idx, k) = 1e-6;
        }
      }

      // Working residual for gradient computation
      arma::mat Working_Residuals = Residuals; 

      // Coordinate Descent: 5 sweeps per Newton step (standard IRLS)
      int inner_maxit = 5;
      double inner_tol = 1e-6;
      arma::mat param_inner = param; // Working copy

      for (int inner = 0; inner < inner_maxit; ++inner) {
        double max_inner_diff = 0.0;

        for (size_t idx = 0; idx < active_set.size(); ++idx) {
          int j = active_set[idx];

          for (int k = 0; k < K; ++k) {
            double beta_old_jk = param_inner(j, k);
            double current_grad_jk = arma::dot(X_active.col(idx), Working_Residuals.col(k)) / n;
            double h_jk = H_diag(idx, k);

            // H_diag already carries the ridge curvature (+lam2); the matching
            // ridge gradient must be added here or lam2 cancels out of the
            // stationarity condition and the solver returns a pure lasso fit.
            if (penalty == 1 && j < reg_p) {
              current_grad_jk += lam2 * beta_old_jk;
            }

            double z = beta_old_jk - current_grad_jk / h_jk;
            double beta_new_jk = 0.0;

            if (penalty == 1) { // Elastic Net
              double thresh = lam1 / h_jk;
              if (z > thresh) beta_new_jk = z - thresh;
              else if (z < -thresh) beta_new_jk = z + thresh;
            } else { // SCAD: R sends code 4, matching apply_prox's "not 1" rule
              double a_scad = (lam2 > 2.0) ? lam2 : 3.7;
              beta_new_jk = scalar_scad_prox(z, lam1 / h_jk, a_scad);
            }

            if (j >= reg_p) beta_new_jk = z;

            double delta = beta_new_jk - beta_old_jk;
            if (std::abs(delta) > 1e-10) {
              param_inner(j, k) = beta_new_jk;
              max_inner_diff = std::max(max_inner_diff, std::abs(delta));
              
              // DAXPY update to working residuals
              // Old code: Grad_current += delta * H_col (gradient increased)
              // Working_Residuals update must produce the same sign:
              // grad_new = X^T WR / n, and we need grad_new = grad_old + delta * H_lj
              // Since H_lj = X_l^T (X_j * W) / n, adding delta * X_j * W to WR does this.
              Working_Residuals.col(k) += delta * X_W_cache[k].col(idx);
            }
          }
        }
        if (max_inner_diff < inner_tol)
          break;
      }

      // 3. Check Outer Convergence Search
      arma::mat direction = param_inner - param_old;

      // Fast Line Search: Precompute Eta_old and Eta_dir
      arma::mat Eta_old = X_active * param_old.rows(active_uvec);
      Eta_old.each_col() += offset;
      
      arma::mat Eta_dir = X_active * direction.rows(active_uvec);

      double step_size = 1.0;
      double alpha = 0.5;
      bool step_accepted = false;

      // Compute penalty only over penalized variables in the active set
      double pen_old = 0.0;
      if (penalty == 1 && reg_p > 0) {
        for (size_t idx = 0; idx < active_set.size(); ++idx) {
          int j = active_set[idx];
          if (j < reg_p) {
            for (int k = 0; k < K; ++k) {
              double b = param_old(j, k);
              pen_old += lam1 * std::abs(b) + 0.5 * lam2 * b * b;
            }
          }
        }
      }
      double current_total_obj = obj_old + pen_old;

      for (int ls = 0; ls < 20; ++ls) {
        arma::mat param_proposal = param_old + step_size * direction;
        
        // O(n K) instead of O(|S| n K)
        arma::mat Eta_proposal = Eta_old + step_size * Eta_dir;
        arma::mat P_proposal(n, K);
        double obj_proposal = compute_obj_from_eta(Eta_proposal, Y, P_proposal, K);

        double pen_proposal = 0.0;
        if (penalty == 1 && reg_p > 0) {
          // Calculate penalty only on active penalized variables to avoid full matrix operations
          for (size_t idx = 0; idx < active_set.size(); ++idx) {
            int j = active_set[idx];
            if (j < reg_p) {
              for (int k = 0; k < K; ++k) {
                double b = param_proposal(j, k);
                pen_proposal += lam1 * std::abs(b) + 0.5 * lam2 * b * b;
              }
            }
          }
        }

        // Allow slight float tolerance to accept full Newton step without false backtracking
        if (obj_proposal + pen_proposal <= current_total_obj + 1e-8) {
          param = param_proposal;
          P = P_proposal;
          current_obj = obj_proposal;
          step_accepted = true;
          break;
        }
        step_size *= alpha;
      }

      if (!step_accepted) {
        if (arma::norm(direction, "fro") < tolerance * 10.0) {
          break;
        }
        param = param_old + step_size * direction;
        arma::mat Eta_new = Eta_old + step_size * Eta_dir;
        arma::mat P_new(n, K);
        current_obj = compute_obj_from_eta(Eta_new, Y, P_new, K);
        P = P_new;
      }

      // Check Convergence of Newton Step
      double diff = arma::norm(param - param_old, "fro") /
                    (1.0 + arma::norm(param_old, "fro"));
      if (diff < tolerance) {
        break;
      }
    } // End Newton Loop

    // --- KKT Check on Inactive Set ---
    arma::mat Residuals = P; // n x K
    for (int i = 0; i < n; ++i) {
      int yi = static_cast<int>(Y(i));
      if (yi > 0 && yi <= K) {
        Residuals(i, yi - 1) -= 1.0;
      }
    }

    bool violations_found = false;
    std::vector<int> new_active;

    std::vector<int> inactive_vars;
    inactive_vars.reserve(reg_p);
    std::vector<bool> is_active(p, false);
    for (int j : active_set)
      is_active[j] = true;

    for (int j = 0; j < reg_p; ++j) {
      if (!is_active[j]) inactive_vars.push_back(j);
    }
    
    if (!inactive_vars.empty()) {
      arma::uvec inactive_uvec = arma::conv_to<arma::uvec>::from(inactive_vars);
      // Fast BLAS-3 gradient computation for all inactive variables
      arma::mat Grad_inactive = (X.cols(inactive_uvec).t() * Residuals) / n; 
      
      for (size_t idx = 0; idx < inactive_vars.size(); ++idx) {
        int j = inactive_vars[idx];
        // Correct KKT for L1 element-wise penalty is max_k |grad_jk| > lam1
        double max_grad = arma::max(arma::abs(Grad_inactive.row(idx)));
        // Use >= to catch borderline violators that are rounded just below lam1
        if (max_grad >= lam1) {
          new_active.push_back(j);
          violations_found = true;
        }
      }
    }

    if (!violations_found) {
      converged = true;
      break; // Global convergence!
    } else {
      active_set.insert(active_set.end(), new_active.begin(), new_active.end());
      std::sort(active_set.begin(), active_set.end());
      active_set.erase(std::unique(active_set.begin(), active_set.end()),
                       active_set.end());

      if (verbose)
        Rcpp::Rcout << "KKT Check: Added " << new_active.size()
                    << " variables. Restarting Newton." << std::endl;
    }
  } // End Outer Loop

  // Ensure exact zeros for any variable below numerical precision threshold
  for (int j = 0; j < reg_p; ++j) {
    for (int k = 0; k < K; ++k) {
      if (std::abs(param(j, k)) < 1e-7) {
        param(j, k) = 0.0;
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("Estimates") = param,
                            Rcpp::Named("Converged") = converged,
                            Rcpp::Named("Convergence Iteration") = total_iter);
}


// -----------------------------------------------------------------------------
// SAGA SOLVER HELPERS & IMPLEMENTATION
// -----------------------------------------------------------------------------

static inline double estimate_lipschitz_default(const arma::mat &X) {
  // cheap approximation: max row norm squared / 4
  double max_row_norm_sq = 0.0;
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    double s = arma::dot(X.row(i), X.row(i));
    if (s > max_row_norm_sq)
      max_row_norm_sq = s;
  }
  return std::max(1e-12, max_row_norm_sq / 4.0);
}

static inline arma::rowvec compute_residual_row(const arma::rowvec &xrow,
                                                int yi, double offset_i,
                                                const arma::mat &param) {
  // linear scores for explicit classes (1 x K)
  arma::rowvec scores = xrow * param; // 1 x K
  // add offset to explicit classes (keeps original code behavior)
  scores += offset_i;

  // stable softmax against baseline 0
  double max_score = scores.max();
  if (max_score < 0.0)
    max_score = 0.0; // compare with baseline
  arma::rowvec ex = arma::exp(scores - max_score);
  double denom = arma::accu(ex) + std::exp(0.0 - max_score) + arma::datum::eps;
  arma::rowvec p_hat = ex / denom; // 1 x K

  arma::rowvec r = p_hat;
  if (yi > 0 && yi <= (int)p_hat.n_cols)
    r(yi - 1) -= 1.0;
  return r; // 1 x K
}

static inline arma::rowvec compute_probs_row(const arma::rowvec &xrow,
                                             double offset_i,
                                             const arma::mat &param) {
  // linear scores
  arma::rowvec scores = xrow * param; // 1 x K
  scores += offset_i;

  // stable softmax
  double max_score = scores.max();
  if (max_score < 0.0)
    max_score = 0.0;
  arma::rowvec ex = arma::exp(scores - max_score);
  double denom = arma::accu(ex) + std::exp(0.0 - max_score) + arma::datum::eps;
  return ex / denom; // 1 x K
}

void apply_proximal_step_native(arma::mat &param, const arma::mat &param_unprox,
                                double learning_rate, int reg_p, int p, int K,
                                int penalty, // 1 = elastic.net, 2 = scad
                                double lam1, double lam2, bool pos) {

  // Check if there is any penalty to apply
  if (lam1 <= 0.0 && (penalty == 2 || lam2 <= 0.0)) {
    param = param_unprox;
    return;
  }

  // Apply penalty only to the first reg_p rows
  if (reg_p > 0) {
    arma::mat U = param_unprox.head_rows(reg_p);

    // Loop over K classes to apply vector-wise prox
    for (int k = 0; k < K; ++k) {
      arma::vec u_k = U.col(k);
      arma::vec prox_k;

      if (penalty == 1) { // Elastic Net
        double scaled_lam1 = lam1 * learning_rate;
        double scaled_lam2 = lam2 * learning_rate;
        prox_k = prox_elastic_net_vec(u_k, scaled_lam1, scaled_lam2);
      } else { // SCAD
        double scaled_lam1 = lam1 * learning_rate;
        double a_scad = (lam2 >= 2.1) ? lam2 : 3.7;
        prox_k = prox_scad_vec(u_k, scaled_lam1, a_scad);
      }

      if (pos) {
        prox_k.elem(arma::find(prox_k < 0.0)).zeros();
      }

      param.submat(0, k, reg_p - 1, k) = prox_k;
    }
  }

  // Copy unpenalized rows
  if (reg_p < p) {
    param.tail_rows(p - reg_p) = param_unprox.tail_rows(p - reg_p);
  }
}

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSAGA_Native(
    const arma::mat &X, const arma::vec &Y, const arma::vec &offset, int K,
    int reg_p,
    int penalty = 1, // 1 = elastic.net, 2 = scad
    double lam1 = 0.0, double lam2 = 0.0, double tolerance = 1e-4,
    double lr_adj = 1.0, double max_lr = 1.0, int maxit = 500,
    bool verbose = false, bool pos = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue,
    double lam1_prev = 0.0) {

  const int p = X.n_cols;
  const int n = X.n_rows;

  if (reg_p < 0)
    reg_p = 0;
  if (reg_p > p)
    reg_p = p;

  // 1) Lipschitz approx (max row norm^2 / 4) - cheap and stable
  if (verbose)
    Rcpp::Rcout << "Estimating Lipschitz constant (approx)..." << std::endl;

  double L = estimate_lipschitz_default(X);
  double base_learning_rate = 1.0 / (6.0 * L + 1e-12);
  double learning_rate = std::min(lr_adj * base_learning_rate, max_lr);
  if (verbose)
    Rcpp::Rcout << "  L = " << L << ", LR = " << learning_rate << std::endl;

  // 2) initialize parameters with warm-start if provided
  arma::mat param(p, K, arma::fill::zeros);
  if (param_start.isNotNull()) {
    param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    if ((int)param.n_rows != p || (int)param.n_cols != K) {
      Rcpp::stop("param_start has incompatible dimensions");
    }
  }

  // 3) Initialize residual cache R (n x K) using current param (warm-start
  // correctness)
  arma::mat Rcache(n, K, arma::fill::zeros);
  for (int i = 0; i < n; ++i) {
    int yi = static_cast<int>(Y(i));
    Rcache.row(i) = compute_residual_row(X.row(i), yi, offset(i), param);
  }

  // 4) grad_avg = X^T * R / n
  arma::mat grad_avg = (X.t() * Rcache) / static_cast<double>(n);

  // Preallocate temporaries
  arma::mat param_old(p, K, arma::fill::zeros);
  arma::rowvec r_old(K);
  arma::rowvec r_new(K);
  arma::rowvec delta_r(K);

  bool converged = false;
  int convergence_iter = -1;

  double best_kkt_violation = arma::datum::inf;
  int epochs_since_improvement = 0;
  const int patience = 10;
  const double lr_decrease_factor = 0.5;

  // Main SAGA Epoch Loop
  for (int iter = 1; iter <= maxit; ++iter) {
    Rcpp::checkUserInterrupt();
    param_old = param;

    arma::uvec indices = arma::randperm(n);

    for (arma::uword jj = 0; jj < indices.n_elem; ++jj) {
      if (jj % 1000 == 0)
        Rcpp::checkUserInterrupt();
      int i = static_cast<int>(indices(jj));

      // residuals
      r_old = Rcache.row(i);
      r_new = compute_residual_row(X.row(i), static_cast<int>(Y(i)), offset(i),
                                   param);
      delta_r = r_new - r_old; // 1 x K

      // Optimize: Avoid creating x_outer (p x K) explicitly
      // SAGA update: param = param - lr * (x_outer + grad_avg)
      // param = param - lr * grad_avg - lr * x_outer
      // x_outer = x_i^T * delta_r

      arma::mat param_unprox = param - learning_rate * grad_avg;

      arma::vec xi = X.row(i).t(); // p x 1
      double n_inv = 1.0 / static_cast<double>(n);

      for (int k = 0; k < K; ++k) {
        // Update param
        double dr = delta_r(k);
        if (dr != 0.0) {
          param_unprox.col(k) -= learning_rate * dr * xi;
        }

        // Update grad_avg
        // grad_avg += x_outer / n
        if (dr != 0.0) {
          grad_avg.col(k) += (dr * n_inv) * xi;
        }
      }

      Rcache.row(i) = r_new;

      // gradient descent step then proximal
      apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p,
                                 K, penalty, lam1, lam2, pos);

      if (jj % 1000 == 0) {
        if (!param.is_finite()) {
          Rcpp::warning("SAGA diverged (NaN detected) within epoch.");
          goto end_of_saga;
        }
      }
    }

    // Convergence & KKT check every epoch
    arma::mat grad_penalized = grad_avg.head_rows(reg_p);
    arma::mat violations =
        arma::max(arma::zeros(reg_p, K), arma::abs(grad_penalized) - lam1);
    double v_max = (violations.n_elem > 0) ? violations.max() : 0.0;

    if (v_max < (best_kkt_violation - tolerance * 0.1)) {
      best_kkt_violation = v_max;
      epochs_since_improvement = 0;
    } else {
      epochs_since_improvement++;
    }
    if (epochs_since_improvement >= patience) {
      learning_rate *= lr_decrease_factor;
      if (verbose)
        Rcpp::Rcout << " (No improvement for " << patience
                    << " epochs. LR decreased to " << learning_rate << ")\n";
      epochs_since_improvement = 0;
      best_kkt_violation = v_max;
    }

    double diff = arma::abs(param - param_old).max();
    if (verbose && (iter % 10 == 0 || iter == 1)) {
      Rcpp::Rcout << "Iter " << iter << " | Max param change: " << diff
                  << " | KKT viol: " << v_max << std::endl;
    }

    if (iter >= 3 && diff < tolerance) {
      converged = true;
      convergence_iter = iter;
      if (verbose)
        Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
      break;
    }

    if (learning_rate < 1e-12) {
      if (verbose)
        Rcpp::Rcout << "Learning rate became too small. Stopping." << std::endl;
      break;
    }

    if (!param.is_finite()) {
      Rcpp::warning("Algorithm diverged with non-finite parameter values.");
      break;
    }
  }

end_of_saga:;

  return Rcpp::List::create(
      Rcpp::Named("Estimates") = param, Rcpp::Named("Converged") = converged,
      Rcpp::Named("Convergence Iteration") = convergence_iter);
}

// -----------------------------------------------------------------------------
// SVRG SOLVER IMPLEMENTATION (Loopless SVRG)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSVRG(
    const arma::mat &X, const arma::vec &Y, const arma::vec &offset, int K,
    int reg_p,
    int penalty = 1, // 1 = elastic.net, 2 = scad
    double lam1 = 0.0, double lam2 = 0.0,
    double update_prob = -1.0, // Default 1/n
    double tolerance = 1e-4, double lr_adj = 1.0, double max_lr = 1.0,
    int maxit = 500, bool verbose = false, bool pos = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue,
    double lam1_prev = 0.0) {

  const int n = X.n_rows;
  const int p = X.n_cols;

  if (reg_p < 0)
    reg_p = 0;
  if (reg_p > p)
    reg_p = p;

  if (update_prob <= 0.0) {
    update_prob = 1.0 / static_cast<double>(n);
  }

  // 1. Learning Rate
  if (verbose)
    Rcpp::Rcout << "Estimating Lipschitz constant..." << std::endl;
  double L = estimate_lipschitz_default(X);
  double base_lr = 1.0 / (6.0 * L + 1e-12);
  double learning_rate = std::min(lr_adj * base_lr, max_lr);
  if (verbose)
    Rcpp::Rcout << "  L=" << L << ", LR=" << learning_rate << std::endl;

  // 2. Initialize Parameters
  arma::mat param(p, K, arma::fill::zeros);
  if (param_start.isNotNull()) {
    param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
  }

  // Reference parameter (snapshot)
  arma::mat param_ref = param;

  // 3. Compute Full Gradient at Reference
  // Grad = X^T * Residuals / n
  arma::mat P_ref(n, K);
  // We can reuse compute_obj_and_probs but we need residuals
  arma::mat Residuals(n, K, arma::fill::zeros);

  // Compute initial residuals
  for (int i = 0; i < n; ++i) {
    Residuals.row(i) = compute_residual_row(X.row(i), static_cast<int>(Y(i)),
                                            offset(i), param_ref);
  }
  arma::mat mu = (X.t() * Residuals) / static_cast<double>(n); // Full gradient

  bool converged = false;
  int convergence_iter = -1;

  // For convergence check
  arma::mat param_old_epoch = param;
  // int check_freq = n; // Unused

  // Main Loop (Iterate by single steps, but track "epochs")
  long long max_steps = static_cast<long long>(maxit) * n;
  long long step = 0;

  for (step = 0; step < max_steps; ++step) {
    // 0. Safety Checks
    if (step % 1000 == 0) {
      Rcpp::checkUserInterrupt();
      if (!param.is_finite()) {
        Rcpp::warning("SVRG diverged (NaN detected).");
        break;
      }
    }

    // 1. Sample index
    int i = std::floor(arma::randu() * n);

    // 2. Compute Gradient Estimate
    // g = \nabla f_i(w) - \nabla f_i(w_ref) + \mu
    // \nabla f_i(w) = x_i^T (p_i(w) - y_i)
    // \nabla f_i(w_ref) = x_i^T (p_i(w_ref) - y_i)
    // g = x_i^T (p_i(w) - p_i(w_ref)) + \mu

    arma::rowvec p_w = compute_probs_row(X.row(i), offset(i), param);
    arma::rowvec p_ref = compute_probs_row(X.row(i), offset(i), param_ref);
    arma::rowvec diff_p = p_w - p_ref; // 1 x K

    // 3. Update w
    // w = prox(w - lr * g)
    // w = prox(w - lr * mu - lr * x_i^T * diff_p)

    arma::mat param_unprox = param - learning_rate * mu;
    arma::vec xi = X.row(i).t();

    for (int k = 0; k < K; ++k) {
      double d = diff_p(k);
      if (std::abs(d) > 1e-12) {
        param_unprox.col(k) -= learning_rate * d * xi;
      }
    }

    apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p, K,
                               penalty, lam1, lam2, pos);

    // 4. Probabilistic Snapshot Update
    if (arma::randu() < update_prob) {
      param_ref = param;
      // Recompute full gradient
      // This is expensive, but happens rarely (1/n)
      for (int r = 0; r < n; ++r) {
        Residuals.row(r) = compute_residual_row(
            X.row(r), static_cast<int>(Y(r)), offset(r), param_ref);
      }
      mu = (X.t() * Residuals) / static_cast<double>(n);
    }

    // 5. Convergence Check (every epoch)
    if ((step + 1) % n == 0) {
      int epoch = (step + 1) / n;
      double diff = arma::abs(param - param_old_epoch).max();

      // KKT Check (approximate using mu if close to ref, or just use mu as
      // proxy) Strictly we should use current gradient, but mu is close enough
      // if update_prob is reasonable Let's use mu (gradient at ref) as proxy
      // for gradient at w
      arma::mat grad_pen = mu.head_rows(reg_p);
      arma::mat violations =
          arma::max(arma::zeros(reg_p, K), arma::abs(grad_pen) - lam1);
      double v_max = (violations.n_elem > 0) ? violations.max() : 0.0;

      if (verbose) {
        Rcpp::Rcout << "Epoch " << epoch << " | Max param change: " << diff
                    << " | KKT viol (approx): " << v_max << std::endl;
      }

      if (diff < tolerance) {
        converged = true;
        convergence_iter = epoch;
        if (verbose)
          Rcpp::Rcout << "Converged at epoch " << epoch << std::endl;
        break;
      }

      if (!param.is_finite()) {
        Rcpp::warning("Algorithm diverged.");
        break;
      }

      param_old_epoch = param;
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("Estimates") = param, Rcpp::Named("Converged") = converged,
      Rcpp::Named("Convergence Iteration") = convergence_iter);
}

// [[Rcpp::export]]
Rcpp::List MultinomLogisticFISTA(const arma::mat& X, const arma::vec& Y, const arma::vec& offset,
                                 int K, int reg_p, int penalty = 1, double lam1 = 0.0,
                                 double lam2 = 0.0, double tolerance = 1e-4, int maxit = 500,
                                 double lr_adj = 1.0, bool verbose = false, bool pos = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue,
    double lam1_prev = 0.0) {
    int n = X.n_rows;
    int p = X.n_cols;

    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }

    arma::mat param_old = param;
    arma::mat y = param;
    double t = 1.0;

    double L = estimate_lipschitz_default(X); 
    double lr = lr_adj / (L + 1e-12);

    bool converged = false;
    int iter = 0;

    for (; iter < maxit; ++iter) {
        if (iter % 100 == 0) Rcpp::checkUserInterrupt();
        
        arma::mat eta = X * y;
        eta.each_col() += offset;

        // Stable softmax with baseline 0
        arma::vec max_scores = arma::max(eta, 1);
        max_scores.elem(arma::find(max_scores < 0.0)).zeros();
        
        arma::mat P = eta;
        P.each_col() -= max_scores;
        P = arma::exp(P);
        
        arma::vec row_sums = arma::sum(P, 1) + arma::exp(-max_scores) + arma::datum::eps;
        P.each_col() /= row_sums;

        arma::mat Grad = X.t() * P;
        for (int i = 0; i < n; ++i) {
            int k = Y(i) - 1;
            if (k >= 0 && k < K) {
                Grad.col(k) -= X.row(i).t();
            }
        }
        Grad /= static_cast<double>(n);

        arma::mat param_new = y - lr * Grad;
        arma::mat param_unprox = param_new;
        
        apply_proximal_step_native(param_new, param_unprox, lr, reg_p, p, K, penalty, lam1, lam2, pos);

        double diff = arma::abs(param_new - param).max();
        if (diff < tolerance) {
            param = param_new;
            converged = true;
            break;
        }

        double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
        y = param_new + ((t - 1.0) / t_new) * (param_new - param);

        param_old = param;
        param = param_new;
        t = t_new;
    }

    return Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = iter
    );
}
