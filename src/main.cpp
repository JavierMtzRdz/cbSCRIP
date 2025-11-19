
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
                             arma::mat &P, int K) {
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

// [[Rcpp::export]]
Rcpp::List MultinomLogisticCCD(
    const arma::mat &X, const arma::vec &Y, const arma::vec &offset, int K,
    int penalty = 1, double lam1 = 0.0, double lam2 = 0.0,
    double tolerance = 1e-5, int maxit = 100,
    double lr_adj = 1.0, // Unused in Newton, kept for compatibility
    bool verbose = false, bool pos = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {

  int n = X.n_rows;
  int p = X.n_cols;

  // Initialize parameters
  arma::mat param(p, K, arma::fill::zeros);
  if (param_start.isNotNull()) {
    param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
  }

  // Probability matrix (stores P(Y=k) for k=1..K. P(Y=0) is implicit)
  arma::mat P(n, K, arma::fill::zeros);

  // Current objective
  double current_obj = compute_obj_and_probs(X, Y, offset, param, P, K);

  bool converged = false;
  int iter = 0;

  // Active set
  std::vector<int> active_set;
  active_set.reserve(p);
  for (int j = 0; j < p; ++j)
    active_set.push_back(j);

  // Precompute X squared for diagonal Hessian upper bound
  arma::mat X2 = arma::square(X);

  // Newton Loop
  for (iter = 1; iter <= maxit; ++iter) {
    Rcpp::checkUserInterrupt();

    arma::mat param_old = param;
    double obj_old = current_obj;

    // 1. Compute Gradient and Hessian Diagonal
    arma::mat Residuals = P; // n x K
    for (int i = 0; i < n; ++i) {
      int yi = static_cast<int>(Y(i));
      if (yi > 0 && yi <= K) {
        Residuals(i, yi - 1) -= 1.0;
      }
    }

    arma::mat Grad = (X.t() * Residuals) / n; // p x K

    // 2. Inner Loop: Coordinate Descent on Quadratic Surrogate
    arma::mat W = P % (1.0 - P);    // n x K, element-wise
    arma::mat H = (X2.t() * W) / n; // p x K

    // Add ridge penalty to Hessian if elastic net
    if (penalty == 1 && lam2 > 0) {
      H += lam2;
    }

    // Ensure Hessian is strictly positive
    H.elem(arma::find(H < 1e-6)).fill(1e-6);

    // Coordinate Descent
    int inner_maxit = 100; // Usually converges very fast
    double inner_tol = 1e-5;

    // We need a working copy of param for the inner loop
    arma::mat param_inner = param;

    for (int inner = 0; inner < inner_maxit; ++inner) {
      double max_inner_diff = 0.0;

      for (int j : active_set) {
        for (int k = 0; k < K; ++k) {
          double beta_old_jk = param_inner(j, k);
          double grad_jk = Grad(j, k);
          double h_jk = H(j, k);

          // Newton step on the quadratic model:
          // Q(z) approx f(x) + g*z + 0.5*h*z^2
          // We want to minimize Q(z) + Penalty(x+z)
          // The gradient 'grad_jk' is at param_old.
          // As we update param_inner, the gradient *should* change, but in CD
          // with fixed Hessian/Grad from outer loop, we typically update the
          // 'residual' or 'grad' effectively. However, the standard "glmnet"
          // style uses the fixed gradient from the outer loop and only updates
          // the coordinate. Wait, if we don't update Grad, we are just
          // minimizing the SAME quadratic approximation repeatedly, which means
          // we just jump to the minimum of that quadratic. One pass is enough
          // if variables are independent. But they are not. To do CD correctly
          // on the quadratic approximation: grad_current = grad_initial + H *
          // (beta_current - beta_initial) This is what we need to track.

          double current_grad_jk =
              grad_jk + h_jk * (beta_old_jk - param_old(j, k));

          double z = beta_old_jk - current_grad_jk / h_jk;
          double beta_new_jk = 0.0;

          if (penalty == 1) { // Elastic Net
            double thresh = lam1 / h_jk;
            if (z > thresh)
              beta_new_jk = z - thresh;
            else if (z < -thresh)
              beta_new_jk = z + thresh;
            else
              beta_new_jk = 0.0;

          } else if (penalty == 2) { // SCAD
            double a_scad = (lam2 > 2.0) ? lam2 : 3.7;
            beta_new_jk = scalar_scad_prox(z, lam1 / h_jk, a_scad);
          }

          if (pos && beta_new_jk < 0)
            beta_new_jk = 0.0;

          if (std::abs(beta_new_jk - beta_old_jk) > 1e-10) {
            param_inner(j, k) = beta_new_jk;
            max_inner_diff =
                std::max(max_inner_diff, std::abs(beta_new_jk - beta_old_jk));
          }
        }
      }

      if (max_inner_diff < inner_tol)
        break;
    }

    // 3. Backtracking Line Search
    arma::mat direction = param_inner - param_old;
    double step_size = 1.0;
    double alpha = 0.5; // Backtracking parameter
    double c = 1e-4;    // Sufficient decrease parameter
    bool step_accepted = false;

    // Compute initial penalty value
    double pen_old = 0.0;
    if (penalty == 1) {
      pen_old = lam1 * arma::accu(arma::abs(param_old)) +
                0.5 * lam2 * arma::accu(arma::square(param_old));
    }

    // We want f(new) <= f(old) + c * step * grad^T * dir (Armijo)
    // But for proximal gradient, we just check descent on the composite
    // objective. Simple backtracking: ensure objective decreases.

    double current_total_obj = obj_old + pen_old;

    for (int ls = 0; ls < 20; ++ls) {
      arma::mat param_proposal = param_old + step_size * direction;

      // Recompute objective
      arma::mat P_proposal(n, K);
      double obj_proposal =
          compute_obj_and_probs(X, Y, offset, param_proposal, P_proposal, K);

      double pen_proposal = 0.0;
      if (penalty == 1) {
        pen_proposal = lam1 * arma::accu(arma::abs(param_proposal)) +
                       0.5 * lam2 * arma::accu(arma::square(param_proposal));
      }

      if (obj_proposal + pen_proposal <=
          current_total_obj + 1e-5) { // Allow tiny increase for numerical noise
        param = param_proposal;
        P = P_proposal;
        current_obj = obj_proposal;
        step_accepted = true;
        break;
      }

      step_size *= alpha;
    }

    if (!step_accepted) {
      // If line search fails, we might be at a minimum or stuck.
      // Take a very small step or stop.
      if (verbose)
        Rcpp::Rcout << "Line search failed at iter " << iter << std::endl;
      // Keep old param, but maybe we converged?
      // If the step is tiny, we are done.
      if (arma::norm(direction, "fro") < tolerance) {
        converged = true;
        break;
      }
      // Otherwise, just accept the small step to try to move? No, unsafe.
      // Break to avoid divergence.
      break;
    }

    // 4. Check Convergence
    double diff = arma::norm(param - param_old, "fro") /
                  (1.0 + arma::norm(param_old, "fro"));

    if (verbose) {
      double pen_val = 0.0;
      if (penalty == 1) {
        pen_val = lam1 * arma::accu(arma::abs(param)) +
                  0.5 * lam2 * arma::accu(arma::square(param));
      }
      Rcpp::Rcout << "Iter " << iter << " | Obj: " << current_obj + pen_val
                  << " | Rel Change: " << diff << std::endl;
    }

    if (diff < tolerance) {
      converged = true;
      break;
    }
  }

  return Rcpp::List::create(Rcpp::Named("Estimates") = param,
                            Rcpp::Named("Converged") = converged,
                            Rcpp::Named("Convergence Iteration") = iter);
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
    Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {

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
      int i = static_cast<int>(indices(jj));

      // residuals
      r_old = Rcache.row(i);
      r_new = compute_residual_row(X.row(i), static_cast<int>(Y(i)), offset(i),
                                   param);
      delta_r = r_new - r_old; // 1 x K

      // outer product x_i^T * delta_r -> p x K
      arma::mat x_outer = X.row(i).t() * delta_r; // p x K

      // SAGA update direction: x_outer + grad_avg
      arma::mat saga_update_direction = x_outer + grad_avg;

      // update grad_avg and cache
      grad_avg += x_outer / static_cast<double>(n);
      Rcache.row(i) = r_new;

      // gradient descent step then proximal
      arma::mat param_unprox = param - learning_rate * saga_update_direction;
      apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p,
                                 K, penalty, lam1, lam2, pos);
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

  return Rcpp::List::create(
      Rcpp::Named("Estimates") = param, Rcpp::Named("Converged") = converged,
      Rcpp::Named("Convergence Iteration") = convergence_iter);
}
