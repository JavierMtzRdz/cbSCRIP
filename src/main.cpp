// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "RcppArmadillo.h"
#include <iostream>
#include <vector> 
#include <string>
#include <cmath>  
#include <map>
#include <numeric>        
#include <unordered_map>  
#include <limits>
#include <algorithm> 
#include <set>  
#include "spams.h"



// [[Rcpp::export]]
arma::vec grad_ls_loss(
    arma::rowvec& x,
    double& y,
    arma::vec& param,
    int& p) {
  arma::vec grad(p);
  grad =  arma::vectorise(x) * (arma::dot(x, param) - y);
  return grad;
}

// [[Rcpp::export]]
arma::vec grad_logistic_loss(
    arma::rowvec& x,
    double& y,
    arma::vec& param,
    int& p) {
  arma::vec grad(p);
  double sig;
  sig = 1.0/( 1.0 + exp(-arma::dot(x, param)) );
  grad =  arma::vectorise(x) * (sig - y);
  return grad;
}


// [[Rcpp::export]]
arma::mat grad_multinom_loss(
        const arma::rowvec& x,
        int y,
        int K,
        double offset,
        const arma::mat& param,
        int p) {
    
    arma::mat grad_out(p, K);
    
    // linear score for each of the K classes.
    arma::vec linear_scores = param.t() * x.t() + offset;
    
    // log-sum-exp trick
    double max_score = linear_scores.max();
    if (max_score < 0.0) {
        max_score = 0.0;
    }
    
    // exponent of the stabilized scores
    arma::vec exp_scores = arma::exp(linear_scores - max_score);
    
    // final probabilities 
    // 'exp(0.0 - max_score)' is for the baseline class.
    arma::vec pi = exp_scores / (arma::accu(exp_scores) + exp(0.0 - max_score));
    
    // Vectorized calculation 
    arma::vec x_col = x.t();
    
    grad_out = x_col * pi.t();
    
    //  column corresponding to the true class.
    if (y > 0) { // indicator
        // (pi_k - 1) * x for the true class.
        //  subtract x.
        grad_out.col(y - 1) -= x_col;
    }
    
    return grad_out;
}



void proximalFlat(
        arma::mat& U,
        const int& p,
        const int& K,
        const std::string& regul,
        Rcpp::IntegerVector& grp_id,
        int num_threads,
        double lam1,
        double lam2 = 0.0,
        double lam3 = 0.0,
        bool pos = false) {
    
    // Wrap the existing Armadillo memory for the spams::Matrix object without copying.
    Matrix<double> alpha0(U.memptr(), p, K);
    
    // Safely get a mutable C-style string from std::string.
    std::vector<char> name_regul_buf(regul.begin(), regul.end());
    name_regul_buf.push_back('\0');
    char* name_regul_ptr = name_regul_buf.data();
    
    // Wrap the existing Rcpp vector memory without copying.
    Vector<int> groups(grp_id.begin(), p);
    
    // Initialize the output matrix for the proximal operator.
    Matrix<double> alpha(p, K);
    alpha.setZeros();
    
    // Call the backend spams function.
    _proximalFlat(&alpha0, &alpha, &groups, num_threads, lam1, lam2, lam3, false,
                  false, name_regul_ptr, false, pos, true, true, 1, false);
    
    // The backend function modifies `alpha`. Since alpha0 and U share memory,
    // U is modified in-place. No explicit copy-back is needed if spams
    // allows modifying the input matrix directly, otherwise a copy is needed.
    // For safety, this implementation assumes spams writes to `alpha` and we copy back.
    U = arma::mat(alpha.rawX(), p, K, true); // Create Armadillo matrix from spams result
}

void proximalGraph(
        arma::mat& U,
        const int& p,
        const int& K,
        const std::string& regul,
        const arma::mat& grp,
        const arma::mat& grpV,
        const Rcpp::NumericVector& etaG,
        int num_threads,
        double lam1,
        double lam2 = 0.0,
        bool pos = false) {
    
    Matrix<double> alpha0(U.memptr(), p, K);
    
    std::vector<char> grp_vec(grp.n_elem);
    for (arma::uword i = 0; i < grp.n_elem; ++i) {
        grp_vec[i] = (grp[i] != 0.0); // Assigns 1 or 0
    }
    Matrix<bool> grp_dense(reinterpret_cast<bool*>(grp_vec.data()), grp.n_rows, grp.n_cols);
    SpMatrix<bool> groups;
    grp_dense.toSparse(groups);
    
    std::vector<char> grpV_vec(grpV.n_elem);
    for (arma::uword i = 0; i < grpV.n_elem; ++i) {
        grpV_vec[i] = (grpV[i] != 0.0);
    }
    Matrix<bool> grpV_dense(reinterpret_cast<bool*>(grpV_vec.data()), grpV.n_rows, grpV.n_cols);
    SpMatrix<bool> groups_var;
    grpV_dense.toSparse(groups_var);
    
    Vector<double> eta_g(const_cast<double*>(etaG.begin()), etaG.length());
    std::vector<char> name_regul_buf(regul.begin(), regul.end());
    name_regul_buf.push_back('\0');
    char* name_regul_ptr = name_regul_buf.data();
    Matrix<double> alpha(p, K);
    alpha.setZeros();
    
    _proximalGraph(&alpha0, &alpha, &eta_g, &groups, &groups_var, num_threads,
                   lam1, lam2, 0.0, false, false, name_regul_ptr, false, pos,
                   true, true, 1, false);
    
    U = arma::mat(alpha.rawX(), p, K, true);
}

void proximalTree(
        arma::mat& U,
        const int& p,
        const int& K,
        const std::string& regul,
        const arma::mat& grp,
        const Rcpp::NumericVector& etaG,
        Rcpp::IntegerVector& own_var,
        Rcpp::IntegerVector& N_own_var,
        int num_threads,
        double lam1,
        double lam2 = 0.0,
        bool pos = false) {
    
    Matrix<double> alpha0(U.memptr(), p, K);
    
    std::vector<char> grp_vec(grp.n_elem);
    for (arma::uword i = 0; i < grp.n_elem; ++i) {
        grp_vec[i] = (grp[i] != 0.0);
    }
    Matrix<bool> grp_dense(reinterpret_cast<bool*>(grp_vec.data()), grp.n_rows, grp.n_cols);
    SpMatrix<bool> groups;
    grp_dense.toSparse(groups);
    
    Vector<double> eta_g(const_cast<double*>(etaG.begin()), etaG.length());
    Vector<int> own_variables(own_var.begin(), own_var.length());
    Vector<int> N_own_variables(N_own_var.begin(), N_own_var.length());
    std::vector<char> name_regul_buf(regul.begin(), regul.end());
    name_regul_buf.push_back('\0');
    char* name_regul_ptr = name_regul_buf.data();
    Matrix<double> alpha(p, K);
    alpha.setZeros();
    
    _proximalTree(&alpha0, &alpha, &eta_g, &groups, &own_variables, &N_own_variables,
                  num_threads, lam1, lam2, 0.0, false, false, name_regul_ptr,
                  false, pos, true, true, 1, false);
    
    U = arma::mat(alpha.rawX(), p, K, true);
}


// [[Rcpp::export]]
double scalar_scad_prox(double val, double lambda, double a) {
    double abs_val = std::abs(val);
    if (a <= 2.0) a = 3.7;
    
    if (abs_val <= 2.0 * lambda) {
        return std::copysign(std::max(0.0, abs_val - lambda), val);
    } else if (abs_val <= a * lambda) {
        return ((a - 1.0) * val - std::copysign(a * lambda, val)) / (a - 2.0);
    } else {
        return val;
    }
}

// [[Rcpp::export]]
void proximalSCAD(
        arma::mat& U,
        double lam1,
        double a_scad = 3.7,
        bool pos = false) {
    
    if (a_scad <= 2.0) {
        Rcpp::stop("The 'a' parameter for the SCAD penalty must be greater than 2.");
    }
    
    U.for_each([&](arma::mat::elem_type& val) {
        double new_val = scalar_scad_prox(val, lam1, a_scad);
        if (pos && new_val < 0) {
            val = 0.0;
        } else {
            val = new_val;
        }
    });
}

void apply_proximal_step(
        arma::mat& param,
        const arma::mat& param_unprox, 
        double learning_rate,
        int reg_p, int K, int p, bool transpose, 
        int penalty, 
        const std::string& regul,
        Rcpp::IntegerVector& grp_id, 
        int ncores, 
        double lam1, double lam2, double lam3, 
        bool pos,
        const arma::mat& grp, 
        const arma::mat& grpV,
        const Rcpp::NumericVector& etaG,
        Rcpp::IntegerVector& own_var, Rcpp::IntegerVector& N_own_var
) {
    
    const double epsilon = 1e-7;
    
    if (std::abs(lam1) < epsilon && std::abs(lam2) < epsilon) {
        param = param_unprox;
        return;
    }
    
    auto param_unprox_reg_view = param_unprox.head_rows(reg_p);
    
    double scaled_lam1 = lam1 * learning_rate;
    double scaled_lam2 = lam2 * learning_rate;
    double scaled_lam3 = lam3 * learning_rate;
    
    if (transpose) {
        arma::mat param_t = param_unprox_reg_view.t();
        if (penalty == 1) proximalFlat(param_t, K, reg_p, regul, grp_id, ncores, scaled_lam1, scaled_lam2, scaled_lam3, pos);
        else if (penalty == 2) proximalGraph(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 3) proximalTree(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 4) proximalSCAD(param_t, scaled_lam1, lam2, pos); 
        param.head_rows(reg_p) = param_t.t();
    } else {
        arma::mat param_unprox_reg_copy = param_unprox_reg_view;
        if (penalty == 1) proximalFlat(param_unprox_reg_copy, reg_p, K, regul, grp_id, ncores, scaled_lam1, scaled_lam2, scaled_lam3, pos);
        else if (penalty == 2) proximalGraph(param_unprox_reg_copy, reg_p, K, regul, grp, grpV, etaG, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 3) proximalTree(param_unprox_reg_copy, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 4) proximalSCAD(param_unprox_reg_copy, scaled_lam1, lam2, pos);
        param.head_rows(reg_p) = param_unprox_reg_copy;
    }
    
    if (reg_p < p) {
        param.tail_rows(p - reg_p) = param_unprox.tail_rows(p - reg_p);
    }
}

static double get_lambda_max(const arma::mat& X) {
    if (X.n_cols == 0) return 0.0;
    arma::vec v(X.n_cols, arma::fill::randn);
    v = v / arma::norm(v);
    double lambda_old = 0.0, lambda_new = 1.0;
    for (int i = 0; i < 100; ++i) {
        arma::vec XtXv = X.t() * (X * v);
        lambda_new = arma::norm(XtXv);
        v = XtXv / lambda_new;
        if (std::abs(lambda_new - lambda_old) < 1e-8) break;
        lambda_old = lambda_new;
    }
    return lambda_new;
}


// [[Rcpp::export]]
Rcpp::List MultinomLogistic(
        arma::mat X,
        arma::vec Y,
        arma::vec offset,
        int K,
        int reg_p,
        int penalty,
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        arma::mat grp,
        arma::mat grpV,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double learning_rate,
        double tolerance,
        int niter_inner,
        int maxit,
        int ncores,
        bool save_history = false,
        bool verbose = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    int p = X.n_cols;
    int n = X.n_rows;
    
    int index;
    arma::rowvec x_sample(p);
    int y_sample;
    double o_sample;
    
    arma::mat grad(p, K);
    arma::mat temp1(p, K);
    arma::mat temp2(p, K);
    arma::mat param_unprox(p, K); 
    
    arma::mat param(p, K);
    if (param_start.isNotNull()) {
        Rcpp::NumericMatrix start_mat_r(param_start);
        if (start_mat_r.nrow() != p || start_mat_r.ncol() != K) {
            Rcpp::stop("Dimensions of 'param_start' do not match the data (p x K).");
        }
        param = Rcpp::as<arma::mat>(start_mat_r);
    } else {
        param.zeros();
    }
    
    arma::mat param_old(p, K);
    
    bool converged = false;
    int convergence_iter = -1;
    Rcpp::List param_history;
    double diff;
    int counter_outer = 0;
    
    while (true) {
        
        Rcpp::checkUserInterrupt();
        
        param_old = param;
        
        if (save_history) {
            param_history.push_back(Rcpp::wrap(param_old));
        }
        
        grad.zeros();
        for (int i = 0; i < n; i++) {
            x_sample = X.row(i);
            y_sample = Y(i);
            o_sample = offset(i);
            grad += grad_multinom_loss(x_sample, y_sample, K, o_sample, param_old, p);
        }
        grad /= n;
        
        for (int i = 0; i < niter_inner; ++i) {
            index = arma::randi(arma::distr_param(0, n - 1));
            x_sample = X.row(index);
            y_sample = Y(index);
            o_sample = offset(index);
            
            temp1 = grad_multinom_loss(x_sample, y_sample, K, o_sample, param, p);
            temp2 = grad_multinom_loss(x_sample, y_sample, K, o_sample, param_old, p);
            
            // Perform the gradient step and store in a temporary variable
            param_unprox = param - learning_rate * (temp1 - temp2 + grad);
            
            apply_proximal_step(
                param, param_unprox, learning_rate,
                reg_p, K, p, transpose, penalty, regul,
                grp_id, ncores, lam1, lam2, lam3, false, 
                grp, grpV, etaG, own_var, N_own_var
            );
        }
        
        counter_outer += 1;
        diff = arma::norm(param - param_old, "fro") / (arma::norm(param_old, "fro") + 1e-10);
        
        if (verbose) {
            Rcpp::Rcout << "Iteration " << counter_outer << " | Relative Change: " << diff << "\n";
        }
        
        if (diff < tolerance) {
            converged = true;
            convergence_iter = counter_outer;
            break;
        }
        if (counter_outer >= maxit) {
            convergence_iter = maxit;
            break;
        }
    }
    
    arma::sp_mat param_sp(param);
    
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Sparse Estimates") = param_sp,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
    
    if (save_history) {
        param_history.push_back(Rcpp::wrap(param));
        result["History"] = param_history;
    }
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List MultinomLogisticAcc(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int reg_p,
        int penalty,
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        const arma::mat& grp,
        const arma::mat& grpV,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double c_factor,  
        double v_factor, 
        double tolerance,
        int niter_inner,
        int maxit,
        int ncores,
        bool save_history = false,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    
    if (verbose) Rcpp::Rcout << "Detecting Lipschitz constant (L)..." << std::endl;
    double lambda_max = get_lambda_max(X);
    double L = (n > 0) ? (lambda_max / static_cast<double>(n)) : 1.0;
    if (L < 1e-8) L = 1.0;
    if (verbose) Rcpp::Rcout << "  - Lipschitz Constant (L) estimated as: " << L << std::endl;
    
    // --- Initialize iterates for the accelerated algorithm ---
    arma::mat param_z(p, K), param_y(p, K), param_x(p, K);
    if (param_start.isNotNull()) {
        arma::mat start_mat = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
        param_z = start_mat;
        param_y = start_mat;
        param_x = start_mat;
    } else {
        param_z.zeros();
        param_y.zeros();
        param_x.zeros();
    }
    
    arma::mat snapshot_param(p, K), param_y_old(p, K), grad_full(p, K), grad_at_x(p, K),
    grad_at_snapshot(p, K), grad_stoch(p, K), param_z_unprox(p, K);
    
    bool converged = false;
    int convergence_iter = -1;
    Rcpp::List param_history;
    double diff;
    int counter_outer = 0;
    
    while (true) {
        Rcpp::checkUserInterrupt();
        param_y_old = param_y;
        counter_outer += 1;
        
        if (save_history) {
            param_history.push_back(Rcpp::wrap(param_y));
        }
        
        snapshot_param = param_y;
        grad_full.zeros();
        for (int i = 0; i < n; i++) {
            grad_full += grad_multinom_loss(X.row(i), Y(i), K, offset(i), snapshot_param, p);
        }
        grad_full /= n;
        
        for (int k = 0; k < niter_inner; ++k) {
            double gamma_k = (static_cast<double>(k) + v_factor + 4.0) / (2.0 * c_factor * L);
            double tau_k = 1.0 / (c_factor * L * gamma_k);
            if (tau_k > 1.0) tau_k = 1.0;
            
            // 1. Extrapolation step
            param_x = tau_k * param_z + (1.0 - tau_k) * param_y;
            
            // 2. Variance-reduced gradient calculation
            int index = arma::randi(arma::distr_param(0, n - 1));
            grad_at_x = grad_multinom_loss(X.row(index), Y(index), K, offset(index), param_x, p);
            grad_at_snapshot = grad_multinom_loss(X.row(index), Y(index), K, offset(index), snapshot_param, p);
            grad_stoch = grad_at_x - grad_at_snapshot + grad_full;
            
            // 3. Proximal gradient step on z-iterate
            param_z_unprox = param_z - gamma_k * grad_stoch;
            apply_proximal_step(param_z, param_z_unprox, gamma_k, reg_p, K, p, transpose, penalty, regul,
                                grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
            
            // 4. Momentum update on y-iterate
            param_y = tau_k * param_z + (1.0 - tau_k) * param_y;
        }
        
        diff = arma::norm(param_y - param_y_old, "fro") / (arma::norm(param_y_old, "fro") + 1e-10);
        
        if (verbose) {
            Rcpp::Rcout << "Iteration " << counter_outer << " | Relative Change: " << diff << "\n";
        }
        
        if (diff < tolerance) {
            converged = true;
            convergence_iter = counter_outer;
            break;
        }
        if (counter_outer >= maxit) {
            convergence_iter = maxit;
            break;
        }
    }
    
    arma::sp_mat param_sp(param_y);
    
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = param_y,
        Rcpp::Named("Sparse Estimates") = param_sp,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
    
    if (save_history) {
        param_history.push_back(Rcpp::wrap(param_y));
        result["History"] = param_history;
    }
    
    return result;
}


// [[Rcpp::export]]
void grad_multinom_loss2(
        const arma::rowvec& x,
        int y,
        int K,
        double offset,
        const arma::mat& param,
        int p,
        arma::mat& grad_out) {
    
    // 1. Calculate the linear score for each of the K classes.
    arma::vec linear_scores = param.t() * x.t() + offset;
    
    // 2. Find the maximum score among all classes, including the baseline's score of 0.
    // This is the key to preventing numerical overflow in the exp() function.
    double max_score = linear_scores.max();
    if (max_score < 0.0) {
        max_score = 0.0;
    }
    
    // 3. Calculate the exponent of the stabilized scores.
    arma::vec exp_scores = arma::exp(linear_scores - max_score);
    
    // 4. Calculate the final probabilities using the correct, stable denominator.
    // The term 'exp(0.0 - max_score)' represents the stabilized score of the baseline class.
    arma::vec pi = exp_scores / (arma::accu(exp_scores) + exp(0.0 - max_score));
    
    // 5. Calculate the gradient based on the true class label.
    arma::vec x_col = x.t();
    
    if (y == 0) {
        // Case: The true outcome is the baseline "survival" class.
        // The gradient for each class k is (pi_k - 0) * x.
        for (int k = 0; k < K; k++) {
            grad_out.col(k) = pi(k) * x_col;
        }
    } else {
        // Case: The true outcome is one of the K explicit classes.
        // Note: y is 1-indexed, so we subtract 1 for the 0-based index.
        int y_idx = y - 1;
        for (int k = 0; k < K; k++) {
            if (k == y_idx) {
                // For the correct class, the gradient is (pi_k - 1) * x.
                grad_out.col(k) = (pi(k) - 1.0) * x_col;
            } else {
                // For all other classes, the gradient is (pi_k - 0) * x.
                grad_out.col(k) = pi(k) * x_col;
            }
        }
    }
}

// Modified 
// void grad_multinom_loss2(
//         const arma::rowvec& x,
//         int y,
//         int K,
//         double offset,
//         const arma::mat& param,
//         int p,
//         arma::mat& grad_out) {
//     arma::mat grad(p, K);
//     arma::vec pi(K);
//     
//     for (int i = 0; i < K; i++) {
//         pi(i) = exp(arma::dot(x, param.col(i)) + offset);
//     }
//     pi = pi/(arma::sum(pi) + 1.0);
//     
//     if (y == 0) {
//         for (int k = 0; k < K; k++) {
//             grad_out.col(k) = pi(k) * arma::vectorise(x);
//         }
//     } else {
//         for (int k = 0; k < K; k++) {
//             if (k == y - 1) {
//                 grad_out.col(k) = (pi(k) - 1) * arma::vectorise(x);
//             } else {
//                 grad_out.col(k) = pi(k) * arma::vectorise(x);
//             }
//         }
//     }
// }


// void grad_multinom_loss2(
//         const arma::rowvec& x,
//         int y,
//         int K,
//         double offset,
//         const arma::mat& param,
//         int p,
//         arma::mat& grad_out) {
// 
// 
//     arma::vec pi(K);
// 
//     // Calculate softmax probabilities pi
//     for (int k = 0; k < K; k++) {
//         pi(k) = exp(arma::dot(x, param.col(k)) + offset);
//     }
// 
//     double sum_pi = arma::sum(pi);
//     // Prevent division by zero
//     if (sum_pi > 1e-10) {
//         pi /= (sum_pi + 1.0);
//     } else {
//         pi.fill(1.0 / K);
//     }
// 
//     arma::vec x_col = x.t();
// 
//     // Calculate gradient columns
//     if (y == 0) {
//         for (int k = 0; k < K; k++) {
//             grad_out.col(k) = pi(k) * x_col;
//         }
//     } else {
//         int y_idx = y - 1;
//         for (int k = 0; k < K; k++) {
//             if (k == y_idx) {
//                 grad_out.col(k) = (pi(k) - 1.0) * x_col;
//             } else {
//                 grad_out.col(k) = pi(k) * x_col;
//             }
//         }
//     }
// }


// [[Rcpp::export]]
Rcpp::List MultinomLogisticExp(
        arma::mat X,
        arma::vec Y,
        arma::vec offset,
        int K,
        int reg_p,              // number of regularized variables
        int penalty,            // 1 - proximalFlat; 2 - proximalGraph; 3 - proximalTree
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        arma::mat grp,
        arma::mat grpV,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double learning_rate,
        double tolerance,
        int niter_inner,
        int maxit,
        int ncores) {
    
    int p = X.n_cols;
    int n = X.n_rows;
    
    // indexing, stochastic sampling, etc..
    int index;    // index of the stochastic sample
    arma::rowvec x_sample(p);
    int y_sample;
    double o_sample; // o for offset
    
    // gradient and related
    arma::mat grad(p, K);
    arma::mat temp1(p, K);
    arma::mat temp2(p, K);
    
    // parameter and related
    arma::mat param(p, K);
    param.zeros();
    arma::mat param_old(p, K);
    
    arma::mat param_reg(reg_p, K);   // regularized coefficients
    arma::mat param_t(K, reg_p);      // regularized coefficients
    
    // convergence related
    double diff;
    int counter_outer = 0;
    
    std::vector<arma::mat> param_history;
    
    // compute mu: mean gradient at param_old
    while (true) {
        param_old = param; 
        arma::mat single_grad(p, K);
        grad.zeros();
        
        for (int i = 0; i < n; i++) {
            const auto& x_sample = X.row(i);
            const int& y_sample = Y(i);
            const double& o_sample = offset(i);
            
            grad_multinom_loss2(x_sample,
                                y_sample,
                                K,
                                o_sample,
                                param_old,
                                p,
                                single_grad);
            
            grad += single_grad;
        }
        
        // Divide the final SUM by n ONCE
        if (n > 0) {
            grad /= n;
        }
        
        // inner loop
        for (int i = 0; i < niter_inner; ++i) {
            
            index = arma::randi(arma::distr_param(0, n - 1));
            
            x_sample = X.row(index);
            y_sample = Y(index);
            o_sample = offset(index);
            
            grad_multinom_loss2(x_sample, y_sample, K, o_sample, param, p, temp1);
            grad_multinom_loss2(x_sample, y_sample, K, o_sample, param_old, p, temp2);
            
            param = param - learning_rate * (temp1 - temp2 + grad);
            
            // extract only variables involved in the penalization
            param_reg = param.head_rows(reg_p);
            
            // call proximal function
            if (transpose) {
                param_t = param_reg.t();
                
                if (penalty == 1) {
                    proximalFlat(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                }
                
                if (penalty == 2) {
                    proximalGraph(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                param_reg = param_t.t();
            } else {
                if (penalty == 1) {
                    proximalFlat(param_reg, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                }
                
                if (penalty == 2) {
                    proximalGraph(param_reg, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree(param_reg, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
            }
            
            param.head_rows(reg_p) = param_reg;
        }
        
        counter_outer += 1;
        Rcpp::Rcout << "\n Iteration " << counter_outer <<"\n";
        
        diff = arma::norm(param - param_old, "fro");
        diff = diff/(p*K);
        Rcpp::Rcout << "Frobenius norm of coefficient update \n" << diff <<"\n";
        
        param_history.push_back(param);
        
        if (diff < tolerance || counter_outer>= maxit) {
            break;
        }
    }
    
    arma::sp_mat param_sp(param);
    
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = param,          
        Rcpp::Named("Sparse Estimates") = param_sp, 
        Rcpp::Named("CoefficientHistory") = param_history 
    );
    return result;
}

arma::mat rowwise_softmax_baseline(const arma::mat& eta) {
    int n = eta.n_rows;
    int K = eta.n_cols;
    arma::mat P(n, K);
    
    for(int i = 0; i < n; ++i) {
        arma::rowvec current_eta = eta.row(i);
        
        // Stabilize using the log-sum-exp trick, including the baseline's score of 0
        double max_score = current_eta.max();
        if (max_score < 0.0) max_score = 0.0;
        
        arma::rowvec exp_scores = arma::exp(current_eta - max_score);
        double denominator = arma::accu(exp_scores) + exp(0.0 - max_score);
        P.row(i) = exp_scores / denominator;
    }
    return P;
}


// [[Rcpp::export]]
Rcpp::List MultinomLogisticSAGA(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int reg_p,
        int penalty,
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        const arma::mat& grp,
        const arma::mat& grpV,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double c_factor,
        double v_factor,
        double tolerance,
        int maxit,
        int ncores = 4,
        bool pos = false,
        bool verbose = false,
        bool save_history = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    if (n == 0) Rcpp::stop("Input data has 0 rows.");
    
    if (verbose) Rcpp::Rcout << "Detecting Lipschitz constant (L)..." << std::endl;
    double lambda_max = get_lambda_max(X);
    double L = (n > 0) ? (lambda_max / static_cast<double>(n)) : 1.0;
    if (L < 1e-8) L = 1.0;
    if (verbose) Rcpp::Rcout << "  - Lipschitz Constant (L) estimated as: " << L << std::endl;
    
    arma::mat param_z(p, K), param_y(p, K), param_x(p, K);
    if (param_start.isNotNull()) {
        arma::mat start_mat = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
        param_z = start_mat;
        param_y = start_mat;
    } else {
        param_z.zeros();
        param_y.zeros();
    }
    param_x = param_y;
    
    if (verbose) Rcpp::Rcout << "Initializing SAGA gradient table (vectorized)..." << std::endl;
    arma::cube grad_table(p, K, n);
    arma::mat grad_avg(p, K);
    
    // --- NEW: Fast, Vectorized SAGA Initialization ---
    arma::mat eta = X * param_y;
    if (!offset.is_empty()) {
        eta.each_col() += offset;
    }
    arma::mat P = rowwise_softmax_baseline(eta);
    arma::mat E = P;
    for (int i = 0; i < n; ++i) {
        if (Y(i) > 0) {
            E(i, static_cast<int>(Y(i)) - 1) -= 1.0;
        }
    }
    // Efficiently calculate the average gradient
    grad_avg = X.t() * E / n;
    // Efficiently populate the gradient table
    for (int i = 0; i < n; ++i) {
        grad_table.slice(i) = X.row(i).t() * E.row(i);
    }
    // --- End of New Initialization ---
    
    arma::mat param_y_old = param_y;
    arma::mat grad_new(p, K), grad_stoch(p, K), param_z_unprox(p, K);
    
    bool converged = false;
    int convergence_pass = -1;
    Rcpp::List param_history;
    double diff;
    long long max_updates = static_cast<long long>(maxit) * n;
    if (verbose) Rcpp::Rcout << "Starting optimization for " << maxit << " passes (" << max_updates << " total updates)." << std::endl;
    
    for (long long k = 0; k < max_updates; ++k) {
        Rcpp::checkUserInterrupt();
        
        double gamma_k = (static_cast<double>(k) + v_factor + 4.0) / (2.0 * c_factor * L);
        double tau_k = 1.0 / (c_factor * L * gamma_k);
        if (tau_k > 1.0) tau_k = 1.0;
        
        param_x = tau_k * param_z + (1.0 - tau_k) * param_y;
        
        int index = arma::randi(arma::distr_param(0, n - 1));
        
        const double* grad_old_ptr = grad_table.slice_memptr(index);
        arma::mat grad_old(grad_old_ptr, p, K);
        
        grad_multinom_loss2(X.row(index), Y(index), K, offset(index), param_x, p, grad_new);
        
        grad_stoch = grad_new - grad_old + grad_avg;
        
        grad_table.slice(index) = grad_new;
        grad_avg += (grad_new - grad_old) / static_cast<double>(n);
        
        param_z_unprox = param_z - gamma_k * grad_stoch;
        apply_proximal_step(param_z, param_z_unprox, gamma_k, reg_p, K, p, transpose, penalty, regul,
                            grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
        
        param_y = (1.0 - tau_k) * param_y + tau_k * param_z;
        
        if ((k + 1) % n == 0) {
            int current_pass = (k + 1) / n;
            if (save_history) {
                param_history.push_back(Rcpp::wrap(param_y));
            }
            diff = arma::norm(param_y - param_y_old, "fro") / (arma::norm(param_y_old, "fro") + 1e-10);
            if (verbose) {
                Rcpp::Rcout << "Pass " << current_pass << " | Rel.Chg: " << diff << std::endl;
            }
            if (diff < tolerance) {
                if (verbose) Rcpp::Rcout << "Convergence tolerance reached." << std::endl;
                converged = true;
                convergence_pass = current_pass;
                break;
            }
            param_y_old = param_y;
        }
    }
    
    if (!converged) {
        convergence_pass = maxit;
    }
    
    arma::sp_mat beta_sp(param_y);
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = Rcpp::wrap(param_y),
        Rcpp::Named("Sparse Estimates") = beta_sp,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_pass
    );
    
    if (save_history) {
        if (param_history.size() < convergence_pass && param_history.size() < maxit) {
            param_history.push_back(Rcpp::wrap(param_y));
        }
        result["History"] = param_history;
    }
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSARAH( 
        arma::mat X,
        arma::vec Y,
        arma::vec offset,
        int K,
        int reg_p,
        int penalty,
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        arma::mat grp,
        arma::mat grpV,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double learning_rate,
        double tolerance,
        int niter_inner,
        int maxit,
        int ncores
        ) {
    
    int p = X.n_cols;
    int n = X.n_rows;
    
    int index;
    int y_sample_int;
    double o_sample;
    
    arma::mat current_grad_estimate(p, K);
    // Temporaries for stochastic gradients
    arma::mat stoch_grad_at_w_curr(p, K); 
    arma::mat stoch_grad_at_w_prev(p, K); 
    
    arma::mat param(p, K, arma::fill::zeros);   
    arma::mat param_prev_inner_step(p, K); 
    
    arma::mat param_t; 
    if (transpose) {
        param_t.set_size(K, reg_p);
    }
    
    double diff;
    int counter_outer = 0;
    std::vector<arma::mat> param_history;
    
    while (true) {
        arma::mat param_at_epoch_start = param; // This is w_0^(s) for current epoch s
        
        current_grad_estimate.zeros();
        arma::mat temp_for_full_grad_sum(p, K); 
        for (int i_full = 0; i_full < n; ++i_full) {
            const auto& x_sample_view_full = X.row(i_full);
            const int& y_sample_ref_full = Y(i_full);
            const double& o_sample_ref_full = offset(i_full);
            grad_multinom_loss2(x_sample_view_full, y_sample_ref_full, K, o_sample_ref_full,
                                param_at_epoch_start, p, temp_for_full_grad_sum);
            current_grad_estimate += temp_for_full_grad_sum;
        }
        if (n > 0) {
            current_grad_estimate /= n; 
        }
        
   
        param_prev_inner_step = param_at_epoch_start;
        
        for (int t = 0; t < niter_inner; ++t) {
           
            if (t > 0) {
                index = arma::randi(arma::distr_param(0, n - 1));
                const auto& x_sample_view_inner = X.row(index);
                y_sample_int = static_cast<int>(Y(index));
                o_sample = offset(index);

                grad_multinom_loss2(x_sample_view_inner, y_sample_int, K, o_sample,
                                    param, p, stoch_grad_at_w_curr);  
                grad_multinom_loss2(x_sample_view_inner, y_sample_int, K, o_sample,
                                    param_prev_inner_step, p, stoch_grad_at_w_prev); 
                
                current_grad_estimate = stoch_grad_at_w_curr - stoch_grad_at_w_prev + current_grad_estimate;
            }
            
            // Store current param (w_t^(s)) to be used as w_{t-1}^(s) 
            param_prev_inner_step = param;
            
            param = param - learning_rate * current_grad_estimate; 
            
            auto param_reg_view = param.head_rows(reg_p);
            if (transpose) {
                param_t = param_reg_view.t(); 
                if (penalty == 1) {
                    proximalFlat(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                } else if (penalty == 2) {
                    proximalGraph(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                } else if (penalty == 3) {
                    proximalTree(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                param.head_rows(reg_p) = param_t.t(); 
            } else { 
                arma::mat param_reg_copy = param_reg_view; // Explicit copy
                if (penalty == 1) {
                    proximalFlat(param_reg_copy, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate );
                } else if (penalty == 2) {
                    proximalGraph(param_reg_copy, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                } else if (penalty == 3) {
                    proximalTree(param_reg_copy, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                param.head_rows(reg_p) = param_reg_copy; 
            }
           
        } 
        
        counter_outer += 1;
        param_history.push_back(param);
        
        double norm_param_at_epoch_start = arma::norm(param_at_epoch_start, "fro");
        diff = arma::norm(param - param_at_epoch_start, "fro") / (norm_param_at_epoch_start + 1e-10);
        
        diff = diff / ( static_cast<double>(p * K) + 1e-10 );
        
        Rcpp::Rcout << "\n Iteration " << counter_outer << "\n";
        Rcpp::Rcout << "Scaled relative Frobenius norm of coefficient update (vs epoch start) \n" << diff << "\n";
        
        if (diff < tolerance || counter_outer >= maxit) {
            break;
        }
    } 
    arma::sp_mat param_sp(param);
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Sparse Estimates") = param_sp,
        Rcpp::Named("CoefficientHistory") = param_history
    );
    return result;
}

arma::rowvec calculate_probabilities_pcd(const arma::rowvec& eta_obs_row) {
    int K_local = eta_obs_row.n_elem;
    arma::rowvec pi_obs(K_local);
    // Numerically stable calculation of exp(eta) / (1 + sum(exp(eta)))
    double max_eta = eta_obs_row.max();
    arma::rowvec exp_eta_shifted = arma::exp(eta_obs_row - max_eta);
    double sum_exp_shifted = arma::accu(exp_eta_shifted);
    double denominator_shifted = std::exp(-max_eta) + sum_exp_shifted;
    
    if (denominator_shifted > 1e-15) {
        pi_obs = exp_eta_shifted / denominator_shifted;
    } else {
        pi_obs.fill(1.0 / K_local); // Fallback
    }
    return pi_obs;
}


// [[Rcpp::export]]
Rcpp::List MultinomLogisticPCD(
        arma::mat X,
        arma::vec Y,
        arma::vec offset_vec,
        int K_classes, // Renamed to avoid conflict with local K_cols_prox in calls
        int reg_p,
        int penalty_code,
        std::string regul,
        bool transpose,
        Rcpp::IntegerVector grp_id,
        Rcpp::NumericVector etaG,
        arma::mat grp_mat,
        arma::mat grpV_mat,
        Rcpp::IntegerVector own_var,
        Rcpp::IntegerVector N_own_var,
        double lam1,
        double lam2,
        double lam3,
        double learning_rate,
        double tolerance,
        int maxit,
        int ncores,
        bool pos = false) {
    
    int p_total = X.n_cols;
    int n_obs = X.n_rows;
    
    if (reg_p > p_total || reg_p < 0) {
        Rcpp::stop("reg_p must be between 0 and p_total.");
    }
    if (penalty_code != 1 && !(regul == "none" || regul == "L0" || regul == "L1" || regul == "L2" || regul == "ElasticNet" || regul == "GroupLasso")) {
        Rcpp::warning("This PCD implementation primarily supports penalties handled by proximalFlat. Graph/Tree penalties might not behave as expected without specialized PCD updates.");
    }
    
    arma::mat param(p_total, K_classes, arma::fill::zeros);
    arma::mat param_old_epoch(p_total, K_classes);
    
    arma::mat param_j_block_for_prox_op; // Reused for proximal calls
    
    std::vector<arma::mat> param_history;
    int counter_outer = 0;
    double diff;
    
    while (true) {
        param_old_epoch = param;
        
        for (int j_feat = 0; j_feat < p_total; ++j_feat) {
            arma::rowvec partial_grad_for_row_j(K_classes, arma::fill::zeros);
            
            for (int i_obs = 0; i_obs < n_obs; ++i_obs) {
                const auto& x_obs_i_row = X.row(i_obs);
                double x_obs_ij_val = x_obs_i_row(j_feat);
                
                if (std::abs(x_obs_ij_val) < 1e-12) continue;
                
                int y_obs_i = static_cast<int>(Y(i_obs));
                double offset_i = offset_vec(i_obs);
                
                arma::rowvec eta_i_row = x_obs_i_row * param;
                eta_i_row += offset_i;
                
                arma::rowvec prob_i_row = calculate_probabilities_pcd(eta_i_row);
                
                arma::rowvec error_vec = prob_i_row;
                if (y_obs_i >= 1 && y_obs_i <= K_classes) {
                    error_vec(y_obs_i - 1) -= 1.0;
                }
                partial_grad_for_row_j += x_obs_ij_val * error_vec;
            }
            
            if (n_obs > 0) {
                partial_grad_for_row_j /= n_obs;
            }
            
            arma::rowvec Bj_unprox = param.row(j_feat) - learning_rate * partial_grad_for_row_j;
            
            if (j_feat < reg_p) {
                Rcpp::IntegerVector current_grp_id_for_prox;
                if(grp_id.length() > j_feat) {
                    current_grp_id_for_prox = Rcpp::IntegerVector::create(grp_id[j_feat]);
                } else {
                    current_grp_id_for_prox = Rcpp::IntegerVector::create(0);
                }
                
                // Variables to hold dimensions as lvalues
                int n_rows_prox, n_cols_prox;
                
                if (transpose) {
                    param_j_block_for_prox_op = Bj_unprox.t(); // K_classes x 1 matrix
                    n_rows_prox = K_classes;  // Dimension of param_j_block_for_prox_op
                    n_cols_prox = 1;      // Dimension of param_j_block_for_prox_op
                    
                    if (penalty_code == 1) {
                        proximalFlat(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, current_grp_id_for_prox, ncores,
                                      lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
                    } else if (penalty_code == 2) {
                        proximalGraph(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, grpV_mat, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    } else if (penalty_code == 3) {
                        proximalTree(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    }
                    param.row(j_feat) = param_j_block_for_prox_op.t();
                } else {
                    param_j_block_for_prox_op = Bj_unprox; // 1 x K_classes matrix
                    n_rows_prox = 1;         // Dimension of param_j_block_for_prox_op
                    n_cols_prox = K_classes; // Dimension of param_j_block_for_prox_op
                    
                    if (penalty_code == 1) {
                        proximalFlat(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, current_grp_id_for_prox, ncores,
                                      lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
                    } else if (penalty_code == 2) {
                        proximalGraph(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, grpV_mat, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    } else if (penalty_code == 3) {
                        proximalTree(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    }
                    param.row(j_feat) = param_j_block_for_prox_op;
                }
            } else {
                param.row(j_feat) = Bj_unprox;
            }
        } // End loop over predictors j_feat
        
        counter_outer += 1;
        param_history.push_back(param);
        
        double norm_param_old_epoch = arma::norm(param_old_epoch, "fro");
        
        diff = arma::norm(param - param_old_epoch, "fro") / (norm_param_old_epoch + 1e-10);
        diff = diff / (static_cast<double>(p_total * K_classes) + 1e-10);
        
        Rcpp::Rcout << "\n PCD Epoch " << counter_outer << "\n";
        Rcpp::Rcout << "Scaled relative Frobenius norm of coefficient update \n" << diff << "\n";
        
        if (diff < tolerance || counter_outer >= maxit) {
            break;
        }
    } // End outer loop
    
    arma::sp_mat param_sp(param);
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Sparse Estimates") = param_sp,
        Rcpp::Named("CoefficientHistory") = param_history
    );
    return result;
}




arma::rowvec calculate_probabilities_pcd2(const arma::rowvec& eta_obs_row, int K_classes_for_helper) {
    if (K_classes_for_helper <= 0) {
        if (eta_obs_row.n_elem > 0) K_classes_for_helper = eta_obs_row.n_elem;
        else Rcpp::stop("K_classes_for_helper must be positive in calculate_probabilities_pcd.");
    }
    if (eta_obs_row.n_elem != K_classes_for_helper && K_classes_for_helper > 0) {
        Rcpp::stop("Dimension mismatch in calculate_probabilities_pcd: eta_obs_row.n_elem != K_classes_for_helper");
    }
    
    arma::rowvec pi_obs(K_classes_for_helper);
    double max_eta = eta_obs_row.max();
    arma::rowvec exp_eta_shifted = arma::exp(eta_obs_row - max_eta);
    double sum_exp_shifted = arma::accu(exp_eta_shifted);
    double denominator_shifted = std::exp(-max_eta) + sum_exp_shifted;
    
    if (denominator_shifted > 1e-100) {
        pi_obs = exp_eta_shifted / denominator_shifted;
    } else {
        pi_obs.fill(1.0 / K_classes_for_helper);
    }
    return pi_obs;
}

// Simplified KKT check for activating a zeroed feature
bool check_kkt_for_activation(
        const arma::rowvec& partial_grad_for_row_j,
        const std::string& regul,
        double lam1_original,
        const Rcpp::IntegerVector& /* grp_id_for_feature */, // Placeholder
        double kkt_abs_tolerance) {
    if (regul == "L1" || regul == "ElasticNet" || regul == "elastic-net") {
        return arma::abs(partial_grad_for_row_j).max() > lam1_original + kkt_abs_tolerance;
    } else if (regul == "GroupLasso" || regul == "group-lasso-l2") {
        return arma::norm(partial_grad_for_row_j, 2) > lam1_original + kkt_abs_tolerance;
    } else if (regul == "L2" || regul == "none" || regul == "L0" || regul == "l2") {
        return arma::norm(partial_grad_for_row_j, 2) > kkt_abs_tolerance;
    }
    return false;
}

// Proximal functions (MUST modify U in-place)
// Ensure these are your actual working implementations (e.g., _nospams or SPAMS wrappers)
void proximalFlat3(arma::mat& U, int n_r, int n_c, const std::string& reg, const Rcpp::IntegerVector& g_id, int ncrs, double eff_lam1, double eff_lam2, double eff_lam3, bool pos_flag) {
    // Example: if (reg == "L1") { U = arma::sign(U) % arma::max(arma::abs(U) - eff_lam1, arma::zeros(U.n_rows, U.n_cols)); } ...
    // Rcpp::Rcout << "Warning: proximalFlat called but using placeholder logic." << std::endl; // Placeholder
    if (reg == "L1" || reg == "ElasticNet" || reg == "elastic-net") {
        if (eff_lam1 > 0.0) U = arma::sign(U) % arma::max(arma::abs(U) - eff_lam1, arma::zeros(U.n_rows, U.n_cols));
    }
    if (reg == "L2" || reg == "ElasticNet" || reg == "elastic-net") {
        if (eff_lam2 > 0.0) U /= (1.0 + 2.0 * eff_lam2); 
    }
    if (reg == "GroupLasso" || reg == "group-lasso-l2") {
        if (eff_lam1 > 0.0 && U.n_rows == 1) { 
            double row_norm = arma::norm(U, 2);
            if (row_norm > 1e-12) U *= std::max(0.0, 1.0 - eff_lam1 / row_norm);
            else U.zeros();
        } else if (eff_lam1 > 0.0 && U.n_cols == 1) { // For transposed case
            double col_norm = arma::norm(U,2);
            if(col_norm > 1e-12) U *= std::max(0.0, 1.0 - eff_lam1 / col_norm);
            else U.zeros();
        }
    }
    if (pos_flag) U.elem( arma::find(U < 0.0) ).zeros();
}


// Rcpp::List MultinomLogisticCCD(
//         arma::mat X,
//         arma::vec Y,
//         arma::vec offset_vec,
//         int K_classes,
//         int reg_p,
//         int penalty_code,
//         std::string regul,
//         bool transpose_prox_input,
//         Rcpp::IntegerVector grp_id,
//         Rcpp::NumericVector etaG,
//         arma::mat grp_mat,
//         arma::mat grpV_mat,
//         Rcpp::IntegerVector own_var,
//         Rcpp::IntegerVector N_own_var,
//         double lam1,
//         double lam2,
//         double lam3,
//         double learning_rate_scale = 1.0,
//         double tolerance = 1e-5,
//         double kkt_abs_check_tol = 1e-6, // This is the parameter name for KKT absolute tolerance
//         int maxit = 100,
//         int max_ccd_passes_active_set = 5,
//         int ncores = 1,
//         bool pos = false
// ) {
//     int p_total = X.n_cols;
//     int n_obs = X.n_rows;
//     
//     if (reg_p > p_total || reg_p < 0) Rcpp::stop("reg_p must be between 0 and p_total.");
//     if (K_classes <= 0) Rcpp::stop("K_classes must be positive.");
//     
//     arma::mat param(p_total, K_classes, arma::fill::zeros);
//     arma::mat param_old_epoch(p_total, K_classes);
//     arma::mat param_j_block_for_prox_op;
//     
//     std::vector<arma::mat> param_history;
//     param_history.reserve(maxit);
//     int counter_outer = 0;
//     double diff_outer;
//     
//     arma::vec sum_Xj_sq(p_total, arma::fill::zeros);
//     for (int j = 0; j < p_total; ++j) {
//         sum_Xj_sq(j) = arma::accu(arma::square(X.col(j)));
//     }
//     
//     std::set<arma::uword> active_set_s;
//     for(int j=0; j<reg_p; ++j) active_set_s.insert(j);
//     
//     for (counter_outer = 0; counter_outer < maxit; ++counter_outer) {
//         param_old_epoch = param;
//         
//         bool active_set_cycling_converged = false; 
//         int  current_ccd_pass_count = 0;
//         
//         while(!active_set_cycling_converged && current_ccd_pass_count < max_ccd_passes_active_set) {
//             current_ccd_pass_count++;
//             arma::mat param_before_this_pass = param; 
//             
//             if (active_set_s.empty() && reg_p > 0) {
//                 break; 
//             }
//             
//             std::vector<arma::uword> active_indices_vec(active_set_s.begin(), active_set_s.end());
//             
//             for (arma::uword j_feat : active_indices_vec) {
//                 for (int i_obs = 0; i_obs < n_obs; ++i_obs) {
//                     const auto& x_obs_i_row = X.row(i_obs);
//                     double x_obs_ij_val = x_obs_i_row(j_feat);
//                     if (std::abs(x_obs_ij_val) < 1e-12) continue;
//                     int y_obs_i = static_cast<int>(Y(i_obs));
//                     double offset_i = offset_vec(i_obs);
//                     arma::rowvec eta_i_row = x_obs_i_row * param; 
//                     eta_i_row += offset_i;
//                     arma::rowvec prob_i_row = calculate_probabilities_pcd2(eta_i_row, K_classes);
//                     arma::rowvec error_vec = prob_i_row;
//                     if (y_obs_i >= 1 && y_obs_i <= K_classes) error_vec(y_obs_i - 1) -= 1.0;
//                     partial_grad_for_row_j += x_obs_ij_val * error_vec;
//                 }
//                 if (n_obs > 0) partial_grad_for_row_j /= static_cast<double>(n_obs);
//                 
//                 double L_j_approx = (sum_Xj_sq(j_feat) / (n_obs > 0 ? static_cast<double>(n_obs) : 1.0)) * 0.5 + 1e-8; 
//                 if (j_feat < reg_p && (regul == "L2" || regul == "ElasticNet" || regul == "elastic-net")) {
//                     L_j_approx += 2.0 * lam2; 
//                 }
//                 double step_j = learning_rate_scale / std::max(L_j_approx, 1e-8);
//                 arma::rowvec Bj_unprox = param.row(j_feat) - step_j * partial_grad_for_row_j;
//                 
//                 param_j_block_for_prox_op = Bj_unprox; 
//                 Rcpp::IntegerVector current_grp_id_for_prox = Rcpp::IntegerVector::create(grp_id.length() > j_feat ? grp_id[j_feat] : 0);
//                 double eff_lam1 = lam1 * step_j, eff_lam2 = lam2 * step_j, eff_lam3 = lam3 * step_j;
//                 
//                 if (transpose_prox_input) { 
//                     param_j_block_for_prox_op = Bj_unprox.t(); int nr=K_classes, nc=1;
//                     if(penalty_code==1) proximalFlat3(param_j_block_for_prox_op,nr,nc,regul,current_grp_id_for_prox,ncores,eff_lam1,eff_lam2,eff_lam3,pos);
//                     // else if(penalty_code==2) proximalGraph(param_j_block_for_prox_op,nr,nc,regul,grp_mat,grpV_mat,etaG,ncores,eff_lam1,eff_lam2,pos);
//                     // else if(penalty_code==3) proximalTree(param_j_block_for_prox_op,nr,nc,regul,grp_mat,etaG,own_var,N_own_var,ncores,eff_lam1,eff_lam2,pos);
//                     param.row(j_feat) = param_j_block_for_prox_op.t();
//                 } else { 
//                     param_j_block_for_prox_op = Bj_unprox; int nr=1, nc=K_classes;
//                     if(penalty_code==1) proximalFlat3(param_j_block_for_prox_op,nr,nc,regul,current_grp_id_for_prox,ncores,eff_lam1,eff_lam2,eff_lam3,pos);
//                     // else if(penalty_code==2) proximalGraph(param_j_block_for_prox_op,nr,nc,regul,grp_mat,grpV_mat,etaG,ncores,eff_lam1,eff_lam2,pos);
//                     // else if(penalty_code==3) proximalTree(param_j_block_for_prox_op,nr,nc,regul,grp_mat,etaG,own_var,N_own_var,ncores,eff_lam1,eff_lam2,pos);
//                     param.row(j_feat) = param_j_block_for_prox_op;
//                 }
//             } 
//             
//             double active_pass_param_diff = arma::norm(param - param_before_this_pass, "fro");
//             if (param_before_this_pass.n_elem > 0) { 
//                 double norm_param_before_pass = arma::norm(param_before_this_pass, "fro");
//                 if (norm_param_before_pass > 1e-10) active_pass_param_diff /= (norm_param_before_pass + 1e-10);
//             }
//             
//             // Corrected: Use a fraction of the main tolerance for inner active set stability
//             if (active_pass_param_diff < tolerance * 0.1) { 
//                 active_set_cycling_converged = true; 
//             }
//         } 
//         
//         bool kkt_scan_added_new_vars = false;
//         if (reg_p > 0) {
//             for (int j_kkt = 0; j_kkt < reg_p; ++j_kkt) {
//                 bool is_in_active_set = active_set_s.count(j_kkt);
//                 bool is_param_j_zero = arma::accu(arma::abs(param.row(j_kkt))) < 1e-8;
//                 
//                 if (is_in_active_set && !is_param_j_zero) continue; 
//                 
//                 arma::rowvec partial_grad_kkt(K_classes, arma::fill::zeros);
//                 for (int i_obs = 0; i_obs < n_obs; ++i_obs) { 
//                     const auto& x_obs_i_row = X.row(i_obs); double x_obs_ij_val = x_obs_i_row(j_kkt);
//                     if (std::abs(x_obs_ij_val) < 1e-12) continue;
//                     int y_obs_i = static_cast<int>(Y(i_obs)); double offset_i = offset_vec(i_obs);
//                     arma::rowvec eta_i_row = x_obs_i_row * param; eta_i_row += offset_i;
//                     arma::rowvec prob_i_row = calculate_probabilities_pcd2(eta_i_row, K_classes);
//                     arma::rowvec error_vec = prob_i_row;
//                     if (y_obs_i >= 1 && y_obs_i <= K_classes) error_vec(y_obs_i - 1) -= 1.0;
//                     partial_grad_kkt += x_obs_ij_val * error_vec;
//                 }
//                 if (n_obs > 0) partial_grad_kkt /= static_cast<double>(n_obs);
//                 
//                 Rcpp::IntegerVector current_grp_id_for_kkt = Rcpp::IntegerVector::create(grp_id.length() > j_kkt ? grp_id[j_kkt] : 0);
//                 // Corrected: Use kkt_abs_check_tol parameter directly
//                 if (check_kkt_for_activation(partial_grad_kkt, regul, lam1, current_grp_id_for_kkt, kkt_abs_check_tol)) {
//                     if (active_set_s.find(j_kkt) == active_set_s.end()) { 
//                         active_set_s.insert(j_kkt);
//                         kkt_scan_added_new_vars = true;
//                     }
//                 }
//             } 
//         }
//         
//         // Update Unregularized Predictors
//         for (int j_feat = reg_p; j_feat < p_total; ++j_feat) {
//             arma::rowvec partial_grad_for_row_j(K_classes, arma::fill::zeros);
//             for (int i_obs = 0; i_obs < n_obs; ++i_obs) {
//                 const auto& x_obs_i_row = X.row(i_obs); double x_obs_ij_val = x_obs_i_row(j_feat);
//                 if (std::abs(x_obs_ij_val) < 1e-12) continue;
//                 int y_obs_i = static_cast<int>(Y(i_obs)); double offset_i = offset_vec(i_obs);
//                 arma::rowvec eta_i_row = x_obs_i_row * param; eta_i_row += offset_i;
//                 arma::rowvec prob_i_row = calculate_probabilities_pcd2(eta_i_row, K_classes);
//                 arma::rowvec error_vec = prob_i_row;
//                 if (y_obs_i >= 1 && y_obs_i <= K_classes) error_vec(y_obs_i - 1) -= 1.0;
//                 partial_grad_for_row_j += x_obs_ij_val * error_vec;
//             }
//             if (n_obs > 0) partial_grad_for_row_j /= static_cast<double>(n_obs);
//             double L_j_approx = (sum_Xj_sq(j_feat) / (n_obs > 0 ? static_cast<double>(n_obs) : 1.0)) * 0.5 + 1e-8;
//             double step_j = learning_rate_scale / std::max(L_j_approx, 1e-8);
//             param.row(j_feat) = param.row(j_feat) - step_j * partial_grad_for_row_j;
//         }
//         
//         param_history.push_back(param);
//         
//         // Overall Epoch Convergence Check
//         double norm_param_old_epoch = arma::norm(param_old_epoch, "fro");
//         if (norm_param_old_epoch > 1e-10) {
//             diff_outer = arma::norm(param - param_old_epoch, "fro") / (norm_param_old_epoch + 1e-10);
//         } else {
//             diff_outer = arma::norm(param - param_old_epoch, "fro");
//         }
//         
//         if (counter_outer % 1 == 0 || diff_outer < tolerance || kkt_scan_added_new_vars) {
//             Rcpp::Rcout << "Epoch " << counter_outer +1 << " Overall Rel.Diff: " << diff_outer 
//                         << ". ActiveSetSize: " << active_set_s.size() 
//                         << ". KKT added: " << kkt_scan_added_new_vars << std::endl;
//         }
//         
//         if (!kkt_scan_added_new_vars && diff_outer < tolerance) {
//             Rcpp::Rcout << "Converged." << std::endl;
//             break; 
//         }
//         // No counter_outer >= maxit - 1 check here, for loop handles maxit
//         Rcpp::checkUserInterrupt();
//     } // --- End Outer Epoch Loop ---
//     
//     arma::sp_mat param_sp(param);
//     Rcpp::List result = Rcpp::List::create(
//         Rcpp::Named("Estimates") = param,
//         Rcpp::Named("SparseEstimates") = param_sp,
//         Rcpp::Named("CoefficientHistory") = Rcpp::wrap(param_history)
//     );
//     return result;
// }



double soft_threshold(double z, double gamma) {
    if (z > gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}


// [[Rcpp::export]]
Rcpp::List MultinomLogisticCCD(
         const arma::mat& X,
         const arma::vec& Y,
         const arma::vec& offset_vec,
         int K_classes,
         int reg_p,
         int penalty_code,
         const std::string& regul,
         bool transpose_prox_input,
         const Rcpp::IntegerVector& grp_id,
         const Rcpp::NumericVector& etaG,
         const arma::mat& grp_mat,
         const arma::mat& grpV_mat,
         const Rcpp::IntegerVector& own_var,
         const Rcpp::IntegerVector& N_own_var,
         double lam1,
         double lam2,
         double lam3,
         double learning_rate_scale,
         double tolerance,
         double kkt_abs_check_tol,
         int maxit,
         int max_ccd_passes_active_set,
         int ncores,
         bool pos
 ) {
     int p_total = X.n_cols;
     int n_obs = X.n_rows;
     int K_eff = K_classes - 1;
     
     if (reg_p > p_total || reg_p < 0) Rcpp::stop("reg_p must be between 0 and p_total.");
     if (K_classes <= 1) Rcpp::stop("K_classes must be greater than 1.");
     
     // --- 1. Initialization ---
     arma::mat param(p_total, K_eff, arma::fill::zeros);
     arma::rowvec beta0(K_eff);
     
     arma::mat Y_mat(n_obs, K_eff, arma::fill::zeros);
     double n_class0 = 0;
     for(int i = 0; i < n_obs; ++i) {
         if(Y(i) > 0 && Y(i) < K_classes) {
             Y_mat(i, Y(i) - 1) = 1.0;
         } else {
             n_class0++;
         }
     }
     
     double q0 = n_class0 / n_obs;
     q0 = std::max(q0, 1e-5);
     for (int k = 0; k < K_eff; ++k) {
         double qk = arma::mean(Y_mat.col(k));
         qk = std::max(qk, 1e-5);
         beta0(k) = std::log(qk / q0);
     }
     
     std::set<arma::uword> active_set;
     for(int j = 0; j < reg_p; ++j) active_set.insert(j);
     
     // --- 2. Main Outer Loop (IRLS Iterations) ---
     for (int counter_outer = 0; counter_outer < maxit; ++counter_outer) {
         arma::mat param_old_epoch = param;
         arma::rowvec beta0_old_epoch = beta0;
         
         arma::mat eta = X * param;
         eta.each_row() += beta0;
         eta.each_col() += offset_vec;
         
         // CORRECTED: Stabilize softmax calculation to prevent numerical overflow
         arma::vec max_eta = arma::max(eta, 1);
         eta.each_col() -= max_eta;
         arma::mat exp_eta = arma::exp(eta);
         arma::vec denom = 1.0 + arma::sum(exp_eta, 1);
         arma::mat prob = exp_eta.each_col() / denom;
         
         arma::mat residuals = Y_mat - prob;
         arma::mat weights = prob % (1.0 - prob);
         
         int current_ccd_pass_count = 0;
         while(current_ccd_pass_count < max_ccd_passes_active_set) {
             current_ccd_pass_count++;
             arma::mat param_before_pass = param;
             arma::rowvec beta0_before_pass = beta0;
             
             for (int k = 0; k < K_eff; ++k) {
                 double sum_w = arma::sum(weights.col(k));
                 double delta_b0 = (sum_w > 1e-8) ? arma::sum(residuals.col(k)) / sum_w : 0.0;
                 
                 if (std::abs(delta_b0) > 1e-10) {
                     beta0(k) += delta_b0;
                     arma::vec delta_eta_k = arma::ones<arma::vec>(n_obs) * delta_b0;
                     for (int l = 0; l < K_eff; ++l) {
                         if (l == k) {
                             residuals.col(l) -= weights.col(k) % delta_eta_k;
                         } else {
                             residuals.col(l) += (prob.col(l) % prob.col(k)) % delta_eta_k;
                         }
                     }
                 }
             }
             
             std::vector<arma::uword> update_indices(active_set.begin(), active_set.end());
             for(int j = reg_p; j < p_total; ++j) update_indices.push_back(j);
             
             for (arma::uword j : update_indices) {
                 for (int k = 0; k < K_eff; ++k) {
                     double grad = arma::dot(X.col(j), residuals.col(k));
                     double hess = arma::dot(weights.col(k), arma::square(X.col(j)));
                     
                     double new_beta;
                     if (j >= reg_p) {
                         new_beta = param(j, k) + (hess > 1e-8 ? grad / hess : 0.0);
                     } else {
                         double u = grad + hess * param(j, k);
                         new_beta = soft_threshold(u, lam1) / (hess + lam2);
                     }
                     
                     if (pos && new_beta < 0) new_beta = 0;
                     double delta = new_beta - param(j, k);
                     
                     if (std::abs(delta) > 1e-10) {
                         param(j, k) = new_beta;
                         arma::vec delta_eta_k = delta * X.col(j);
                         for (int l = 0; l < K_eff; ++l) {
                             if (l == k) {
                                 residuals.col(l) -= weights.col(k) % delta_eta_k;
                             } else {
                                 residuals.col(l) += (prob.col(l) % prob.col(k)) % delta_eta_k;
                             }
                         }
                     }
                 }
             }
             
             double pass_diff = arma::norm(param - param_before_pass, "fro") + arma::norm(beta0 - beta0_before_pass, "fro");
             double pass_norm = arma::norm(param_before_pass, "fro") + arma::norm(beta0_before_pass, "fro");
             if (pass_diff / (pass_norm + 1e-10) < tolerance * 0.01) break;
         }
         
         bool kkt_added_new_vars = false;
         for (int j = 0; j < reg_p; ++j) {
             if (active_set.count(j)) continue;
             double max_grad = 0.0;
             for (int k = 0; k < K_eff; ++k) {
                 max_grad = std::max(max_grad, std::abs(arma::dot(X.col(j), residuals.col(k))));
             }
             if (max_grad > lam1 + kkt_abs_check_tol) {
                 active_set.insert(j);
                 kkt_added_new_vars = true;
             }
         }
         
         double epoch_diff = arma::norm(param - param_old_epoch, "fro") + arma::norm(beta0 - beta0_old_epoch, "fro");
         double epoch_norm = arma::norm(param_old_epoch, "fro") + arma::norm(beta0_old_epoch, "fro");
         
         if (!kkt_added_new_vars && (epoch_diff / (epoch_norm + 1e-10) < tolerance)) {
             Rcpp::Rcout << "Converged after " << counter_outer + 1 << " epochs." << std::endl;
             break;
         }
         if (counter_outer == maxit - 1) {
             Rcpp::Rcout << "Maximum epochs reached." << std::endl;
         }
         Rcpp::checkUserInterrupt();
     }
     
     arma::sp_mat param_sp(param);
     return Rcpp::List::create(
         Rcpp::Named("Estimates") = param,
         Rcpp::Named("Intercepts") = beta0,
         Rcpp::Named("SparseEstimates") = param_sp
     );
 }