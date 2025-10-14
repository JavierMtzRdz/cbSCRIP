
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

arma::mat grad_multinom_loss(
        const arma::rowvec& x,
        int y,
        int K,
        double offset,
        const arma::mat& param,
        int p
) {
    arma::mat grad_out(p, K, arma::fill::zeros);
    
    // Linear scores
    arma::vec linear_scores = param.t() * x.t(); // K x 1
    double baseline_score = offset;
    
    double max_score = std::max(linear_scores.max(), baseline_score);
    arma::vec exp_scores = arma::exp(linear_scores - max_score);
    double exp_baseline = std::exp(baseline_score - max_score);
    
    double denom = arma::accu(exp_scores) + exp_baseline;
    if (denom < 1e-12 || !std::isfinite(denom)) return grad_out;
    
    arma::vec probs = exp_scores / denom;
    grad_out = x.t() * probs.t(); 
    
    if (y > 0 && y <= K)
        grad_out.col(y - 1) -= x.t();
    
    return grad_out;
}


arma::mat grad_multinom_single(
        const arma::rowvec& x,
        int y,
        const arma::vec& offset_val,
        const arma::mat& param,
        int K) {
    
    arma::vec linear_scores = param.t() * x.t(); 
    
    // Soft-clip offset
    double baseline_score = offset_val(0);
    if (std::abs(baseline_score) > 20.0) {
        baseline_score = 20.0 * std::tanh(baseline_score / 20.0);
    }
    
    // softmax computation
    double max_score = std::max(linear_scores.max(), baseline_score);
    arma::vec exp_scores = arma::exp(linear_scores - max_score);
    double denom = arma::accu(exp_scores) + std::exp(baseline_score - max_score)+ 1e-12;
    
    arma::mat grad_out(param.n_rows, K, arma::fill::zeros);
    if (denom < 1e-10 || !std::isfinite(denom)) {
        return grad_out; // Return zero gradient on numerical error
    }
    
    arma::vec pi = exp_scores / denom; 
    
    // Gradient 
    grad_out = x.t() * pi.t(); 
    
    if (y > 0 && y <= K) {
        grad_out.col(y - 1) -= x.t();
    }
    return grad_out;
}

arma::mat compute_full_grad_vectorized(
        const arma::mat& X,
        const arma::vec& offsets,
        const arma::mat& param,
        const arma::vec& Y
) {
    const int n = X.n_rows;
    const int K = param.n_cols;
    
    arma::mat S = X * param; 
    arma::vec max_scores = arma::max(S, 1);
    max_scores = arma::max(max_scores, offsets); 
    
    S.each_col() -= max_scores;
    arma::mat expS = arma::exp(S);
    arma::vec exp_base = arma::exp(offsets - max_scores);
    arma::vec denom = arma::sum(expS, 1) + exp_base;
    
    expS.each_col() /= denom;
    
    arma::mat Yone(n, K, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
        if (Y(i) > 0 && Y(i) <= K)
            Yone(i, Y(i) - 1) = 1.0;
    }
    
    arma::mat grad = X.t() * (expS - Yone);
    grad /= static_cast<double>(n);
    grad.replace(arma::datum::nan, 0.0);
    return grad;
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
    
    if (lam2 > 0 && lam2 < 1e-5) {
        lam2 = 1e-5;
    }
    
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

static double get_lambda_max(const arma::mat& X, int max_iters = 100) {
    if (X.n_cols == 0) return 0.0;
    arma::vec v = arma::randn<arma::vec>(X.n_cols);
    v /= arma::norm(v);
    
    for (int i = 0; i < max_iters; ++i) {
        arma::vec w = X.t() * (X * v);
        double norm_w = arma::norm(w);
        if (norm_w < 1e-12) return 0.0;
        v = w / norm_w;
    }
    return arma::dot(v, X.t() * (X * v));
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


// Rcpp::List MultinomLogisticAcc(
//         const arma::mat& X,
//         const arma::vec& Y,
//         const arma::vec& offset,
//         int K,
//         int reg_p,
//         int penalty,
//         std::string regul,
//         bool transpose,
//         Rcpp::IntegerVector grp_id,
//         Rcpp::NumericVector etaG,
//         const arma::mat& grp,
//         const arma::mat& grpV,
//         Rcpp::IntegerVector own_var,
//         Rcpp::IntegerVector N_own_var,
//         double lam1,
//         double lam2,
//         double lam3,
//         double c_factor,  
//         double v_factor, 
//         double tolerance,
//         int niter_inner,
//         int maxit,
//         int ncores,
//         bool save_history = false,
//         bool verbose = false,
//         bool pos = false,
//         Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
//     
//     const int p = X.n_cols;
//     const int n = X.n_rows;
//     
//     if (verbose) Rcpp::Rcout << "Detecting Lipschitz constant (L)..." << std::endl;
//     double lambda_max = get_lambda_max(X);
//     double L = (n > 0) ? (lambda_max / (4.0 * static_cast<double>(n))) : 1.0;
//     if (L < 1e-8) L = 1.0;
//     if (verbose) Rcpp::Rcout << "  - Lipschitz Constant (L) estimated as: " << L << std::endl;
//     
//     // --- Initialize iterates for the accelerated algorithm ---
//     arma::mat param_z(p, K), param_y(p, K), param_x(p, K);
//     if (param_start.isNotNull()) {
//         arma::mat start_mat = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
//         param_z = start_mat;
//         param_y = start_mat;
//         param_x = start_mat;
//     } else {
//         param_z.zeros();
//         param_y.zeros();
//         param_x.zeros();
//     }
//     
//     arma::mat snapshot_param(p, K), param_y_old(p, K), grad_full(p, K), grad_at_x(p, K),
//     grad_at_snapshot(p, K), grad_stoch(p, K), param_z_unprox(p, K);
//     
//     bool converged = false;
//     int convergence_iter = -1;
//     Rcpp::List param_history;
//     double diff;
//     int counter_outer = 0;
//     
//     while (true) {
//         Rcpp::checkUserInterrupt();
//         param_y_old = param_y;
//         counter_outer += 1;
//         
//         if (save_history) {
//             param_history.push_back(Rcpp::wrap(param_y));
//         }
//         
//         snapshot_param = param_y;
//         grad_full.zeros();
//         grad_full = compute_full_grad_vectorized(X, offset, snapshot_param, Y);
//         
//         for (int k = 0; k < niter_inner; ++k) {
//             double gamma_k = (static_cast<double>(k) + v_factor + 4.0) / (2.0 * c_factor * L);
//             double tau_k = 1.0 / (c_factor * L * gamma_k);
//             if (tau_k > 1.0) tau_k = 1.0;
//             
//             // 1. Extrapolation step
//             param_x = tau_k * param_z + (1.0 - tau_k) * param_y;
//             
//             // 2. Variance-reduced gradient calculation
//             int index = arma::randi(arma::distr_param(0, n - 1));
//             grad_at_x = grad_multinom_loss(X.row(index), Y(index), K, offset(index), param_x, p);
//             grad_at_snapshot = grad_multinom_loss(X.row(index), Y(index), K, offset(index), snapshot_param, p);
//             grad_stoch = grad_at_x - grad_at_snapshot + grad_full;
//             
//             // 3. Proximal gradient step on z-iterate
//             param_z_unprox = param_z - gamma_k * grad_stoch;
//             apply_proximal_step(param_z, param_z_unprox, gamma_k, reg_p, K, p, transpose, penalty, regul,
//                                 grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
//             
//             // 4. Momentum update on y-iterate
//             param_y = tau_k * param_z + (1.0 - tau_k) * param_y;
//         }
//         
//         diff = arma::norm(param_y - param_y_old, "fro") / (arma::norm(param_y_old, "fro") + 1e-10);
//         
//         if (verbose) {
//             Rcpp::Rcout << "Iteration " << counter_outer << " | Relative Change: " << diff << "\n";
//         }
//         
//         if (diff < tolerance) {
//             converged = true;
//             convergence_iter = counter_outer;
//             break;
//         }
//         if (counter_outer >= maxit) {
//             convergence_iter = maxit;
//             break;
//         }
//     }
//     
//     arma::sp_mat param_sp(param_y);
//     
//     Rcpp::List result = Rcpp::List::create(
//         Rcpp::Named("Estimates") = param_y,
//         Rcpp::Named("Sparse Estimates") = param_sp,
//         Rcpp::Named("Converged") = converged,
//         Rcpp::Named("Convergence Iteration") = convergence_iter
//     );
//     
//     if (save_history) {
//         param_history.push_back(Rcpp::wrap(param_y));
//         result["History"] = param_history;
//     }
//     
//     return result;
// }


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
        double tolerance = 1e-6,
        int maxit = 500,
        int ncores = 1,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {

    const int p = X.n_cols;
    const int n = X.n_rows;

    // learning rate based on Lipschitz constant
    if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant..." << std::endl;
    double L = get_lambda_max(X) / 4.0; 
    L = std::max(L, 1e-4); 
    double learning_rate = 1.0 / L;
    if (verbose) Rcpp::Rcout << "  Lipschitz constant L = " << L << ", Learning Rate = " << learning_rate << std::endl;

    // Initialize parameters 
    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }

    // store the gradient of each sample
    std::vector<arma::mat> grad_table(n);
    arma::mat grad_avg(p, K, arma::fill::zeros);

    for(int i = 0; i < n; ++i) {
        grad_table[i] = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);
        grad_avg += grad_table[i];
    }
    grad_avg /= static_cast<double>(n);

    arma::mat param_old(p, K);
    bool converged = false;
    int convergence_iter = -1;

    // Outer loop
    for (int iter = 1; iter <= maxit; ++iter) {
        Rcpp::checkUserInterrupt();
        param_old = param;

        // stochastic step
        arma::uvec indices = arma::randperm(n);
        for (arma::uword j = 0; j < n; ++j) {
            int i = indices(j);

            // old gradient for sample i
            arma::mat grad_old_i = grad_table[i];

            // new gradient for sample i 
            arma::mat grad_new_i = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);

            // average gradient
            grad_avg += (grad_new_i - grad_old_i) / static_cast<double>(n);

            // Update gradient table
            grad_table[i] = grad_new_i;

            // proximal gradient descent step
            arma::mat param_unprox = param - learning_rate * grad_avg;
            apply_proximal_step(param, param_unprox, learning_rate, reg_p, K, p, transpose, penalty, regul,
                                grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
        }

        // Convergence Check 
        double param_norm = std::max(arma::norm(param_old, "fro"), 1.0);
        double diff = arma::norm(param - param_old, "fro") / param_norm;

        if (verbose && (iter % 10 == 0 || iter == 1)) {
            Rcpp::Rcout << "Iter " << iter << " | Relative Change: " << diff << std::endl;
        }

        if (diff < tolerance) {
            converged = true;
            convergence_iter = iter;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
            break;
        }

        if (!param.is_finite()) {
            Rcpp::warning("Algorithm diverged with non-finite parameter values.");
            break;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
}


arma::mat compute_gradient(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offsets,
        const arma::mat& param,
        const arma::uvec& indices,
        bool is_minibatch = true 
) {
    const int K = param.n_cols;
    const arma::mat& X_subset = is_minibatch ? X.rows(indices) : X;
    const arma::vec& Y_subset = is_minibatch ? Y.elem(indices) : Y;
    arma::vec offsets_subset = is_minibatch ? offsets.elem(indices) : offsets;
    const int n_samples = X_subset.n_rows;
    
    // Vectorized Offset Clipping
    const double clip_threshold = 10.0;
    offsets_subset.clamp(-clip_threshold, clip_threshold);
    
    // Vectorized Softmax Calculation
    arma::mat scores = X_subset * param;
    arma::vec max_scores = arma::max(scores, 1);
    max_scores = arma::max(max_scores, offsets_subset);
    scores.each_col() -= max_scores;
    
    arma::mat probs = arma::exp(scores);
    arma::vec exp_baseline = arma::exp(offsets_subset - max_scores);
    arma::vec denom = arma::sum(probs, 1) + exp_baseline;
    
    // Safe Division
    probs.each_col() /= (denom + 1e-12);
    
    // One-Hot Encoding
    arma::mat Y_onehot(n_samples, K, arma::fill::zeros);
    for (int i = 0; i < n_samples; ++i) {
        if (Y_subset(i) > 0 && Y_subset(i) <= K) {
            Y_onehot(i, Y_subset(i) - 1) = 1.0;
        }
    }
    
    // Gradient Calculation
    arma::mat grad = X_subset.t() * (probs - Y_onehot);
    grad /= static_cast<double>(n_samples);
    grad.replace(arma::datum::nan, 0.0);
    
    return grad;
}

double estimate_lipschitz(const arma::mat& X) {
    int n = X.n_rows;
    int p = X.n_cols;
    
    if (p == 0) return 1e-8;
    
    double max_eig;
    if (p <= 150) {
        // Exact method for smaller matrices
        arma::mat XtX = X.t() * X;
        arma::vec eigvals = arma::eig_sym(XtX);
        max_eig = eigvals.max();
    } else {
        // Power iteration for larger matrices
        arma::vec v = arma::randn<arma::vec>(p);
        v /= arma::norm(v);
        for (int i = 0; i < 50; ++i) {
            arma::vec w = X.t() * (X * v);
            v = arma::normalise(w);
        }
        max_eig = arma::dot(v, X.t() * (X * v));
    }
    
    // L = lambda_max(X'X) / 4
    return std::max(max_eig / (4.0 * n), 1e-8);
}


// [[Rcpp::export]]
Rcpp::List accelerated_stochastic_optimizer(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        std::string estimator_type,
        double tolerance,
        int maxit,
        int niter_inner,
        int batch_size,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start,
        bool verbose,
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
        bool pos,
        int ncores
) {
    const int n = X.n_rows;
    const int p = X.n_cols;
    bool converged = false;
    int convergence_iter = -1;
    
    double L = estimate_lipschitz(X) + lam2;
    double mu = lam2; 
    
    double c_param = std::max(1.0, 3.0 * (double)n / batch_size);
    double rho = (double)batch_size / n;
    
    double gamma; 
    double tau; 
    
    if (mu > 0) { 
        gamma = std::min(1.0 / std::sqrt(mu * c_param * L), rho / (2.0 * mu));
        tau = mu * gamma;
    } else { 
        gamma = 1.0 / (2.0 * c_param * L);
        tau = 1.0 / (c_param * L * gamma);
    }
    
    if (verbose) {
        Rcpp::Rcout << "--- Optimizer Settings ---\n"
                    << "Estimator: " << estimator_type << "\n"
                    << "Lipschitz (L): " << L << ", μ: " << mu << "\n"
                    << "Step Size (γ): " << gamma << ", Momentum (τ): " << tau << std::endl;
    }
    
    arma::mat z_k(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        z_k = Rcpp::as<arma::mat>(param_start);
    }
    arma::mat y_k = z_k;
    arma::mat param_old = z_k;
    
    arma::mat snapshot_param;
    arma::mat grad_full;
    std::vector<arma::mat> saga_grad_table;
    arma::mat saga_grad_avg;
    
    if (estimator_type == "SAGA") {
        saga_grad_table.resize(n);
        saga_grad_avg.zeros(p, K);
        
        for (int i = 0; i < n; ++i) {
            saga_grad_table[i] = compute_gradient(X, Y, offset, z_k, {(arma::uword)i}, true);
            saga_grad_avg += saga_grad_table[i];
        }
        saga_grad_avg /= static_cast<double>(n);
    }
    
    // Outer loop
    for (int outer = 1; outer <= maxit; ++outer) {
        Rcpp::checkUserInterrupt();
        
        arma::uvec all_indices = arma::regspace<arma::uvec>(0, n - 1);
        
        if (estimator_type == "SVRG") {
            snapshot_param = y_k; 
            grad_full = compute_gradient(X, Y, offset, snapshot_param, all_indices, false);
        }
        
        int loop_limit = (estimator_type == "SVRG") ? niter_inner : n / batch_size;
        
        for (int inner = 0; inner < loop_limit; ++inner) {
            arma::mat x_k1 = tau * z_k + (1.0 - tau) * y_k;
            arma::mat grad_vr;
            arma::uvec batch_idx = arma::randperm(n, batch_size);
            
            if (estimator_type == "SVRG") {
                arma::mat grad_current = compute_gradient(X, Y, offset, x_k1, batch_idx, true);
                arma::mat grad_snapshot = compute_gradient(X, Y, offset, snapshot_param, batch_idx, true);
                grad_vr = grad_current - grad_snapshot + grad_full;
            } else if (estimator_type == "SAGA") {
                arma::mat grad_new_i(p, K);
                arma::mat grad_old_i(p, K, arma::fill::zeros);
                
                grad_new_i = compute_gradient(X, Y, offset, x_k1, batch_idx, true);
                for(arma::uword i : batch_idx) { grad_old_i += saga_grad_table[i]; }
                if (batch_size > 1) grad_old_i /= batch_size;
                
                grad_vr = grad_new_i - grad_old_i + saga_grad_avg;
                
                saga_grad_avg += (grad_new_i - grad_old_i) * (batch_size / static_cast<double>(n));
                for(arma::uword i : batch_idx) { saga_grad_table[i] = grad_new_i; }
            }
            
            arma::mat z_k_unprox = z_k - gamma * grad_vr;
            apply_proximal_step(
                z_k, z_k_unprox, gamma, reg_p, K, p, transpose, penalty, regul,
                grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, 
                own_var, N_own_var
            );
            
            y_k = tau * z_k + (1.0 - tau) * y_k;
        }
        
        // Convergence Check
        double change = arma::norm(y_k - param_old, "fro") / (1.0 + arma::norm(param_old, "fro"));
        if (verbose && (outer % 10 == 0 || outer == 1)) {
            Rcpp::Rcout << "Iter " << outer << " | Change: " << change << std::endl;
        }
        if (change < tolerance) {
            converged = true;
            convergence_iter = outer;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << outer << "." << std::endl;
            break;
        }
        if (!y_k.is_finite()) {
            Rcpp::warning("Algorithm diverged with non-finite parameter values.");
            break;
        }
        param_old = y_k;
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Estimates") = y_k,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
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
        double c_factor, // Deprecated
        double v_factor, // Deprecated
        double tolerance = 1e-3,
        int niter_inner = 100,
        int maxit = 500,
        int ncores = 1,
        bool save_history = false, // Deprecated
        bool verbose = false,
        bool pos = false,
        int batch_size = 64,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue
) {
    const int n = X.n_rows;
    const int p = X.n_cols;
    bool converged = false;
    int convergence_iter = -1;
    
    double L = estimate_lipschitz(X) + lam2;
    double mu = lam2; 
    
    double c_param = std::max(1.0, 3.0 * (double)n / batch_size);
    double rho = (double)batch_size / n;
    
    double gamma; // Step size
    double tau;   // Momentum parameter
    
    if (mu > 0) { // Strongly convex case
        gamma = std::min(1.0 / std::sqrt(mu * c_param * L), rho / (2.0 * mu));
        tau = mu * gamma;
    } else { // Not strongly convex - use a small but stable constant step size
        gamma = 1.0 / (3.0 * c_param * L);
        tau = std::min(0.5, 1.0 / (c_param * L * gamma));
    }
    
    if (verbose) {
        Rcpp::Rcout << "--- Final Optimizer Settings ---\n"
                    << "Lipschitz (L): " << L << ", μ: " << mu << "\n"
                    << "Step Size (γ): " << gamma << ", Momentum (τ): " << tau << std::endl;
    }
    
    arma::mat z_k(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) z_k = Rcpp::as<arma::mat>(param_start);
    arma::mat y_k = z_k;
    arma::mat param_old = z_k;
    
    for (int outer = 1; outer <= maxit; ++outer) {
        Rcpp::checkUserInterrupt();
        
        arma::mat snapshot_param = y_k;
        arma::mat grad_full = compute_gradient(X, Y, offset, snapshot_param, arma::regspace<arma::uvec>(0, n - 1), false);
        
        if (!grad_full.is_finite()) {
            if (verbose) Rcpp::warning("Full gradient is non-finite. Stopping.");
            break;
        }
        
        for (int inner = 0; inner < niter_inner; ++inner) {
            arma::mat x_k1 = tau * z_k + (1.0 - tau) * y_k;
            
            arma::uvec batch_idx = arma::randperm(n, batch_size);
            arma::mat grad_current = compute_gradient(X, Y, offset, x_k1, batch_idx, true);
            arma::mat grad_snapshot = compute_gradient(X, Y, offset, snapshot_param, batch_idx, true);
            arma::mat grad_vr = grad_current - grad_snapshot + grad_full;
            
            arma::mat z_k_unprox = z_k - gamma * grad_vr;
            apply_proximal_step(
                z_k, z_k_unprox, gamma, reg_p, K, p, transpose, penalty, regul,
                grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG,
                own_var, N_own_var
            );
            
            y_k = tau * z_k + (1.0 - tau) * y_k;
        }
        
        double change = arma::norm(y_k - param_old, "fro") / (1.0 + arma::norm(param_old, "fro"));
        if (verbose && (outer % 10 == 0 || outer == 1)) {
            Rcpp::Rcout << "Iter " << outer << " | Change: " << change << std::endl;
        }
        if (change < tolerance) {
            converged = true;
            convergence_iter = outer;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << outer << "." << std::endl;
            break;
        }
        if (!y_k.is_finite()) {
            Rcpp::warning("Algorithm diverged with non-finite parameter values.");
            break;
        }
        param_old = y_k;
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Estimates") = y_k,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
}

// // [[Rcpp::export]]
// Rcpp::List MultinomLogisticSAGA(
//         const arma::mat& X,
//         const arma::vec& Y,
//         const arma::vec& offset,
//         int K,
//         int reg_p,
//         int penalty,
//         std::string regul,
//         bool transpose,
//         Rcpp::IntegerVector grp_id,
//         Rcpp::NumericVector etaG,
//         const arma::mat& grp,
//         const arma::mat& grpV,
//         Rcpp::IntegerVector own_var,
//         Rcpp::IntegerVector N_own_var,
//         double lam1,
//         double lam2,
//         double lam3,
//         double tolerance = 1e-6,
//         int maxit = 500,
//         int ncores = 1,
//         bool verbose = false,
//         bool pos = false,
//         Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue
// ) {
//     const int p = X.n_cols;
//     const int n = X.n_rows;
//     
//     // --- 1. Set up learning rate (Identical to your stable version) ---
//     if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant..." << std::endl;
//     double L = get_lambda_max(X) / 4.0;
//     L = std::max(L, 1e-4);
//     double learning_rate = 1.0 / (L + lam2); // Add L2 effect
//     if (verbose) Rcpp::Rcout << "  Lipschitz constant L_sum = " << L << ", Learning Rate = " << learning_rate << std::endl;
//     
//     // --- 2. Initialize Parameters and SAGA variables ---
//     arma::mat param(p, K, arma::fill::zeros);
//     if (param_start.isNotNull()) {
//         param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
//     }
//     
//     std::vector<arma::mat> grad_table(n);
//     arma::mat grad_avg(p, K, arma::fill::zeros);
//     
//     if (verbose) Rcpp::Rcout << "Initializing SAGA gradient table..." << std::endl;
//     for(int i = 0; i < n; ++i) {
//         // PERFORMANCE FIX: Call the fast, vectorized gradient function
//         grad_table[i] = compute_gradient(X, Y, offset, param, {(arma::uword)i});
//         grad_avg += grad_table[i];
//     }
//     grad_avg /= static_cast<double>(n);
//     
//     arma::mat param_old(p, K);
//     bool converged = false;
//     int convergence_iter = -1;
//     
//     // --- 3. Main SAGA Optimization Loop (Identical logic, but faster calls) ---
//     if (verbose) Rcpp::Rcout << "Starting SAGA optimization..." << std::endl;
//     for (int iter = 1; iter <= maxit; ++iter) {
//         Rcpp::checkUserInterrupt();
//         param_old = param;
//         
//         arma::uvec indices = arma::randperm(n);
//         for (arma::uword j = 0; j < n; ++j) {
//             int i = indices(j);
//             arma::mat grad_old_i = grad_table[i];
//             
//             // PERFORMANCE FIX: Call the fast, vectorized gradient function
//             arma::mat grad_new_i = compute_gradient(X, Y, offset, param, {(arma::uword)i});
//             
//             grad_avg += (grad_new_i - grad_old_i) / static_cast<double>(n);
//             grad_table[i] = grad_new_i;
//             
//             arma::mat param_unprox = param - learning_rate * grad_avg;
//             apply_proximal_step(param, param_unprox, learning_rate, reg_p, K, p, transpose, penalty, regul,
//                                 grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
//         }
//         
//         // --- 4. Convergence Check (Identical) ---
//         double diff = arma::norm(param - param_old, "fro") / (1.0 + arma::norm(param_old, "fro"));
//         if (verbose && (iter % 10 == 0 || iter == 1)) {
//             Rcpp::Rcout << "Iter " << iter << " | Relative Change: " << diff << std::endl;
//         }
//         
//         if (diff < tolerance) {
//             converged = true;
//             convergence_iter = iter;
//             break;
//         }
//         if (!param.is_finite()) {
//             Rcpp::warning("Algorithm diverged with non-finite parameter values.");
//             break;
//         }
//     }
//     
//     return Rcpp::List::create(
//         Rcpp::Named("Estimates") = param,
//         Rcpp::Named("Converged") = converged,
//         Rcpp::Named("Convergence Iteration") = convergence_iter
//     );
// }