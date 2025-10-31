
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
        arma::rowvec& x,
        int& y,
        int& K,
        double& offset,
        arma::mat& param,
        int& p) {
    arma::mat grad(p, K);
    arma::vec pi(K);
    
    for (int i = 0; i < K; i++) {
        pi(i) = exp(arma::dot(x, param.col(i)) + offset);
    }
    pi = pi/(arma::sum(pi) + 1.0);
    
    if (y == 0) {
        for (int k = 0; k < K; k++) {
            grad.col(k) = pi(k) * arma::vectorise(x);
        }
    } else {
        for (int k = 0; k < K; k++) {
            if (k == y - 1) {
                grad.col(k) = (pi(k) - 1) * arma::vectorise(x);
            } else {
                grad.col(k) = pi(k) * arma::vectorise(x);
            }
        }
    }
    return grad;
}



arma::mat grad_multinom_single(
        const arma::rowvec& x,
        int y,
        const arma::vec& offset_val,
        const arma::mat& param,
        int K) {
    
    double offset = offset_val(0);
    
    // Soft-clip the offset for numerical stability
    if (std::abs(offset) > 50.0) {
        offset = 50.0 * std::tanh(offset / 50.0);
    }
    
    // Compute linear scores for K explicit classes
    arma::vec linear_scores = param.t() * x.t() + offset; 
    
    // Baseline class has score of 0
    double baseline_score = 0.0;
    
    // Stabilized softmax computation including baseline
    double max_score = std::max(linear_scores.max(), baseline_score);
    arma::vec exp_scores = arma::exp(linear_scores - max_score);
    double denom = arma::accu(exp_scores) + std::exp(baseline_score - max_score) + 1e-12;
    
    arma::mat grad_out(param.n_rows, K, arma::fill::zeros);
    if (denom < 1e-10 || !std::isfinite(denom)) {
        return grad_out;
    }
    
    arma::vec pi = exp_scores / denom; 
    
    // Gradient calculation
    grad_out = x.t() * pi.t(); 
    
    if (y > 0 && y <= K) {
        grad_out.col(y - 1) -= x.t();
    }
    return grad_out;
}

arma::mat grad_multinom_batch(
        const arma::mat& Xb,
        const arma::vec& Yb,
        const arma::vec& offset_b,
        const arma::mat& param,
        int K) {
    
    const int b = Xb.n_rows; // Batch size
    arma::vec offset_clipped = offset_b;
    offset_clipped.transform([](double val) {
        if (std::abs(val) > 50.0) return 50.0 * std::tanh(val / 50.0);
        return val;
    });
    
    // linear predictor (b x K)
    arma::mat Eta = Xb * param; // (b x K)
    
    // add offset to explicit class scores
    Eta.each_col() += offset_clipped;
    
    // stable softmax including baseline score 0
    arma::vec max_scores = arma::max(Eta, 1);
    max_scores = arma::max(max_scores, arma::zeros<arma::vec>(b)); // compare with baseline 0
    Eta.each_col() -= max_scores; // broadcast subtraction
    arma::mat Exp_Eta = arma::exp(Eta);
    arma::vec Exp_Baseline = arma::exp(-max_scores);
    arma::vec Denom = arma::sum(Exp_Eta, 1) + Exp_Baseline + 1e-12; // (b x 1)
    arma::mat Pi = Exp_Eta.each_col() / Denom; // (b x K)
    
    // one-hot Y
    arma::mat Y_one_hot(b, K, arma::fill::zeros);
    for (int i = 0; i < b; ++i) {
        int yi = static_cast<int>(Yb(i));
        if (yi > 0 && yi <= K) Y_one_hot(i, yi - 1) = 1.0;
    }
    
    arma::mat Residuals = Pi - Y_one_hot; // (b x K)
    arma::mat grad_batch = Xb.t() * Residuals; // (p x K)
    return grad_batch;
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
        double tolerance = 1e-4,
        double lr_adj = 1.0,
        double max_lr = 1.0,
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

    double learning_rate = std::min(lr_adj / L, max_lr);
    
    if (verbose) Rcpp::Rcout << "  Lipschitz constant L = " << L << ", Learning Rate = " << learning_rate << std::endl;

    // Initialize parameters
    arma::mat param(p, K, arma::fill::zeros);
    
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }

    // store the gradient of each sample
    arma::cube grad_table(p, K, n); // A single (p x K x n) memory block
    arma::mat grad_avg(p, K, arma::fill::zeros);

    for(int i = 0; i < n; ++i) {
        grad_table.slice(i) = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);
        grad_avg += grad_table.slice(i);
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
            
            arma::mat grad_old_i = grad_table.slice(i);
            arma::mat grad_new_i = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);
            
            arma::mat saga_update_direction = grad_new_i - grad_old_i + grad_avg;
            
            grad_avg += (grad_new_i - grad_old_i) / static_cast<double>(n);
            
            grad_table.slice(i) = grad_new_i;
            
            arma::mat param_unprox = param - learning_rate * saga_update_direction;
            
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
        const arma::mat& X, const arma::vec& Y, const arma::vec& offsets,
        const arma::mat& param, const arma::uvec& indices
) {
    const int K = param.n_cols;
    const arma::mat& X_subset = X.rows(indices);
    const arma::vec& Y_subset = Y.elem(indices);
    arma::vec offsets_subset = offsets.elem(indices);
    const int n_samples = X_subset.n_rows;
    
    offsets_subset.for_each([](arma::vec::elem_type& val) {
        if (std::abs(val) > 20.0) {
            val = 20.0 * std::tanh(val / 20.0);
        }
    });
    
    arma::mat scores = X_subset * param;
    scores.each_col() += offsets_subset;
    double baseline_score = 0.0;
    
    arma::vec max_scores = arma::max(scores, 1);
    max_scores.elem( arma::find(max_scores < baseline_score) ).fill(baseline_score);
    scores.each_col() -= max_scores;
    
    arma::mat probs = arma::exp(scores);
    arma::vec exp_baseline = arma::exp(arma::zeros<arma::vec>(n_samples) - max_scores);
    arma::vec denom = arma::sum(probs, 1) + exp_baseline;
    
    probs.each_col() /= (denom + 1e-12);
    
    arma::mat Y_onehot(n_samples, K, arma::fill::zeros);
    for (int i = 0; i < n_samples; ++i) {
        if (Y_subset(i) > 0 && Y_subset(i) <= K)
            Y_onehot(i, Y_subset(i) - 1) = 1.0;
    }
    
    arma::mat grad = X_subset.t() * (probs - Y_onehot);
    grad /= static_cast<double>(n_samples);
    grad.replace(arma::datum::nan, 0.0);
    return grad;
}

// Lipschitz constant
double estimate_lipschitz(const arma::mat& X, int max_iters = 10) {
    int p = X.n_cols;
    arma::vec v = arma::randn<arma::vec>(p);
    v /= arma::norm(v);
    
    for (int i = 0; i < max_iters; ++i) {
        v = X.t() * (X * v);
        double norm_v = arma::norm(v);
        if (norm_v < 1e-12) break;
        v /= norm_v;
    }
    
    return arma::norm(X.t() * (X * v)) / 4.0;
}


// // [[Rcpp::export]]
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
//         double c_factor, // Deprecated
//         double v_factor, // Deprecated
//         double tolerance = 1e-3,
//         int niter_inner = 100,
//         int maxit = 500,
//         int ncores = 1,
//         bool save_history = false, // Deprecated
//         bool verbose = false,
//         bool pos = false,
//         int batch_size = 64,
//         Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue
// ) {
//     const int n = X.n_rows;
//     const int p = X.n_cols;
//     bool converged = false;
//     int convergence_iter = -1;
//     
//     double L = estimate_lipschitz(X) + lam2;
//     double mu = lam2; 
//     
//     double c_param = std::max(1.0, 3.0 * (double)n / batch_size);
//     double rho = (double)batch_size / n;
//     
//     double gamma; // Step size
//     double tau;   // Momentum parameter
//     
//     if (mu > 0) { // Strongly convex case
//         gamma = std::min(1.0 / std::sqrt(mu * c_param * L), rho / (2.0 * mu));
//         tau = mu * gamma;
//     } else { // Not strongly convex - use a small but stable constant step size
//         gamma = 1.0 / (3.0 * c_param * L);
//         tau = std::min(0.5, 1.0 / (c_param * L * gamma));
//     }
//     
//     if (verbose) {
//         Rcpp::Rcout << "--- Final Optimizer Settings ---\n"
//                     << "Lipschitz (L): " << L << ", μ: " << mu << "\n"
//                     << "Step Size (γ): " << gamma << ", Momentum (τ): " << tau << std::endl;
//     }
//     
//     arma::mat z_k(p, K, arma::fill::zeros);
//     if (param_start.isNotNull()) z_k = Rcpp::as<arma::mat>(param_start);
//     arma::mat y_k = z_k;
//     arma::mat param_old = z_k;
//     
//     for (int outer = 1; outer <= maxit; ++outer) {
//         Rcpp::checkUserInterrupt();
//         
//         arma::mat snapshot_param = y_k;
//         arma::mat grad_full = compute_gradient(X, Y, offset, snapshot_param, arma::regspace<arma::uvec>(0, n - 1), false);
//         
//         if (!grad_full.is_finite()) {
//             if (verbose) Rcpp::warning("Full gradient is non-finite. Stopping.");
//             break;
//         }
//         
//         for (int inner = 0; inner < niter_inner; ++inner) {
//             arma::mat x_k1 = tau * z_k + (1.0 - tau) * y_k;
//             
//             arma::uvec batch_idx = arma::randperm(n, batch_size);
//             arma::mat grad_current = compute_gradient(X, Y, offset, x_k1, batch_idx, true);
//             arma::mat grad_snapshot = compute_gradient(X, Y, offset, snapshot_param, batch_idx, true);
//             arma::mat grad_vr = grad_current - grad_snapshot + grad_full;
//             
//             arma::mat z_k_unprox = z_k - gamma * grad_vr;
//             apply_proximal_step(
//                 z_k, z_k_unprox, gamma, reg_p, K, p, transpose, penalty, regul,
//                 grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG,
//                 own_var, N_own_var
//             );
//             
//             y_k = tau * z_k + (1.0 - tau) * y_k;
//         }
//         
//         double change = arma::norm(y_k - param_old, "fro") / (1.0 + arma::norm(param_old, "fro"));
//         if (verbose && (outer % 10 == 0 || outer == 1)) {
//             Rcpp::Rcout << "Iter " << outer << " | Change: " << change << std::endl;
//         }
//         if (change < tolerance) {
//             converged = true;
//             convergence_iter = outer;
//             if (verbose) Rcpp::Rcout << "Converged at iteration " << outer << "." << std::endl;
//             break;
//         }
//         if (!y_k.is_finite()) {
//             Rcpp::warning("Algorithm diverged with non-finite parameter values.");
//             break;
//         }
//         param_old = y_k;
//     }
//     
//     return Rcpp::List::create(
//         Rcpp::Named("Estimates") = y_k,
//         Rcpp::Named("Converged") = converged,
//         Rcpp::Named("Convergence Iteration") = convergence_iter
//     );
// }

// // [[Rcpp::export]]
// Rcpp::List MultinomLogisticSAGA(
//         const arma::mat& X, const arma::vec& Y, const arma::vec& offset, int K,
//         int reg_p, int penalty, std::string regul, bool transpose,
//         Rcpp::IntegerVector grp_id, Rcpp::NumericVector etaG,
//         const arma::mat& grp, const arma::mat& grpV,
//         Rcpp::IntegerVector own_var, Rcpp::IntegerVector N_own_var,
//         double lam1, double lam2, double lam3, double tolerance = 1e-6,
//         int maxit = 500, int ncores = 1, bool verbose = false, bool pos = false,
//         int batch_size = 64, Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue
// ) {
//     const int p = X.n_cols;
//     const int n = X.n_rows;
//     
//     // --- 1. Correctly Scaled Learning Rate ---
//     double L_avg = estimate_lipschitz(X) + lam2;
//     // A standard, theoretically sound learning rate for SAGA.
//     double learning_rate = 1.0 / (3.0*L_avg);
//     
//     if (verbose) {
//         Rcpp::Rcout << "--- Definitive SAGA Settings ---\n"
//                     << "L_avg: " << L_avg << ", Correctly Scaled Learning Rate: " << learning_rate << std::endl;
//     }
//     
//     // --- 2. Initialization ---
//     arma::mat param(p, K, arma::fill::zeros);
//     if (param_start.isNotNull()) param = Rcpp::as<arma::mat>(param_start);
//     
//     std::vector<arma::mat> grad_table(n);
//     arma::mat grad_avg(p, K, arma::fill::zeros);
//     
//     if (verbose) Rcpp::Rcout << "Initializing gradient table..." << std::endl;
//     for(int i = 0; i < n; ++i) {
//         grad_table[i] = compute_gradient(X, Y, offset, param, {(arma::uword)i});
//         grad_avg += grad_table[i];
//     }
//     grad_avg /= static_cast<double>(n);
//     
//     arma::mat param_old;
//     bool converged = false;
//     int convergence_iter = -1;
//     
//     // --- 3. Main SAGA Loop (Fast, Batched, and Correct) ---
//     for (int iter = 1; iter <= maxit; ++iter) {
//         Rcpp::checkUserInterrupt();
//         param_old = param;
//         
//         arma::uvec indices = arma::randperm(n);
//         for (arma::uword i = 0; i < n / batch_size; ++i) {
//             arma::uvec batch_idx = indices.subvec(i * batch_size, (i + 1) * batch_size - 1);
//             
//             arma::mat grad_new_batch = compute_gradient(X, Y, offset, param, batch_idx);
//             
//             arma::mat grad_old_batch(p, K, arma::fill::zeros);
//             for (arma::uword idx : batch_idx) {
//                 grad_old_batch += grad_table[idx];
//             }
//             grad_old_batch /= static_cast<double>(batch_size);
//             
//             arma::mat saga_grad = grad_new_batch - grad_old_batch + grad_avg;
//             
//             arma::mat param_unprox = param - learning_rate * saga_grad;
//             apply_proximal_step(param, param_unprox, learning_rate, reg_p, K, p, transpose, penalty, regul,
//                                 grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
//             
//             for (arma::uword idx : batch_idx) {
//                 grad_avg += (grad_new_batch - grad_table[idx]) / static_cast<double>(n);
//                 grad_table[idx] = grad_new_batch;
//             }
//         }
//         
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
//     }
//     
//     return Rcpp::List::create(
//         Rcpp::Named("Estimates") = param,
//         Rcpp::Named("Converged") = converged,
//         Rcpp::Named("Convergence Iteration") = convergence_iter
//     );
// }

double estimate_multinomial_lipschitz(const arma::mat& X, int K, int max_iters = 15) {
    int p = X.n_cols;
    int n = X.n_rows;
    
    // For multinomial, we need to consider the structure
    // A simple heuristic: use spectral norm of X'X / 4
    arma::vec v = arma::randn<arma::vec>(p);
    v /= arma::norm(v);
    
    for (int i = 0; i < max_iters; ++i) {
        arma::vec v_old = v;
        v = X.t() * (X * v);
        double norm_v = arma::norm(v);
        if (norm_v < 1e-12) break;
        v /= norm_v;
        
        // Early convergence check
        if (arma::norm(v - v_old) < 1e-8) break;
    }
    
    double spectral_norm_sq = arma::norm(X.t() * (X * v));
    
    // For multinomial logistic regression, Lipschitz constant <= ||X||^2 / 4
    // Use a conservative estimate
    return spectral_norm_sq / 4.0;
}

// Adaptive learning rate finder using line search
double find_adaptive_learning_rate(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        const arma::mat& param,
        const arma::mat& gradient,
        int K,
        double initial_lr = 10.0) {
    
    int n = X.n_rows;
    double lr = initial_lr;
    
    // Try a few learning rates and pick the one that gives best objective improvement
    double best_obj = arma::datum::inf;
    double best_lr = initial_lr;
    
    // Test a range of learning rates
    for (int attempt = 0; attempt < 5; ++attempt) {
        double test_lr = lr * std::pow(0.5, attempt);
        arma::mat test_param = param - test_lr * gradient;
        
        // Compute objective for a subset of data (for speed)
        int subset_size = std::min(1000, n);
        double obj = 0.0;
        int count = 0;
        
        for (int i = 0; i < subset_size && count < 100; i += n/100) {
            if (i >= n) break;
            
            arma::vec offset_i(1);
            offset_i(0) = offset(i);
            arma::rowvec x_i = X.row(i);
            int y_i = Y(i);
            
            arma::vec linear_scores = test_param.t() * x_i.t() + offset_i(0);
            double max_score = std::max(linear_scores.max(), 0.0);
            arma::vec exp_scores = arma::exp(linear_scores - max_score);
            double denom = arma::accu(exp_scores) + std::exp(-max_score) + 1e-12;
            
            if (y_i == 0) {
                obj -= std::log(std::exp(-max_score) / denom);
            } else {
                obj -= std::log(exp_scores(y_i - 1) / denom);
            }
            count++;
        }
        
        obj /= count;
        
        if (obj < best_obj && std::isfinite(obj)) {
            best_obj = obj;
            best_lr = test_lr;
        }
    }
    
    return best_lr;
}

// [[Rcpp::export]]
Rcpp::List AutoTunedMultinomSAGA(
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
        double tolerance = 1e-4,
        int maxit = 500,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    
    // Initialize parameters
    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }
    
    // Step 1: Estimate Lipschitz constant and set initial learning rate
    if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant..." << std::endl;
    double L = estimate_multinomial_lipschitz(X, K);
    double learning_rate = 1.0 / (2.0 * L); // Conservative initial estimate
    
    if (verbose) Rcpp::Rcout << "Initial learning rate: " << learning_rate << std::endl;
    
    // Gradient table
    arma::cube grad_table(p, K, n, arma::fill::zeros);
    arma::mat grad_avg(p, K, arma::fill::zeros);
    
    // Initialize gradient table
    if (verbose) Rcpp::Rcout << "Initializing gradient table..." << std::endl;
    for(int i = 0; i < n; ++i) {
        arma::vec offset_i(1);
        offset_i(0) = offset(i);
        grad_table.slice(i) = grad_multinom_single(X.row(i), Y(i), offset_i, param, K);
        grad_avg += grad_table.slice(i);
    }
    grad_avg /= n;
    
    arma::mat param_old(p, K);
    bool converged = false;
    int convergence_iter = -1;
    
    // Track progress for adaptive learning rate
    double best_obj = arma::datum::inf;
    int no_improvement_count = 0;
    const int max_no_improvement = 100;
    
    if (verbose) Rcpp::Rcout << "Starting optimization..." << std::endl;
    
    for (int iter = 1; iter <= maxit; ++iter) {
        Rcpp::checkUserInterrupt();
        param_old = param;
        
        // Adaptive learning rate adjustment every 20 iterations
        if (iter % 10 == 1 && iter > 1) {
            // Compute current full gradient for learning rate tuning
            arma::mat full_grad(p, K, arma::fill::zeros);
            for(int i = 0; i < std::min(500, n); i += n/500) {
                if (i >= n) break;
                arma::vec offset_i(1);
                offset_i(0) = offset(i);
                full_grad += grad_multinom_single(X.row(i), Y(i), offset_i, param, K);
            }
            full_grad /= std::min(500, n);
            
            double new_lr = find_adaptive_learning_rate(X, Y, offset, param, full_grad, K, learning_rate);
            
            if (std::abs(new_lr - learning_rate) / learning_rate > 0.1) {
                if (verbose) Rcpp::Rcout << "  Adjusting learning rate: " << learning_rate << " -> " << new_lr << std::endl;
                learning_rate = new_lr;
            }
        }
        
        // Process samples
        arma::uvec indices = arma::randperm(n);
        double iter_max_change = 0.0;
        
        for (int j = 0; j < n; ++j) {
            int i = indices(j);
            arma::vec offset_i(1);
            offset_i(0) = offset(i);
            
            // Compute new gradient
            arma::mat grad_new = grad_multinom_single(X.row(i), Y(i), offset_i, param, K);
            arma::mat grad_old = grad_table.slice(i);
            
            // SAGA update
            arma::mat saga_direction = grad_new - grad_old + grad_avg;
            
            // Update gradient average and table
            grad_avg += (grad_new - grad_old) / n;
            grad_table.slice(i) = grad_new;
            
            // Proximal gradient step
            arma::mat param_unprox = param - learning_rate * saga_direction;
            
            apply_proximal_step(param, param_unprox, learning_rate, reg_p, K, p, 
                                transpose, penalty, regul, grp_id, 1, lam1, lam2, 
                                lam3, pos, grp, grpV, etaG, own_var, N_own_var);
            
            // Track maximum parameter change in this iteration
            double max_change = arma::norm(param - param_old, "fro");
            if (max_change > iter_max_change) {
                iter_max_change = max_change;
            }
        }
        
        // Convergence check
        double diff = arma::norm(param - param_old, "fro") / std::max(arma::norm(param_old, "fro"), 1.0);
        
        // Monitor objective (approximate)
        double current_obj = 0.0;
        int obj_count = 0;
        for (int i = 0; i < std::min(100, n); i += n/100) {
            if (i >= n) break;
            arma::vec offset_i(1);
            offset_i(0) = offset(i);
            arma::rowvec x_i = X.row(i);
            int y_i = Y(i);
            
            arma::vec linear_scores = param.t() * x_i.t() + offset_i(0);
            double max_score = std::max(linear_scores.max(), 0.0);
            arma::vec exp_scores = arma::exp(linear_scores - max_score);
            double denom = arma::accu(exp_scores) + std::exp(-max_score) + 1e-12;
            
            if (y_i == 0) {
                current_obj -= std::log(std::exp(-max_score) / denom);
            } else {
                current_obj -= std::log(exp_scores(y_i - 1) / denom);
            }
            obj_count++;
        }
        current_obj /= obj_count;
        
        // Check for improvement
        if (current_obj < best_obj - 1e-6) {
            best_obj = current_obj;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
        }
        
        // Reduce learning rate if no improvement
        if (no_improvement_count >= 5) {
            learning_rate *= 0.8;
            no_improvement_count = 0;
            if (verbose) Rcpp::Rcout << "  Reducing learning rate to: " << learning_rate << std::endl;
        }
        
        if (verbose && iter % 50 == 0) {
            Rcpp::Rcout << "Iter " << iter << " | Change: " << diff 
                        << " | LR: " << learning_rate 
                        << " | Obj: " << current_obj << std::endl;
        }
        
        if (diff < tolerance) {
            converged = true;
            convergence_iter = iter;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
            break;
        }
        
        if (iter > 50 && diff > 10.0) {
            // Divergence detected - reduce learning rate and restart from previous
            learning_rate *= 0.5;
            param = param_old;
            if (verbose) Rcpp::Rcout << "Divergence detected, reducing LR to: " << learning_rate << std::endl;
        }
        
        if (!param.is_finite()) {
            Rcpp::warning("Non-finite parameter values detected.");
            param = param_old;
            break;
        }
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter,
        Rcpp::Named("FinalLearningRate") = learning_rate
    );
}

double prox_scad_scalar(double val, double lambda, double a = 3.7) {
    if (lambda <= 0.0) return val;
    
    double abs_val = std::abs(val);
    if (a <= 2.0) a = 3.7;
    
    if (abs_val <= lambda) {
        return 0.0;
    } else if (abs_val <= 2.0 * lambda) {
        return std::copysign(abs_val - lambda, val);
    } else if (abs_val <= a * lambda) {
        return ((a - 1.0) * val - std::copysign(a * lambda, val)) / (a - 2.0);
    } else {
        return val;
    }
}

double prox_elastic_net_scalar(double val, double lam1, double lam2) {
    if (lam1 <= 0.0 && lam2 <= 0.0) return val;
    
    double abs_val = std::abs(val);
    if (abs_val <= lam1) return 0.0;
    
    double st = std::copysign(abs_val - lam1, val);
    return st / (1.0 + lam2);
}

arma::vec prox_scad_vec(const arma::vec& vals, double lambda, double a = 3.7) {
    arma::vec result = vals;
    result.for_each([&](double& val) {
        val = prox_scad_scalar(val, lambda, a);
    });
    return result;
}

arma::vec prox_elastic_net_vec(const arma::vec& vals, double lam1, double lam2) {
    arma::vec result = vals;
    result.for_each([&](double& val) {
        val = prox_elastic_net_scalar(val, lam1, lam2);
    });
    return result;
}

// -----------------------------------------------------------------------------
// CCD HELPER FUNCTIONS
// -----------------------------------------------------------------------------

void update_probabilities_optimized(const arma::mat& Eta, 
                                    arma::mat& P, 
                                    const arma::vec& offset) {
    int n = Eta.n_rows;
    int K = Eta.n_cols;
    
    for (int i = 0; i < n; ++i) {
        // Find maximum for numerical stability
        double max_val = 0.0; // Baseline class has score 0
        for (int k = 0; k < K; ++k) {
            double current = Eta(i, k) + offset(i);
            if (current > max_val) max_val = current;
        }
        
        // Compute exponentials and sum
        double sum_exp = 0.0;
        
        // Baseline class (class 0)
        double baseline_exp = std::exp(-max_val);
        P(i, 0) = baseline_exp;
        sum_exp += baseline_exp;
        
        // Non-baseline classes
        for (int k = 0; k < K; ++k) {
            double exp_val = std::exp(Eta(i, k) + offset(i) - max_val);
            P(i, k + 1) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize probabilities
        if (sum_exp > 1e-12) {
            for (int k = 0; k <= K; ++k) {
                P(i, k) /= sum_exp;
            }
        } else {
            double uniform_prob = 1.0 / (K + 1);
            for (int k = 0; k <= K; ++k) {
                P(i, k) = uniform_prob;
            }
        }
    }
}

arma::vec compute_hessian_bounds(const arma::mat& X, int K) {
    int p = X.n_cols;
    arma::vec h_j(p);
    
    for (int j = 0; j < p; ++j) {
        double norm_sq = arma::accu(arma::square(X.col(j)));
        h_j(j) = 0.25 * norm_sq + 1e-12;
    }
    
    return h_j;
}

// -----------------------------------------------------------------------------
// OPTIMIZED CCD SOLVER
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List MultinomLogisticCCD(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int penalty = 1,
        double lam1 = 0.0,
        double lam2 = 0.0,
        double tolerance = 1e-5,
        int maxit = 500,
        double lr_adj = 1.0,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    
    // Initialize parameters
    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }
    
    // Very conservative learning rates
    arma::vec h_j = compute_hessian_bounds(X, K);
    arma::vec learning_rates = lr_adj / h_j;  // Even more conservative
    
    // Initialize matrices
    arma::mat Eta = X * param;
    arma::mat P(n, K + 1, arma::fill::zeros);
    update_probabilities_optimized(Eta, P, offset);
    
    arma::mat Y_one_hot(n, K + 1, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
        Y_one_hot(i, static_cast<int>(Y(i))) = 1.0;
    }
    
    arma::mat R = Y_one_hot.cols(1, K) - P.cols(1, K);
    
    bool converged = false;
    int convergence_iter = -1;
    
    // Simple version without active set for stability
    for (int iter = 1; iter <= maxit; ++iter) {
        Rcpp::checkUserInterrupt();
        
        double max_delta = 0.0;
        
        for (int j = 0; j < p; ++j) {
            arma::vec beta_old_j = param.row(j).t();
            arma::rowvec block_grad_j = X.col(j).t() * R;
            
            double lr = learning_rates(j);
            arma::vec beta_unprox_j = beta_old_j - lr * block_grad_j.t();
            arma::vec beta_new_j;
            
            if (penalty == 1) {
                double scaled_lam1 = lam1 * lr;
                double scaled_lam2 = lam2 * lr;
                beta_new_j = prox_elastic_net_vec(beta_unprox_j, scaled_lam1, scaled_lam2);
            } else {
                double scaled_lam1 = lam1 * lr;
                double a_scad = (lam2 >= 2.1) ? lam2 : 3.7;
                beta_new_j = prox_scad_vec(beta_unprox_j, scaled_lam1, a_scad);
            }
            
            if (pos) {
                beta_new_j.elem(arma::find(beta_new_j < 0.0)).zeros();
            }
            
            arma::vec delta_beta_j = beta_new_j - beta_old_j;
            double max_coef_change = arma::max(arma::abs(delta_beta_j));
            
            if (max_coef_change > 1e-12) {
                param.row(j) = beta_new_j.t();
                Eta += X.col(j) * delta_beta_j.t();
                max_delta = std::max(max_delta, max_coef_change);
            }
        }
        
        // Always update probabilities
        update_probabilities_optimized(Eta, P, offset);
        R = Y_one_hot.cols(1, K) - P.cols(1, K);
        
        if (verbose && iter % 50 == 0) {
            Rcpp::Rcout << "Iter " << iter << " | Max Change: " << max_delta << std::endl;
        }
        
        if (max_delta < tolerance) {
            converged = true;
            convergence_iter = iter;
            break;
        }
        
        if (!param.is_finite()) {
            Rcpp::warning("Non-finite parameter values detected.");
            break;
        }
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
}


/**
 * Applies the native proximal step (Elastic Net or SCAD).
 */
void apply_proximal_step_native(
        arma::mat& param,
        const arma::mat& param_unprox,
        double learning_rate,
        int reg_p, int p, int K,
        int penalty, // 1 = elastic.net, 2 = scad
        double lam1, double lam2,
        bool pos
) {
    
    // Check if there is any penalty to apply
    if (lam1 <= 0.0 && (penalty == 2 || lam2 <= 0.0)) {
        param = param_unprox;
        return;
    }
    
    // Apply penalty only to the first reg_p rows
    if (reg_p > 0) {
        arma::mat U = param_unprox.head_rows(reg_p);
        
        // Loop over K classes to apply vector-wise prox
        for(int k = 0; k < K; ++k) {
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


// -----------------------------------------------------------------------------
// NATIVE SAGA SOLVER
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSAGAN(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int reg_p,
        int penalty = 1, // 1 = elastic.net, 2 = scad
        double lam1 = 0.0,
        double lam2 = 0.0,
        double tolerance = 1e-4,
        double lr_adj = 1.0,
        double max_lr = 1.0,
        int maxit = 500,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    
    //  Learning Rate
    if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant..." << std::endl;
    double L = get_lambda_max(X) / 4.0;
    double base_learning_rate = 1.0 / (6.0 * L + 1e-12);
    double learning_rate = std::min(lr_adj * base_learning_rate, max_lr);
    
    if (verbose) Rcpp::Rcout << "  Lipschitz constant L = " << L << ", Learning Rate = " << learning_rate << std::endl;
    
    // Initialize Parameters 
    arma::mat param(p, K, arma::fill::zeros);
    arma::mat param_for_init(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }
    
    // Initialize Gradient Table
    if (verbose) Rcpp::Rcout << "Initializing gradient table..." << std::endl;
    arma::cube grad_table(p, K, n);
    arma::mat grad_avg(p, K, arma::fill::zeros);
    
    for(int i = 0; i < n; ++i) {
        grad_table.slice(i) = grad_multinom_single(X.row(i), Y(i), offset.row(i), 
                         param_for_init, // <-- USE ZEROS
                         K);
        grad_avg += grad_table.slice(i);
    }
    grad_avg /= static_cast<double>(n);
    if (verbose) Rcpp::Rcout << "Initialization complete." << std::endl;
    
    // SAGA Loop 
    arma::mat param_old(p, K);
    bool converged = false;
    int convergence_iter = -1;
    
    double best_kkt_violation = arma::datum::inf;
    int epochs_since_improvement = 0;
    const int patience = 10; 
    const double lr_decrease_factor = 0.5;
    double diff;
    
    for (int iter = 1; iter <= maxit; ++iter) {
        Rcpp::checkUserInterrupt();
        param_old = param;
        
        arma::uvec indices = arma::randperm(n);
        
        for (arma::uword j = 0; j < n; ++j) {
            int i = indices(j);
            
            // Get old gradient from table
            arma::mat grad_old_i = grad_table.slice(i);
            
            // Compute new gradient for this sample
            arma::mat grad_new_i = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);
            
            // SAGA update direction
            arma::mat saga_update_direction = grad_new_i - grad_old_i + grad_avg;
            
            // Update average gradient and gradient table
            grad_avg += (grad_new_i - grad_old_i) / static_cast<double>(n);
            grad_table.slice(i) = grad_new_i;
            
            // Gradient descent step
            arma::mat param_unprox = param - learning_rate * saga_update_direction;
            
            // Apply native proximal step
            apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p, K,
                                       penalty, lam1, lam2, pos);
        }
        
        // Convergence Check
        double diff;
        arma::mat grad_penalized = grad_avg.head_rows(reg_p);
        arma::mat violations;
        
        violations = arma::max(
            arma::zeros(reg_p, K), 
            arma::abs(grad_penalized) - lam1
        );
        double v_max = violations.max();
        
        // --- 6. Adaptive Learning Rate Logic ---
        // (This logic is fine, but now it's working with a stabler base LR)
        if (v_max < (best_kkt_violation - tolerance * 0.1)) {
            best_kkt_violation = v_max;
            epochs_since_improvement = 0;
        } else {
            epochs_since_improvement++;
        }
        
        if (epochs_since_improvement >= patience) {
            learning_rate *= lr_decrease_factor;
            if (verbose) {
                Rcpp::Rcout << " (No improvement for " << patience << " epochs. LR decreased to " 
                            << learning_rate << ")";
            }
            epochs_since_improvement = 0;
            best_kkt_violation = v_max; 
        }
        
        diff = arma::abs(param - param_old).max();
        
        if (verbose && (iter % 10 == 0 || iter == 1)) {
            Rcpp::Rcout << "Iter " << iter << " | Relative Change: " << diff << std::endl;
        }
        
        if (iter >= 3 && diff < tolerance) {
            converged = true;
            convergence_iter = iter;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
            break;
        }
        
        if (learning_rate < 1e-12) {
            if(verbose) Rcpp::Rcout << "Learning rate is zero. Stopping." << std::endl;
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

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSVRG_Native(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int reg_p,
        int penalty = 1, // 1 = elastic.net, 2 = scad
        double lam1 = 0.0,
        double lam2 = 0.0,
        double tolerance = 1e-4,
        double learning_rate = 1e-2, 
        int maxit = 100,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    const int p = X.n_cols;
    const int n = X.n_rows;
    
    // --- 1. Initialize Parameters ---
    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    }
    
    // --- 2. SVRG Loop (No grad_table) ---
    arma::mat param_old(p, K);
    arma::mat param_snapshot(p, K);
    arma::mat grad_full_snapshot(p, K);
    
    bool converged = false;
    int convergence_iter = -1;
    const int min_iter = (param_start.isNotNull()) ? 1 : 2; // min_iter fix
    double diff;
    
    // --- Adaptive LR Settings ---
    double best_diff = arma::datum::inf;
    int epochs_since_improvement = 0;
    const int patience = 5; // How many epochs to wait before reducing LR
    const double lr_decrease_factor = 0.5;
    
    for (int iter = 1; iter <= maxit; ++iter) {
        Rcpp::checkUserInterrupt();
        
        // --- SVRG Snapshot ---
        // 1. Store the current parameter state
        param_snapshot = param;
        
        // 2. Compute the full, *average* gradient at this snapshot
        // This is one O(n*p*K) operation, but it's done *outside* the inner loop
        grad_full_snapshot = grad_multinom_batch(X, Y, offset, param_snapshot, K);
        
        // Store parameters before the inner loop for convergence check
        param_old = param;
        
        // --- 3. Inner Stochastic Loop ---
        arma::uvec indices = arma::randperm(n);
        
        for (arma::uword j = 0; j < n; ++j) {
            int i = indices(j);
            
            // Compute new gradient at *current* param
            arma::mat grad_new_i = grad_multinom_single(X.row(i), Y(i), offset.row(i), param, K);
            
            // Compute "old" gradient at the *snapshot* param
            arma::mat grad_old_i = grad_multinom_single(X.row(i), Y(i), offset.row(i), param_snapshot, K);
            
            // 4. SVRG update direction (corrects the noise)
            arma::mat svrg_update_direction = grad_new_i - grad_old_i + grad_full_snapshot;
            
            // Gradient descent step
            arma::mat param_unprox = param - learning_rate * svrg_update_direction;
            
            // Apply native proximal step
            apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p, K,
                                       penalty, lam1, lam2, pos);
        } // --- End inner loop ---
        
        // --- 5. Convergence & Adaptive LR Check ---
        diff = arma::abs(param - param_old).max();
        
        // Check if we are improving
        if (diff < best_diff) {
            best_diff = diff;
            epochs_since_improvement = 0;
        } else {
            epochs_since_improvement++;
        }
        
        // If no improvement for 'patience' epochs, cut the learning rate
        if (epochs_since_improvement >= patience) {
            learning_rate *= lr_decrease_factor;
            if (verbose) {
                Rcpp::Rcout << " (No improvement. LR decreased to " << learning_rate << ")";
            }
            // Reset counters
            epochs_since_improvement = 0;
            best_diff = arma::datum::inf; // Reset best diff
        }
        
        if (verbose && (iter % 1 == 0)) { // SVRG epochs are slow, print every time
            Rcpp::Rcout << "Epoch " << iter << " | Max Param Change: " << diff << std::endl;
        }
        
        if (iter >= min_iter && diff < tolerance) {
            converged = true;
            convergence_iter = iter;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
            break;
        }
        
        if (learning_rate < 1e-12) {
            if(verbose) Rcpp::Rcout << "Learning rate is zero. Stopping." << std::endl;
            break;
        }
        
        if (!param.is_finite()) {
            Rcpp::warning("Algorithm diverged with non-finite parameter values.");
            break;
        }
    } // --- End epoch loop ---
    
    return Rcpp::List::create(
        Rcpp::Named("Estimates") = param,
        Rcpp::Named("Converged") = converged,
        Rcpp::Named("Convergence Iteration") = convergence_iter
    );
}

// -----------------------------------------------------------------------------
// Helper: compute per-sample residual row (1 x K) = p_hat - y_onehot
// Uses stable softmax with baseline score = 0.0 (explicit K classes in param)
// -----------------------------------------------------------------------------

static inline double estimate_lipschitz_default(const arma::mat &X) {
    // cheap approximation: max row norm squared / 4
    double max_row_norm_sq = 0.0;
    for (arma::uword i = 0; i < X.n_rows; ++i) {
        double s = arma::dot(X.row(i), X.row(i));
        if (s > max_row_norm_sq) max_row_norm_sq = s;
    }
    return std::max(1e-12, max_row_norm_sq / 4.0);
}

static inline arma::rowvec compute_residual_row(const arma::rowvec& xrow,
                                                int yi,
                                                double offset_i,
                                                const arma::mat& param) {
    // linear scores for explicit classes (1 x K)
    arma::rowvec scores = xrow * param; // 1 x K
    // add offset to explicit classes (keeps original code behavior)
    scores += offset_i;
    
    // stable softmax against baseline 0
    double max_score = scores.max();
    if (max_score < 0.0) max_score = 0.0; // compare with baseline
    arma::rowvec ex = arma::exp(scores - max_score);
    double denom = arma::accu(ex) + std::exp(0.0 - max_score) + arma::datum::eps;
    arma::rowvec p_hat = ex / denom; // 1 x K
    
    arma::rowvec r = p_hat;
    if (yi > 0 && yi <= (int)p_hat.n_cols) r(yi - 1) -= 1.0;
    return r; // 1 x K
}

// -----------------------------------------------------------------------------
// NATIVE SAGA SOLVER (residual-cache implementation)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List MultinomLogisticSAGA_Native(
        const arma::mat& X,
        const arma::vec& Y,
        const arma::vec& offset,
        int K,
        int reg_p,
        int penalty = 1, // 1 = elastic.net, 2 = scad
        double lam1 = 0.0,
        double lam2 = 0.0,
        double tolerance = 1e-4,
        double lr_adj = 1.0,
        double max_lr = 1.0,
        int maxit = 500,
        bool verbose = false,
        bool pos = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {

    const int p = X.n_cols;
    const int n = X.n_rows;

    if (reg_p < 0) reg_p = 0;
    if (reg_p > p) reg_p = p;

    // 1) Lipschitz approx (max row norm^2 / 4) - cheap and stable
    if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant (approx)..." << std::endl;

    double L = estimate_lipschitz_default(X);
    double base_learning_rate = 1.0 / (6.0 * L + 1e-12);
    double learning_rate = std::min(lr_adj * base_learning_rate, max_lr);
    if (verbose) Rcpp::Rcout << "  L = " << L << ", LR = " << learning_rate << std::endl;

    // 2) initialize parameters with warm-start if provided
    arma::mat param(p, K, arma::fill::zeros);
    if (param_start.isNotNull()) {
        param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
        if ((int)param.n_rows != p || (int)param.n_cols != K) {
            Rcpp::stop("param_start has incompatible dimensions");
        }
    }

    // 3) Initialize residual cache R (n x K) using current param (warm-start correctness)
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
            r_new = compute_residual_row(X.row(i), static_cast<int>(Y(i)), offset(i), param);
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
            apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p, K,
                                       penalty, lam1, lam2, pos);
        }

        // Convergence & KKT check every epoch
        arma::mat grad_penalized = grad_avg.head_rows(reg_p);
        arma::mat violations = arma::max(arma::zeros(reg_p, K), arma::abs(grad_penalized) - lam1);
        double v_max = (violations.n_elem > 0) ? violations.max() : 0.0;

        if (v_max < (best_kkt_violation - tolerance * 0.1)) {
            best_kkt_violation = v_max;
            epochs_since_improvement = 0;
        } else {
            epochs_since_improvement++;
        }
        if (epochs_since_improvement >= patience) {
            learning_rate *= lr_decrease_factor;
            if (verbose) Rcpp::Rcout << " (No improvement for " << patience << " epochs. LR decreased to " << learning_rate << ")\n";
            epochs_since_improvement = 0;
            best_kkt_violation = v_max;
        }

        double diff = arma::abs(param - param_old).max();
        if (verbose && (iter % 10 == 0 || iter == 1)) {
            Rcpp::Rcout << "Iter " << iter << " | Max param change: " << diff << " | KKT viol: " << v_max << std::endl;
        }

        if (iter >= 3 && diff < tolerance) {
            converged = true;
            convergence_iter = iter;
            if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << std::endl;
            break;
        }

        if (learning_rate < 1e-12) {
            if (verbose) Rcpp::Rcout << "Learning rate became too small. Stopping." << std::endl;
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
// Rcpp::List MultinomLogisticSAGA_Native(
//         const arma::mat& X,
//         const arma::vec& Y,
//         const arma::vec& offset,
//         int K,
//         int reg_p,
//         int penalty = 1, // 1 = elastic.net, 2 = scad
//         double lam1 = 0.0,
//         double lam2 = 0.0,
//         double tolerance = 1e-4,
//         double lr_adj = 1.0,
//         double max_lr = 1.0,
//         int maxit = 500,
//         bool verbose = false,
//         bool pos = false,
//         Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue,
//         int min_epochs = 5,          // NEW optional: minimum epochs before stopping
//         int patience_stop = 3        // NEW optional: consecutive epochs needed to accept stopping
// ) {
//     const int p = X.n_cols;
//     const int n = X.n_rows;
//     
//     if (reg_p < 0) reg_p = 0;
//     if (reg_p > p) reg_p = p;
//     
//     // --- 1. Learning Rate ---
//     if (verbose) Rcpp::Rcout << "Estimating Lipschitz constant (approx)..." << std::endl;
//     double L = estimate_lipschitz_default(X);
//     double base_learning_rate = 1.0 / (6.0 * L + 1e-12);
//     double learning_rate = std::min(lr_adj * base_learning_rate, max_lr);
//     if (verbose) Rcpp::Rcout << "  L = " << L << ", Base LR = " << base_learning_rate << ", Initial LR = " << learning_rate << std::endl;
//     
//     // --- 2. Initialize Parameters (warm start allowed) ---
//     arma::mat param(p, K, arma::fill::zeros);
//     bool is_warm = false;
//     if (param_start.isNotNull()) {
//         param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
//         if ((int)param.n_rows != p || (int)param.n_cols != K) {
//             Rcpp::stop("param_start has incompatible dimensions");
//         }
//         // consider warm if not all zeros
//         if (arma::abs(param).max() > 0.0) is_warm = true;
//     }
//     
//     // If warm-starting, be conservative on LR and increase min_epochs
//     if (is_warm) {
//         learning_rate = std::min(learning_rate, base_learning_rate * 0.5);
//         min_epochs = std::max(min_epochs, std::min(50, min_epochs * 3)); // give more time to settle (tunable)
//         if (verbose) Rcpp::Rcout << " Warm start detected. LR scaled to " << learning_rate << " and min_epochs -> " << min_epochs << std::endl;
//     }
//     
//     // --- 3. Residual cache and grad_avg initialized from param (warm-start correct) ---
//     arma::mat Rcache(n, K, arma::fill::zeros);
//     for (int i = 0; i < n; ++i) {
//         int yi = static_cast<int>(Y(i));
//         Rcache.row(i) = compute_residual_row(X.row(i), yi, offset(i), param);
//     }
//     arma::mat grad_avg = (X.t() * Rcache) / static_cast<double>(n);
//     
//     // Preallocs
//     arma::mat param_old(p, K);
//     arma::rowvec r_old(K); arma::rowvec r_new(K);
//     
//     bool converged = false;
//     int convergence_iter = -1;
//     
//     // Adaptive LR logic
//     double last_kkt_violation = arma::datum::inf;
//     const double lr_increase_factor = 1.02;
//     const double lr_decrease_factor = 0.5;
//     
//     int consecutive_good = 0; // must have `patience_stop` consecutive good epochs before stopping
//     
//     // Main loop
//     for (int iter = 1; iter <= maxit; ++iter) {
//         Rcpp::checkUserInterrupt();
//         param_old = param;
//         
//         arma::uvec indices = arma::randperm(n);
//         
//         for (arma::uword jj = 0; jj < indices.n_elem; ++jj) {
//             int i = static_cast<int>(indices(jj));
//             r_old = Rcache.row(i);
//             r_new = compute_residual_row(X.row(i), static_cast<int>(Y(i)), offset(i), param);
//             
//             arma::rowvec delta_r = r_new - r_old;
//             arma::mat x_outer = X.row(i).t() * delta_r; // p x K
//             arma::mat saga_update_direction = x_outer + grad_avg;
//             
//             // update cache and grad_avg
//             grad_avg += x_outer / static_cast<double>(n);
//             Rcache.row(i) = r_new;
//             
//             // step + prox
//             arma::mat param_unprox = param - learning_rate * saga_update_direction;
//             apply_proximal_step_native(param, param_unprox, learning_rate, reg_p, p, K, penalty, lam1, lam2, pos);
//         }
//         
//         // compute KKT-ish violation (penalized rows)
//         arma::mat grad_penalized = grad_avg.head_rows(reg_p);
//         arma::mat violations;
//         if (penalty == 1 && lam1 > 0.0) {
//             violations = arma::max(arma::zeros(reg_p, K), arma::abs(grad_penalized) - lam1);
//         } else {
//             violations = arma::abs(grad_penalized);
//         }
//         double kkt_violation = (violations.n_elem > 0) ? violations.max() : 0.0;
//         
//         // parameter change checks (global + penalized)
//         double diff_all = arma::abs(param - param_old).max();
//         double diff_pen = (reg_p > 0) ? arma::abs(param.head_rows(reg_p) - param_old.head_rows(reg_p)).max() : diff_all;
//         
//         // Adaptive LR update and revert if necessary
//         if (kkt_violation < (last_kkt_violation - tolerance * 0.1)) {
//             // improving
//             last_kkt_violation = kkt_violation;
//             learning_rate = std::min(learning_rate * lr_increase_factor, max_lr);
//             // do not reset consecutive_good here since we need both KKT and stability
//         } else if (iter > 3) {
//             // oscillating/worse, reduce LR and revert to previous epoch state
//             double new_lr = std::max(learning_rate * lr_decrease_factor, 1e-16);
//             if (verbose) Rcpp::Rcout << "Epoch " << iter << ": oscillation detected. LR decreased " << learning_rate << " -> " << new_lr << std::endl;
//             learning_rate = new_lr;
//             
//             // revert params to previous epoch and recompute caches
//             param = param_old;
//             for (int ii = 0; ii < n; ++ii) {
//                 Rcache.row(ii) = compute_residual_row(X.row(ii), static_cast<int>(Y(ii)), offset(ii), param);
//             }
//             grad_avg = (X.t() * Rcache) / static_cast<double>(n);
//             
//             // recompute baseline KKT
//             arma::mat grad_pen_recomp = grad_avg.head_rows(reg_p);
//             arma::mat viol_recomp;
//             if (penalty == 1 && lam1 > 0.0) {
//                 viol_recomp = arma::max(arma::zeros(reg_p, K), arma::abs(grad_pen_recomp) - lam1);
//             } else {
//                 viol_recomp = arma::abs(grad_pen_recomp);
//             }
//             last_kkt_violation = (viol_recomp.n_elem > 0) ? viol_recomp.max() : 0.0;
//             
//             // reset consecutive_good because we reverted
//             consecutive_good = 0;
//         }
//         
//         if (verbose && (iter % 5 == 0 || iter == 1)) {
//             Rcpp::Rcout << "Iter " << iter << " | KKT: " << kkt_violation << " | diff_all: " << diff_all
//                         << " | diff_pen: " << diff_pen << " | LR: " << learning_rate << " | consec_good: " << consecutive_good << std::endl;
//         }
//         
//         // Stopping: require min_epochs, small KKT, small penalized change, and hold for patience_stop consecutive epochs
//         bool small_enough = (kkt_violation < tolerance) && (diff_pen < tolerance);
//         if (iter >= min_epochs && small_enough) {
//             consecutive_good += 1;
//         } else {
//             consecutive_good = 0;
//         }
//         
//         if (consecutive_good >= patience_stop) {
//             converged = true;
//             convergence_iter = iter;
//             if (verbose) Rcpp::Rcout << "Converged at iteration " << iter << " (consecutive_good=" << consecutive_good << ")" << std::endl;
//             break;
//         }
//         
//         if (learning_rate < 1e-12) {
//             if (verbose) Rcpp::Rcout << "Learning rate became too small. Stopping." << std::endl;
//             break;
//         }
//         if (!param.is_finite()) {
//             Rcpp::warning("Algorithm diverged with non-finite parameter values.");
//             break;
//         }
//     } // end epoch loop
//     
//     
//     return Rcpp::List::create(
//         Rcpp::Named("Estimates") = param,
//         Rcpp::Named("Converged") = converged,
//         Rcpp::Named("Convergence Iteration") = convergence_iter
//     );
// }

