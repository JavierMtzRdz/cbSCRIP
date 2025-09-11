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



void proximalFlat(
    arma::mat& U,
    int& p,              // # of rows of U
    int& K,              // # of columns of U
    std::string& regul,
    Rcpp::IntegerVector& grp_id, // grp is a vector for proximalFlat
    int num_threads,
    double lam1,
    double lam2 = 0.0,
    double lam3 = 0.0,
    bool intercept = false,
    bool resetflow = false,
    bool verbose = false,
    bool pos = false,
    bool clever = true,
    bool eval = true,
    int size_group = 1,
    bool transpose = false) {
  // dimensions
  // int p = U.n_rows;
  // int K = U.n_cols;
  int pK = int (p*K);
  
  // read in U and convert to spams::Matrix<double> alpha0
  double* ptrU = new double[pK];
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      ptrU[c * p + r] = U(r, c);
    }
  }
  Matrix<double> alpha0(ptrU,p,K);
  
  
  
  // read in regul and convert to char
  int l = regul.length();
  char* name_regul = new char[l];
  for (int i = 0; i < l+1; i++) {
    name_regul[i] = regul[i];
  }
  
  
  
  // Initialize alpha - proximal operator
  Matrix<double> alpha(p,K);
  alpha.setZeros();
  
  // read in grp_id and convert to spams::Vector<int> groups
  int* ptrG = new int[p];
  for (int i = 0; i < p; i++) {
    ptrG[i] = grp_id(i);
  }
  
  
  
  Vector<int> groups(ptrG, p);
  
  _proximalFlat(&alpha0,&alpha,&groups,num_threads,
                lam1,lam2,lam3,intercept,
                resetflow,name_regul,verbose,pos,
                clever,eval,size_group,transpose);
  
  // put updated alpha back into U
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      U(r, c) = alpha[p * c + r];
    }
  }
  
  
  // free the dynamic memory
  delete[] ptrU;
  delete[] name_regul;
  delete[] ptrG;
}





void proximalGraph(
    arma::mat& U,
    int& p,              // # of rows of U
    int& K,              // # of columns of U
    std::string& regul,
    arma::mat& grp,      // grp is a matrix for proximalGraph and Tree
    arma::mat& grpV,
    Rcpp::NumericVector& etaG,
    int num_threads,
    double lam1,
    double lam2 = 0.0,
    double lam3 = 0.0,
    bool intercept = false,
    bool resetflow = false,
    bool verbose = false,
    bool pos = false,
    bool clever = true,
    bool eval = true,
    int size_group = 1,
    bool transpose = false) {
  
  // U dimensions
  // int p = U.n_rows;
  // int K = U.n_cols;
  int pK = int (p*K);
  
  // read in U and convert to spams::Matrix<double> alpha0
  double* ptrU = new double[pK];
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      ptrU[c * p + r] = U(r, c);
    }
  }
  Matrix<double> alpha0(ptrU,p,K);
  
  
  
  // grp dimensions
  int gr = grp.n_rows;
  int gc = grp.n_cols;
  int grc = int (gr * gc);
  
  // read in grp and convert to spams::Matrix<bool> grp_dense
  // then to spams::SpMatrix<bool> groups
  bool* ptrG = new bool[grc];
  for (int r = 0; r < gr; r++) {
    for (int c = 0; c < gc; c++) {
      ptrG[c * gr + r] = (grp(r, c) != 0.0);
    }
  }
  Matrix<bool> grp_dense(ptrG, gr, gc);
  SpMatrix<bool> groups;
  grp_dense.toSparse(groups);
  
  
  
  // grpV dimensions
  int gvr = grpV.n_rows;
  int gvc = grpV.n_cols;
  int gvrc = int (gvr * gvc);
  
  // read in grpV and convert to spams::Matrix<bool> grpV_dense
  // then to spams::SpMatrix<bool> groups_var
  bool* ptrGV = new bool[gvrc];
  for (int r = 0; r < gvr; r++) {
    for (int c = 0; c < gvc; c++) {
      ptrGV[c * gvr + r] = (grpV(r, c) != 0.0);
    }
  }
  Matrix<bool> grpV_dense(ptrGV, gvr, gvc);
  SpMatrix<bool> groups_var;
  grpV_dense.toSparse(groups_var);
  
  
  
  // read in etaG and convert to spams::Vector<int> eta_g
  int n_etaG = etaG.length();
  double* ptrEG = new double[n_etaG];
  for (int i = 0; i < n_etaG; i++) {
    ptrEG[i] = etaG(i);
  }
  Vector<double> eta_g(ptrEG, n_etaG);
  
  
  
  // read in regul and convert to char
  int l = regul.length();
  char* name_regul = new char[l];
  for (int i = 0; i < l+1; i++) {
    name_regul[i] = regul[i];
  }
  
  
  
  // Initialize alpha - proximal operator
  Matrix<double> alpha(p,K);
  alpha.setZeros();
  
  
  // call _proximalGraph
  _proximalGraph(&alpha0, &alpha,
                 &eta_g, &groups, &groups_var,
                 num_threads, lam1, lam2, lam3,
                 intercept, resetflow, name_regul,
                 verbose, pos, clever, eval,
                 size_group, transpose);
  
  
  
  // put updated alpha back into U
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      U(r, c) = alpha[c * p + r];
    }
  }
  
  // free the dynamic memory
  delete[] ptrU;
  delete[] ptrG;
  delete[] ptrGV;
  delete[] ptrEG;
  delete[] name_regul;
}




void proximalTree(
    arma::mat& U,
    int& p,              // # of rows of U
    int& K,              // # of columns of U
    std::string& regul,
    arma::mat& grp,      // grp is a matrix for proximalGraph and Tree
    Rcpp::NumericVector& etaG,
    Rcpp::IntegerVector& own_var,
    Rcpp::IntegerVector& N_own_var,
    int num_threads,
    double lam1,
    double lam2 = 0.0,
    double lam3 = 0.0,
    bool intercept = false,
    bool resetflow = false,
    bool verbose = false,
    bool pos = false,
    bool clever = true,
    bool eval = true,
    int size_group = 1,
    bool transpose = false) {
  
  
  
  // U dimensions
  // int p = U.n_rows;
  // int K = U.n_cols;
  int pK = int (p*K);
  
  // read in U and convert to spams::Matrix<double> alpha0
  double* ptrU = new double[pK];
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      ptrU[c * p + r] = U(r, c);
    }
  }
  Matrix<double> alpha0(ptrU,p,K);
  
  
  
  // grp dimensions
  int gr = grp.n_rows;
  int gc = grp.n_cols;
  int grc = int (gr * gc);
  
  // read in grp and convert to spams::Matrix<bool> grp_dense
  // then to spams::SpMatrix<bool> groups
  bool* ptrG = new bool[grc];
  for (int r = 0; r < gr; r++) {
    for (int c = 0; c < gc; c++) {
      ptrG[c * gr + r] = ( grp(r, c) != 0.0 );
    }
  }
  Matrix<bool> grp_dense(ptrG, gr, gc);
  SpMatrix<bool> groups;
  grp_dense.toSparse(groups);
  
  
  
  // read in etaG and convert to spams::Vector<double> eta_g
  int n_etaG = etaG.length();
  double* ptrEG = new double[n_etaG];
  for (int i = 0; i < n_etaG; i++) {
    ptrEG[i] = etaG(i);
  }
  Vector<double> eta_g(ptrEG, n_etaG);
  
  
  
  // read in own_var and convert to spams::Vector<int> own_variables
  int n_ov = own_var.length();
  int* ptrOV = new int[n_ov];
  for (int i = 0; i < n_ov; i++) {
    ptrOV[i] = own_var(i);
  }
  Vector<int> own_variables(ptrOV, n_ov);
  
  
  
  // read in N_own_var and convert to spams::Vector<int> N_own_variables
  int n_Nov = N_own_var.length();
  int* ptrNOV = new int[n_Nov];
  for (int i = 0; i < n_Nov; i++) {
    ptrNOV[i] = N_own_var(i);
  }
  Vector<int> N_own_variables(ptrNOV, n_Nov);
  
  
  
  // read in regul and convert to char
  int l = regul.length();
  char* name_regul = new char[l];
  for (int i = 0; i < l+1; i++) {
    name_regul[i] = regul[i];
  }
  
  
  
  // Initialize alpha - proximal operator
  Matrix<double> alpha(p,K);
  alpha.setZeros();
  
  
  
  // call _proximalTree
  _proximalTree(&alpha0, &alpha,
                &eta_g, &groups,
                &own_variables, &N_own_variables,
                num_threads, lam1, lam2, lam3,
                intercept, resetflow, name_regul,
                verbose, pos, clever, eval,
                size_group, transpose);
  
  
  
  // put updated alpha back into U
  for (int r = 0; r < p; r++) {
    for (int c = 0; c < K; c++) {
      U(r, c) = alpha[c * p + r];
    }
  }
  
  // free the dynamic memory
  delete[] ptrU;
  delete[] ptrG;
  delete[] ptrEG;
  delete[] ptrOV;
  delete[] ptrNOV;
  delete[] name_regul;
}





// [[Rcpp::export]]
Rcpp::List mtool(
    arma::mat X,            // input
    arma::mat Y,            // outcome
    arma::vec wt,
    int K,                  // number of tasks
    int reg_p,              // number of regularized variables
    arma::vec nk_vec,
    arma::vec task_rowid,
    int loss,               // 1 - least squares; 2 - logistic regression; 3 - Cox
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
  
  // initialize param
  int p = X.n_cols;
  
  // indexing, stochastic sampling, etc..
  int init_k;   // index of the first obs in task k
  int n_k;      // number of obs in task k
  int id_ik;    // index of the i-th obs in task k
  int index;    // index of the stochastic sample
  arma::rowvec x_sample(p);
  double y_sample;
  double wt_sample;
  
  // gradient and related
  arma::mat grad(p, K);
  arma::vec grad_k(p);         // k-th column of grad
  arma::vec temp1(p);
  arma::vec temp2(p);
  
  // parameter and related
  arma::mat param(p, K);
  param.zeros();
  arma::vec param_k(p);     // k-th column of param
  arma::mat param_old(p, K);
  arma::vec param_old_k(p);
  
  arma::mat param_reg(reg_p, K);   // regularized coefficients
  arma::mat param_t(K, reg_p);      // regularized coefficients
  
  // convergence related
  // arma::mat param_update(p, K);
  double diff;
  int counter_outer = 0;
  
  // compute mu: mean gradient at param_old
  while (true) {
    param_old = param;
    init_k = 0;
    
    for (int k = 0; k < K; k++) {
      n_k = nk_vec(k);
      param_old_k = param_old.col(k);
      grad_k.zeros();
      
      for (int i = 0; i < n_k; i++) {
        id_ik = task_rowid(init_k + i) - 1;
        x_sample = X.row(id_ik);
        y_sample = Y(id_ik, k);
        wt_sample = wt(id_ik);
        if (loss == 1) {
          grad_k = grad_k + wt_sample * grad_ls_loss(x_sample, y_sample, param_old_k, p)/n_k;
        }
        if (loss == 2) {
          grad_k = grad_k + wt_sample * grad_logistic_loss(x_sample, y_sample, param_old_k, p)/n_k;
        }
      }
      grad.col(k) = grad_k;
      init_k += n_k;
    }
    
    // inner loop
    for (int i = 0; i < niter_inner; ++i) {
      init_k = 0;
      
      for (int k = 0; k < K; k++) {
        n_k = nk_vec(k);
        index = arma::randi(arma::distr_param(0, n_k - 1));
        id_ik = task_rowid(init_k + index) - 1;
        
        x_sample = X.row(id_ik);
        y_sample = Y(id_ik, k);
        wt_sample = wt(id_ik);
        
        param_k = param.col(k);
        param_old_k = param_old.col(k);
        
        if (loss == 1) {
          temp1 = grad_ls_loss(x_sample, y_sample, param_k, p);
          temp2 = grad_ls_loss(x_sample, y_sample, param_old_k, p);
        }
        
        if (loss == 2) {
          temp1 = grad_logistic_loss(x_sample, y_sample, param_k, p);
          temp2 = grad_logistic_loss(x_sample, y_sample, param_old_k, p);
        }
        
        param_k = param_k - learning_rate * (wt_sample * (temp1 - temp2) + grad.col(k));
        
        param.col(k) = param_k;
        
        init_k += n_k;
      }
      
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
    Rcpp::Rcout << "Mean Frobenius norm of coefficient update \n" << diff <<"\n";
    
    if (diff < tolerance || counter_outer>= maxit) {
      break;
    }
  }
  
  arma::sp_mat param_sp(param);
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("Estimates")                     = param,
                                         Rcpp::Named("Sparse Estimates")              = param_sp);
  return result;
}



// [[Rcpp::export]]
Rcpp::List MultinomLogistic(
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
  
  // compute mu: mean gradient at param_old
  while (true) {
    param_old = param;
    grad.zeros();
    
    for (int i = 0; i < n; i++) {
      x_sample = X.row(i);
      y_sample = Y(i);
      o_sample = offset(i);
      grad = grad + grad_multinom_loss(x_sample, y_sample, K, o_sample, param_old, p)/n;
    }

    // inner loop
    for (int i = 0; i < niter_inner; ++i) {

      index = arma::randi(arma::distr_param(0, n - 1));

      x_sample = X.row(index);
      y_sample = Y(index);
      o_sample = offset(index);

      temp1 = grad_multinom_loss(x_sample, y_sample, K, o_sample, param, p);
      temp2 = grad_multinom_loss(x_sample, y_sample, K, o_sample, param_old, p);

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
    
    
    if (diff < tolerance || counter_outer>= maxit) {
      break;
    }
  }
  
  arma::sp_mat param_sp(param);
  
  Rcpp::List result = Rcpp::List::create(Rcpp::Named("Estimates")                     = param,
                                         Rcpp::Named("Sparse Estimates")              = param_sp);
  return result;
}





void proximalFlat2(
        arma::mat& U,
        const int& p,     
        const int& K,
        const std::string& regul,      
        Rcpp::IntegerVector& grp_id,
        int num_threads,
        double lam1,
        double lam2 = 0.0,
        double lam3 = 0.0,
        bool intercept = false,
        bool resetflow = false,
        bool verbose = false,
        bool pos = false,
        bool clever = true,
        bool eval = true,
        int size_group = 1,
        bool transpose = false) {
    // dimensions
    // int p = U.n_rows;
    // int K = U.n_cols;
    // int pK = int (p*K);
    
    // read in U and convert to spams::Matrix<double> alpha0
    // double* ptrU = new double[pK];
    // for (int r = 0; r < p; r++) {
    //     for (int c = 0; c < K; c++) {
    //         ptrU[c * p + r] = U(r, c);
    //     }
    // }
    // Matrix<double> alpha0(ptrU,p,K);
    Matrix<double> alpha0(const_cast<double*>(U.memptr()), p, K);
    
    
    
    // read in regul and convert to char
    // int l = regul.length();
    // char* name_regul = new char[l];
    // for (int i = 0; i < l+1; i++) {
    //     name_regul[i] = regul[i];
    // }
    std::vector<char> name_regul_buf(regul.begin(), regul.end()); 
    name_regul_buf.push_back('\0'); 
    char* name_regul_ptr = name_regul_buf.data();
    
    
    
    // Initialize alpha - proximal operator
    Matrix<double> alpha(p,K);
    alpha.setZeros();
    
    // read in grp_id and convert to spams::Vector<int> groups
    // int* ptrG = new int[p];
    // for (int i = 0; i < p; i++) {
    //     ptrG[i] = grp_id(i);
    // }
    // Vector<int> groups(ptrG, p);
    
    Vector<int> groups(const_cast<int*>(grp_id.begin()), p); 
    
    
    
    
    _proximalFlat(&alpha0,&alpha,&groups,num_threads,
                   lam1,lam2,lam3,intercept,
                   resetflow,name_regul_ptr,verbose,pos,
                   clever,eval,size_group,transpose);
    
    // put updated alpha back into U
    for (int r = 0; r < p; r++) {
        for (int c = 0; c < K; c++) {
            U(r, c) = alpha[p * c + r];
        }
    }
    
    
    // free the dynamic memory
    // delete[] ptrU;
    // delete[] name_regul;
    // delete[] ptrG;
}


void proximalFlat2_nospams(
        arma::mat& U,             
        int p,                    
        int K,                    
        const std::string& regul, 
        const Rcpp::IntegerVector& grp_id, 
        double lam1,              
        double lam2 = 0.0,        
        double lam3 = 0.0,        
        bool intercept = false,   
        bool resetflow = false,   
        bool verbose = false,     
        bool pos = false,         
        bool clever = true,       
        bool eval = true,         
        int size_group = 1,       
        bool transpose = false) { 
    
    if (lam1 < 0.0 || lam2 < 0.0 || lam3 < 0.0) {
        Rcpp::stop("Lambda penalty parameters cannot be negative.");
    }
    
    
    if (regul == "L1") {
        // Elementwise L1 Soft-thresholding: sign(u) * max(|u| - lam1, 0)
        if (lam1 == 0.0) return; // No thresholding needed
        U = arma::sign(U) % arma::max(arma::abs(U) - lam1, arma::zeros(p, K));
        
    } else if (regul == "L2") {
        // Elementwise L2 Scaling (Ridge): u / (1 + 2*lam2)
        if (lam2 == 0.0) return; 
        U /= (1.0 + 2.0 * lam2);
        
    } else if (regul == "ElasticNet") {
        // Composition: Apply L2 scaling then L1 thresholding
        if (lam1 == 0.0 && lam2 == 0.0) return; // No penalty
        
        //  L2 Scaling (if lam2 > 0)
        if (lam2 > 0.0) {
            U /= (1.0 + 2.0 * lam2);
        }
        // L1 Soft-thresholding (if lam1 > 0)
        if (lam1 > 0.0) {
            U = arma::sign(U) % arma::max(arma::abs(U) - lam1, arma::zeros(p, K));
        }
        
    } else if (regul == "GroupLasso") {
        // Group Lasso Soft-thresholding (Block Soft-thresholding).
        if (lam1 == 0.0) return; 
        if (grp_id.length() != p) {
            Rcpp::stop("grp_id length must match number of rows (p) for GroupLasso.");
        }
        
        std::unordered_map<int, std::vector<arma::uword>> groups;
        for(int i = 0; i < p; ++i) {
            groups[grp_id[i]].push_back(i); 
        }
        
        // Iterate through each identified group
        for (auto const& [id, row_indices_vec] : groups) {
            if (row_indices_vec.empty()) continue; // Skip empty groups if any
            
            arma::uvec row_indices(row_indices_vec);
            
            double group_norm = arma::norm(U.rows(row_indices), "fro");
            
            // Apply block soft-thresholding
            double scaling_factor = std::max(0.0, 1.0 - lam1 / group_norm);
            U.rows(row_indices) *= scaling_factor;
            
        } 
        
    } else if (regul == "FusedLasso" || regul == "Graph" || regul == "Tree") {
        Rcpp::warning("Proximal operator for '%s' not implemented in this function. Skipping proximal step.", regul.c_str());
        
    } else {
        Rcpp::warning("Unsupported 'regul' type in proximalFlat2_nospams: '%s'. Skipping proximal step.", regul.c_str());
    }
    
    if (pos) {
        U.elem( arma::find(U < 0.0) ).zeros(); 
    }
    
}

void proximalGraph2(
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
        double lam3 = 0.0,
        bool intercept = false,
        bool resetflow = false,
        bool verbose = false,
        bool pos = false,
        bool clever = true,
        bool eval = true,
        int size_group = 1,
        bool transpose = false) {
    
    // U dimensions
    // int p = U.n_rows;
    // int K = U.n_cols;
    // int pK = int (p*K);
    
    // read in U and convert to spams::Matrix<double> alpha0
    // double* ptrU = new double[pK];
    // for (int r = 0; r < p; r++) {
    //     for (int c = 0; c < K; c++) {
    //         ptrU[c * p + r] = U(r, c);
    //     }
    // }
    //Matrix<double> alpha0(ptrU,p,K);
    Matrix<double> alpha0(const_cast<double*>(U.memptr()), p, K);
    
    
    // grp dimensions
    int gr = grp.n_rows;
    int gc = grp.n_cols;
    int grc = int (gr * gc);
    
    // read in grp and convert to spams::Matrix<bool> grp_dense
    // then to spams::SpMatrix<bool> groups
    bool* ptrG = new bool[grc];
    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            ptrG[c * gr + r] = (grp(r, c) != 0.0);
        }
    }
    Matrix<bool> grp_dense(ptrG, gr, gc);
    SpMatrix<bool> groups;
    grp_dense.toSparse(groups);
    
    
    
    // grpV dimensions
    int gvr = grpV.n_rows;
    int gvc = grpV.n_cols;
    int gvrc = int (gvr * gvc);
    
    // read in grpV and convert to spams::Matrix<bool> grpV_dense
    // then to spams::SpMatrix<bool> groups_var
    bool* ptrGV = new bool[gvrc];
    for (int r = 0; r < gvr; r++) {
        for (int c = 0; c < gvc; c++) {
            ptrGV[c * gvr + r] = (grpV(r, c) != 0.0);
        }
    }
    Matrix<bool> grpV_dense(ptrGV, gvr, gvc);
    SpMatrix<bool> groups_var;
    grpV_dense.toSparse(groups_var);
    
    
    
    // read in etaG and convert to spams::Vector<int> eta_g
    int n_etaG = etaG.length();
    double* ptrEG = new double[n_etaG];
    for (int i = 0; i < n_etaG; i++) {
        ptrEG[i] = etaG(i);
    }
    Vector<double> eta_g(ptrEG, n_etaG);
    
    
    
    // read in regul and convert to char
    // int l = regul.length();
    // char* name_regul = new char[l];
    // for (int i = 0; i < l+1; i++) {
    //     name_regul[i] = regul[i];
    // }
    std::vector<char> name_regul_buf(regul.begin(), regul.end()); // Copy string content
    name_regul_buf.push_back('\0'); // Add the null terminator
    char* name_regul_ptr = name_regul_buf.data();
    
    
    
    // Initialize alpha - proximal operator
    Matrix<double> alpha(p,K);
    alpha.setZeros();
    
    
    // call _proximalGraph
    _proximalGraph(&alpha0, &alpha,
                    &eta_g, &groups, &groups_var,
                    num_threads, lam1, lam2, lam3,
                    intercept, resetflow, name_regul_ptr,
                    verbose, pos, clever, eval,
                    size_group, transpose);
    
    
    
    // put updated alpha back into U
    // for (int r = 0; r < p; r++) {
    //     for (int c = 0; c < K; c++) {
    //         U(r, c) = alpha[c * p + r];
    //     }
    // }
    
    // free the dynamic memory
    // delete[] ptrU;
    delete[] ptrG;
    delete[] ptrGV;
    delete[] ptrEG;
    // delete[] name_regul;
}




void proximalTree2(
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
        double lam3 = 0.0,
        bool intercept = false,
        bool resetflow = false,
        bool verbose = false,
        bool pos = false,
        bool clever = true,
        bool eval = true,
        int size_group = 1,
        bool transpose = false) {
    
    
    
    // U dimensions
    // int p = U.n_rows;
    // int K = U.n_cols;
    // int pK = int (p*K);
    
    // read in U and convert to spams::Matrix<double> alpha0
    // double* ptrU = new double[pK];
    // for (int r = 0; r < p; r++) {
    //     for (int c = 0; c < K; c++) {
    //         ptrU[c * p + r] = U(r, c);
    //     }
    // }
    //Matrix<double> alpha0(ptrU,p,K);
    Matrix<double> alpha0(const_cast<double*>(U.memptr()), p, K);
    
    
    
    // grp dimensions
    int gr = grp.n_rows;
    int gc = grp.n_cols;
    int grc = int (gr * gc);
    
    // read in grp and convert to spams::Matrix<bool> grp_dense
    // then to spams::SpMatrix<bool> groups
    bool* ptrG = new bool[grc];
    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            ptrG[c * gr + r] = ( grp(r, c) != 0.0 );
        }
    }
    Matrix<bool> grp_dense(ptrG, gr, gc);
    SpMatrix<bool> groups;
    grp_dense.toSparse(groups);
    
    
    
    // read in etaG and convert to spams::Vector<double> eta_g
    int n_etaG = etaG.length();
    double* ptrEG = new double[n_etaG];
    for (int i = 0; i < n_etaG; i++) {
        ptrEG[i] = etaG(i);
    }
    Vector<double> eta_g(ptrEG, n_etaG);
    
    
    
    // read in own_var and convert to spams::Vector<int> own_variables
    int n_ov = own_var.length();
    int* ptrOV = new int[n_ov];
    for (int i = 0; i < n_ov; i++) {
        ptrOV[i] = own_var(i);
    }
    Vector<int> own_variables(ptrOV, n_ov);
    
    
    
    // read in N_own_var and convert to spams::Vector<int> N_own_variables
    int n_Nov = N_own_var.length();
    int* ptrNOV = new int[n_Nov];
    for (int i = 0; i < n_Nov; i++) {
        ptrNOV[i] = N_own_var(i);
    }
    Vector<int> N_own_variables(ptrNOV, n_Nov);
    
    
    
    // read in regul and convert to char
    // int l = regul.length();
    // char* name_regul = new char[l];
    // for (int i = 0; i < l+1; i++) {
    //     name_regul[i] = regul[i];
    // }
    std::vector<char> name_regul_buf(regul.begin(), regul.end()); // Copy string content
    name_regul_buf.push_back('\0'); // Add the null terminator
    char* name_regul_ptr = name_regul_buf.data();
    
    
    
    // Initialize alpha - proximal operator
    Matrix<double> alpha(p,K);
    alpha.setZeros();
    
    
    
    // call _proximalTree
    _proximalTree(&alpha0, &alpha,
                   &eta_g, &groups,
                   &own_variables, &N_own_variables,
                   num_threads, lam1, lam2, lam3,
                   intercept, resetflow, name_regul_ptr,
                   verbose, pos, clever, eval,
                   size_group, transpose);
    
    
    
    // put updated alpha back into U
    for (int r = 0; r < p; r++) {
        for (int c = 0; c < K; c++) {
            U(r, c) = alpha[c * p + r];
        }
    }
    
    // free the dynamic memory
    // delete[] ptrU;
    delete[] ptrG;
    delete[] ptrEG;
    delete[] ptrOV;
    delete[] ptrNOV;
    // delete[] name_regul;
}




// [[Rcpp::export]]
double scalar_scad_prox(double val, double lambda, double a) {
    double abs_val = std::abs(val);
    
    // Standard SCAD requires a > 2. We use 3.7 as a common default if an invalid 'a' is given.
    if (a <= 2.0) {
        a = 3.7;
    }
    
    if (abs_val <= 2.0 * lambda) {
        // This region is equivalent to soft-thresholding (Lasso).
        return std::copysign(std::max(0.0, abs_val - lambda), val);
    } else if (abs_val <= a * lambda) {
        // This is the quadratic region of the SCAD penalty.
        return ((a - 1.0) * val - std::copysign(a * lambda, val)) / (a - 2.0);
    } else {
        // In this region, the penalty is constant, so the proximal operator is the identity.
        return val;
    }
}

// [[Rcpp::export]]
void proximalSCAD(
        arma::mat& U,
        double lam1,
        double a_scad = 3.7,
        bool pos = false) {
    
    // The 'a' parameter must be greater than 2 for the SCAD penalty to be well-defined.
    if (a_scad <= 2.0) {
        Rcpp::stop("The 'a' parameter for the SCAD penalty must be greater than 2.");
    }
    
    // Use Armadillo's .for_each() to apply the logic to each element efficiently.
    U.for_each([&](arma::mat::elem_type& val) {
        double z = val;
        double abs_z = std::abs(z);
        double new_val;
        
        if (abs_z <= 2.0 * lam1) {
            // This is the soft-thresholding part, identical to the Lasso proximal operator.
            double sign_z = (z > 0) - (z < 0); // Extracts the sign of z
            new_val = sign_z * std::max(0.0, abs_z - lam1);
        } else if (abs_z <= a_scad * lam1) {
            // This is the second case for the SCAD solution.
            double sign_z = (z > 0) - (z < 0);
            new_val = ((a_scad - 1.0) * z - sign_z * a_scad * lam1) / (a_scad - 2.0);
        } else {
            // For large values, the penalty derivative is zero, so the solution is the input value itself.
            new_val = z;
        }
        
        // Enforce the positivity constraint if the 'pos' flag is true.
        if (pos && new_val < 0) {
            val = 0.0;
        } else {
            val = new_val;
        }
    });
}

// void proximalSCAD2(
//         arma::mat& U,
//         int p,
//         int K,
//         const std::string& regul,
//         const Rcpp::IntegerVector& grp_id,
//         int ncores,
//         double lambda,
//         double a,
//         bool pos,
//         bool intercept,
//         int ref_cat) {
//     
//     // Determine the dimension that corresponds to features based on grp_id length.
//     // This implicitly handles the transposed vs. non-transposed cases.
//     int feature_dim_size = grp_id.size();
//     bool group_by_rows = (U.n_rows == feature_dim_size);
//     bool group_by_cols = (U.n_cols == feature_dim_size);
//     
//     if (!group_by_rows && !group_by_cols) {
//         Rcpp::stop("Matrix dimensions are inconsistent with grp_id length. Cannot determine feature dimension.");
//     }
//     
//     // Map group IDs to the indices of the features belonging to them.
//     std::map<int, std::vector<int>> groups;
//     for (int i = 0; i < feature_dim_size; ++i) {
//         groups[grp_id[i]].push_back(i);
//     }
//     
//     // Iterate over each group of features.
//     for (auto const& [group_id, indices] : groups) {
//         // Skip the intercept group if the flag is set.
//         if (intercept && group_id == 0) {
//             continue;
//         }
//         
//         if (indices.empty()) {
//             continue;
//         }
//         
//         if (group_by_rows) {
//             // NON-TRANSPOSED CASE: U is (features x tasks), group rows.
//             // Apply penalty to each task (column) vector for the current group.
//             for (int c = 0; c < U.n_cols; ++c) {
//                 if (c == ref_cat) continue; // Skip reference category.
//                 
//                 // Extract the coefficients for the current group and task.
//                 arma::vec group_coeffs(indices.size());
//                 for (size_t i = 0; i < indices.size(); ++i) {
//                     group_coeffs(i) = U(indices[i], c);
//                 }
//                 
//                 double norm_u_g = arma::norm(group_coeffs, 2);
//                 
//                 if (norm_u_g > 1e-10) {
//                     // Apply scalar SCAD prox to the L2 norm of the group vector.
//                     double prox_norm = scalar_scad_prox(norm_u_g, lambda, a);
//                     double shrinkage = prox_norm / norm_u_g;
//                     
//                     // Apply the shrinkage factor to all coefficients in the group.
//                     for (int idx : indices) {
//                         U(idx, c) *= shrinkage;
//                         if (pos) {
//                             U(idx, c) = std::max(0.0, U(idx, c));
//                         }
//                     }
//                 } else {
//                     // If norm is zero, all coefficients are zero.
//                     for (int idx : indices) {
//                         U(idx, c) = 0.0;
//                     }
//                 }
//             }
//         } else { // group_by_cols
//             // TRANSPOSED CASE: U is (tasks x features), group columns.
//             // Apply penalty to each task (row) vector for the current group.
//             for (int r = 0; r < U.n_rows; ++r) {
//                 if (r == ref_cat) continue; // Skip reference category.
//                 
//                 // Extract the coefficients for the current group and task.
//                 arma::vec group_coeffs(indices.size());
//                 for (size_t i = 0; i < indices.size(); ++i) {
//                     group_coeffs(i) = U(r, indices[i]);
//                 }
//                 
//                 double norm_u_g = arma::norm(group_coeffs, 2);
//                 
//                 if (norm_u_g > 1e-10) {
//                     // Apply scalar SCAD prox to the L2 norm of the group vector.
//                     double prox_norm = scalar_scad_prox(norm_u_g, lambda, a);
//                     double shrinkage = prox_norm / norm_u_g;
//                     
//                     // Apply the shrinkage factor to all coefficients in the group.
//                     for (int idx : indices) {
//                         U(r, idx) *= shrinkage;
//                         if (pos) {
//                             U(r, idx) = std::max(0.0, U(r, idx));
//                         }
//                     }
//                 } else {
//                     // If norm is zero, all coefficients are zero.
//                     for (int idx : indices) {
//                         U(r, idx) = 0.0;
//                     }
//                 }
//             }
//         }
//     }
// }

// void proximalSCAD(
//         arma::mat& U,
//         const int& p,
//         const int& K,
//         double lambda,
//         double a = 3.7,
//         bool intercept = false,
//         bool pos = false,
//         bool transpose = false) {
//     
//     if (lambda < 0) Rcpp::stop("lambda for SCAD must be non-negative.");
//     if (a <= 2.0) {
//         Rcpp::warning("Parameter 'a' for SCAD was <= 2.0; using default 3.7.");
//         a = 3.7;
//     }
//     
//     arma::uword n_rows = U.n_rows;
//     arma::uword n_cols = U.n_cols;
//     
//     if (transpose && (n_rows != K || n_cols != p))
//         Rcpp::stop("Matrix dimensions do not match K and p in transposed mode.");
//     if (!transpose && (n_rows != p || n_cols != K))
//         Rcpp::stop("Matrix dimensions do not match p and K in standard mode.");
//     
//     arma::uword row_start = (intercept && !transpose) ? 1 : 0;
//     arma::uword col_start = (intercept && transpose) ? 1 : 0;
//     
//     arma::uword row_end = n_rows;
//     arma::uword col_end = n_cols;
//     
//     for (arma::uword r = row_start; r < row_end; ++r) {
//         for (arma::uword c = col_start; c < col_end; ++c) {
//             
//             double& u_val = U(r, c);
//             double abs_u = std::abs(u_val);
//             double sign_u = (u_val > 0.0) - (u_val < 0.0);
//             double new_val = 0.0;
//             
//             if (abs_u <= lambda) {
//                 new_val = 0.0;
//             } else if (abs_u <= 2.0 * lambda) {
//                 new_val = sign_u * std::max(0.0, abs_u - lambda);
//             } else if (abs_u <= a * lambda) {
//                 new_val = ((a - 1.0) * u_val - sign_u * a * lambda) / (a - 2.0);
//             } else {
//                 new_val = u_val;
//             }
//             
//             if (pos && new_val < 0.0)
//                 u_val = 0.0;
//             else
//                 u_val = new_val;
//         }
//     }
// }





// [[Rcpp::export]]
arma::vec grad_ls_loss2(
        arma::rowvec& x,
        double& y,
        arma::vec& param,
        int& p) {
    arma::vec grad(p);
    grad =  arma::vectorise(x) * (arma::dot(x, param) - y);
    return grad;
}

// [[Rcpp::export]]
arma::vec grad_logistic_loss2(
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
void grad_multinom_loss2(
        const arma::rowvec& x,
        int y,
        int K,
        double offset,
        const arma::mat& param,
        int p,
        arma::mat& grad_out) {
    
    // 1. Calculate the linear score for each of the K classes.
    arma::vec linear_scores(K);
    for (int i = 0; i < K; i++) {
        linear_scores(i) = arma::dot(x, param.col(i)) + offset;
    }
    
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
Rcpp::List MultinomLogistic2(
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
        int ncores,
        bool verbose = false,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
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
    // arma::mat param(p, K, arma::fill::zeros);
    arma::mat param(p, K); 
    
    if (param_start.isNotNull()) {
        Rcpp::NumericMatrix param_start_mat(param_start); 
        
        // Check dimensions of the provided initial matrix
        if (param_start_mat.nrow() != p || param_start_mat.ncol() != K) {
            Rcpp::stop("Dimensions of provided 'param_start' matrix must be [X.n_cols x K].");
        }
        
        param = Rcpp::as<arma::mat>(param_start_mat);
        
    } else {
        param.zeros();
    }
    arma::mat param_old(p, K);
    
    arma::mat param_reg(reg_p, K);   // regularized coefficients
    arma::mat param_t(K, reg_p);      // regularized coefficients
    
    // convergence related
    double diff;
    int counter_outer = 0;
    
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
            
            // x_sample = X.row(index);
            const auto& x_sample = X.row(index);
            y_sample = Y(index);
            o_sample = offset(index);
            
            // temp1 = grad_multinom_loss2(x_sample, y_sample, K, o_sample, param, p);
            // temp2 = grad_multinom_loss2(x_sample, y_sample, K, o_sample, param_old, p);
            grad_multinom_loss2(x_sample, y_sample, K, o_sample, param, p, temp1);
            grad_multinom_loss2(x_sample, y_sample, K, o_sample, param_old, p, temp2);
            
            param = param - learning_rate * (temp1 - temp2 + grad);
            
            // extract only variables involved in the penalization
            param_reg = param.head_rows(reg_p);
            
            // call proximal function
            if (transpose) {
                param_t = param_reg.t();
                
                if (penalty == 1) {
                    // proximalFlat2(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                    arma::mat param_reg_copy = param_reg; // Explicit copy
                    proximalFlat2_nospams(param_reg_copy, reg_p, K, regul, grp_id,
                                          lam1 * learning_rate,
                                          lam2 * learning_rate,
                                          lam3 * learning_rate);
                    param_reg = param_reg_copy; 
                }
                
                if (penalty == 2) {
                    proximalGraph2(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree2(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                param_reg = param_t.t();
            } else {
                if (penalty == 1) {
                    proximalFlat2(param_reg, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                }
                
                if (penalty == 2) {
                    proximalGraph2(param_reg, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree2(param_reg, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
            }
            
            param.head_rows(reg_p) = param_reg;
        }
        
        counter_outer += 1;
        
        diff = arma::norm(param - param_old, "fro");
        diff = diff/(p*K);
        if (verbose) {
            Rcpp::Rcout << "\n Iteration " << counter_outer << "\n";
            Rcpp::Rcout << "Frobenius norm of coefficient update \n" << diff <<"\n";
        }
        
        
        if (diff < tolerance || counter_outer>= maxit) {
            break;
        }
    }
    
    arma::sp_mat param_sp(param);
    
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("Estimates")                     = param,
                                           Rcpp::Named("Sparse Estimates")              = param_sp);
    return result;
}






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
                    proximalFlat2(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                }
                
                if (penalty == 2) {
                    proximalGraph2(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree2(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                param_reg = param_t.t();
            } else {
                if (penalty == 1) {
                    proximalFlat2(param_reg, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                }
                
                if (penalty == 2) {
                    proximalGraph2(param_reg, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                
                if (penalty == 3) {
                    proximalTree2(param_reg, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
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


// The apply_proximal_step function remains unchanged from your original implementation
void apply_proximal_step(
        arma::mat& param, const arma::mat& param_unprox, double learning_rate,
        int reg_p, int K, int p, bool transpose, int penalty, const std::string& regul,
        Rcpp::IntegerVector& grp_id, int ncores, double lam1, double lam2, double lam3, bool pos,
        const arma::mat& grp, const arma::mat& grpV, const Rcpp::NumericVector& etaG,
        Rcpp::IntegerVector& own_var, Rcpp::IntegerVector& N_own_var
) {
    auto param_unprox_reg_view = param_unprox.head_rows(reg_p);
    double scaled_lam1 = lam1 * learning_rate;
    double scaled_lam2 = lam2 * learning_rate;
    double scaled_lam3 = lam3 * learning_rate;
    
    if (transpose) {
        arma::mat param_t = param_unprox_reg_view.t();
        if (penalty == 1) proximalFlat2(param_t, K, reg_p, regul, grp_id, ncores, scaled_lam1, scaled_lam2, scaled_lam3, pos);
        else if (penalty == 2) proximalGraph2(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 3) proximalTree2(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 4) proximalSCAD(param_t, scaled_lam1, lam2, pos);
        param.head_rows(reg_p) = param_t.t();
    } else {
        arma::mat param_unprox_reg_copy = param_unprox_reg_view;
        if (penalty == 1) proximalFlat2(param_unprox_reg_copy, reg_p, K, regul, grp_id, ncores, scaled_lam1, scaled_lam2, scaled_lam3, pos);
        else if (penalty == 2) proximalGraph2(param_unprox_reg_copy, reg_p, K, regul, grp, grpV, etaG, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 3) proximalTree2(param_unprox_reg_copy, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, scaled_lam1, scaled_lam2, pos);
        else if (penalty == 4) proximalSCAD(param_unprox_reg_copy, scaled_lam1, lam2, pos);
        param.head_rows(reg_p) = param_unprox_reg_copy;
    }
    if (reg_p < p) {
        param.tail_rows(p - reg_p) = param_unprox.tail_rows(p - reg_p);
    }
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
        double learning_rate,
        double momentum_gamma,
        double tolerance,
        int niter_inner,
        int maxit,
        int ncores = 4,
        bool pos = false,
        bool verbose = false,
        bool add_intercept = true,
        Rcpp::Nullable<Rcpp::NumericMatrix> param_start = R_NilValue) {
    
    int p_orig = X.n_cols;
    int n = X.n_rows;
    arma::mat X_eff;
    int p;
    
    if (add_intercept) {
        p = p_orig + 1;
        X_eff.set_size(n, p);
        if (p_orig > 0) X_eff.head_cols(p_orig) = X;
        X_eff.col(p - 1).ones();
    } else {
        p = p_orig;
        X_eff = X;
    }
    
    // --- Determine the learning rate ---
    double final_learning_rate;
    if (learning_rate <= 0.0) {
        if (verbose) {
            Rcpp::Rcout << "Learning rate not specified. Detecting automatically..." << std::endl;
        }
        double lambda_max = get_lambda_max(X_eff);
        double L = lambda_max / static_cast<double>(n);
        
        if (L > 1e-8) {
            const double safety_factor = 0.5;
            final_learning_rate = safety_factor / L;
        } else {
            final_learning_rate = 1.0;
        }
        if (verbose) {
            Rcpp::Rcout << "  - Lipschitz Constant (L): " << L << std::endl;
            Rcpp::Rcout << "  - Setting target learning rate to: " << final_learning_rate << std::endl;
        }
    } else {
        final_learning_rate = learning_rate;
    }
    
    arma::mat param(p, K);
    if (param_start.isNotNull()) param = Rcpp::as<arma::mat>(Rcpp::NumericMatrix(param_start));
    else param.zeros();
    
    arma::mat grad(p, K), temp1(p, K), temp2(p, K);
    arma::mat param_old(p, K), param_unprox(p, K);
    arma::mat velocity(p, K, arma::fill::zeros);
    
    double diff;
    int counter_outer = 0;
    
    // Use a single, unified duration for all ramp-up schedules
    const int ramp_up_duration = 100;
    
    while (true) {
        param_old = param;
        counter_outer += 1;
        
        // --- Dynamic Inner Iteration Schedule ---
        int current_niter_inner;
        if (counter_outer < ramp_up_duration) {
            double start_iter = 0.1 * n;
            double end_iter = static_cast<double>(niter_inner);
            if (start_iter > end_iter) start_iter = end_iter;
            double ramp_fraction = (ramp_up_duration > 1) ? (static_cast<double>(counter_outer - 1) / (ramp_up_duration - 1)) : 1.0;
            current_niter_inner = static_cast<int>(std::ceil(start_iter + (end_iter - start_iter) * ramp_fraction));
        } else {
            current_niter_inner = niter_inner;
        }
        
        // --- Dynamic Learning Rate Schedule (Warm-up) ---
        double current_learning_rate;
        if (counter_outer < ramp_up_duration) {
            double start_lr = 0.01 * final_learning_rate; 
            double ramp_fraction = (ramp_up_duration > 1) ? (static_cast<double>(counter_outer - 1) / (ramp_up_duration - 1)) : 1.0;
            current_learning_rate = start_lr + (final_learning_rate - start_lr) * ramp_fraction;
        } else {
            current_learning_rate = final_learning_rate;
        }
        
        grad.zeros();
        
        // --- PERFORMANCE: This loop is the main bottleneck and can be parallelized. ---
#pragma omp parallel for reduction(+:grad) num_threads(ncores)
        for (int i = 0; i < n; i++) {
            arma::mat single_grad(p, K); // Thread-private gradient matrix
            grad_multinom_loss2(X_eff.row(i), Y(i), K, offset(i), param_old, p, single_grad);
            grad += single_grad; // OMP reduction handles this safely
        }
        if (n > 0) grad /= n;
        
        param = param_old;
        velocity.zeros();
        
        for (int i = 0; i < current_niter_inner; ++i) {
            int index = arma::randi(arma::distr_param(0, n - 1));
            grad_multinom_loss2(X_eff.row(index), Y(index), K, offset(index), param, p, temp1);
            grad_multinom_loss2(X_eff.row(index), Y(index), K, offset(index), param_old, p, temp2);
            arma::mat svr_grad_current = temp1 - temp2 + grad;
            
            velocity = momentum_gamma * velocity - current_learning_rate * svr_grad_current;
            param_unprox = param + velocity;
            
            apply_proximal_step(param, param_unprox, current_learning_rate, reg_p, K, p, transpose, penalty, regul,
                                grp_id, ncores, lam1, lam2, lam3, pos, grp, grpV, etaG, own_var, N_own_var);
        }
        
        diff = arma::norm(param - param_old, "fro") / (arma::norm(param_old, "fro") + 1e-10);
        
        if (verbose) {
            Rcpp::Rcout << "Iteration " << counter_outer << " | LR: " << current_learning_rate << " | Inner Iter: " << current_niter_inner << " | Rel.Chg: " << diff << std::endl;
        }
        
        if (diff < tolerance || counter_outer >= maxit) break;
    }
    
    Rcpp::List result;
    if (add_intercept) {
        arma::mat beta = param.head_rows(p_orig);
        arma::rowvec intercepts = param.row(p_orig);
        arma::sp_mat beta_sp(beta);
        result = Rcpp::List::create(Rcpp::Named("Estimates") = Rcpp::wrap(beta), Rcpp::Named("Intercepts") = intercepts, Rcpp::Named("Sparse Estimates") = beta_sp);
    } else {
        arma::sp_mat beta_sp(param);
        result = Rcpp::List::create(Rcpp::Named("Estimates") = Rcpp::wrap(param), Rcpp::Named("Intercepts") = R_NilValue, Rcpp::Named("Sparse Estimates") = beta_sp);
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
//         double learning_rate,
//         double momentum_gamma,
//         double tolerance,
//         int niter_inner,
//         int maxit,
//         int ncores,
//         bool pos,
//         bool verbose,
//         Rcpp::Nullable<Rcpp::NumericMatrix> param_start) {
//     
//     int p = X.n_cols;
//     int n = X.n_rows;
//     
//     int index;
//     int y_sample_int;
//     double o_sample;
//     
//     arma::mat grad(p, K);
//     arma::mat temp1(p, K);
//     arma::mat temp2(p, K);
//     
//     arma::mat param(p, K); 
//     
//     if (param_start.isNotNull()) {
//         Rcpp::NumericMatrix param_start_mat(param_start); 
//         if (param_start_mat.nrow() != p || param_start_mat.ncol() != K) {
//             Rcpp::stop("Dimensions of provided 'param_start' matrix must be [X.n_cols x K].");
//         }
//         param = Rcpp::as<arma::mat>(param_start_mat);
//     } else {
//         param.zeros();
//     }
//     
//     arma::mat param_old(p, K);
//     
//     const double beta = 0.5;
//     const int max_line_search_iter = 100;
//     
//     double diff;
//     int counter_outer = 0;
//     
//     while (true) {
//         param_old = param; 
//         arma::mat single_grad(p, K);
//         grad.zeros();
//         
//         for (int i = 0; i < n; i++) {
//             const auto& x_sample_view = X.row(i);
//             const int& y_val = Y(i); 
//             const double& o_val = offset(i);
//             grad_multinom_loss2(x_sample_view, y_val, K, o_val, param_old, p, single_grad);
//             grad += single_grad;
//         }
//         grad /= n;
//         
//         for (int i = 0; i < niter_inner; ++i) {
//             if (n == 0) continue;
//             index = arma::randi(arma::distr_param(0, n - 1));
//             const auto& x_sample_view = X.row(index);
//             y_sample_int = static_cast<int>(Y(index)); 
//             o_sample = offset(index);
//             
//             grad_multinom_loss2(x_sample_view, y_sample_int, K, o_sample, param, p, temp1);
//             grad_multinom_loss2(x_sample_view, y_sample_int, K, o_sample, param_old, p, temp2);
//             arma::mat svr_grad_current = temp1 - temp2 + grad;
//             
//             double current_lr = learning_rate;
//             double loss_current = eval_multinom_loss(X, Y, offset, K, param, p, n);
//             
//             for (int ls_iter = 0; ls_iter < max_line_search_iter; ++ls_iter) {
//                 arma::mat param_unprox = param - current_lr * svr_grad_current;
//                 arma::mat param_candidate = param;
//                 auto param_unprox_reg_view = param_unprox.head_rows(reg_p);
//                 
//                 if (transpose) {
//                     arma::mat param_t = param_unprox_reg_view.t();
//                     if (penalty == 1) {
//                         proximalFlat2(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
//                     } else if (penalty == 2) {
//                         proximalGraph2(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
//                     } else if (penalty == 3) {
//                         proximalTree2(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
//                     } else if (penalty == 4) { // SCAD
//                         
//                         proximalSCAD(
//                             param_t,
//                             lam1 * learning_rate, // The lambda parameter
//                             lam2,                 // The 'a' parameter (e.g., 3.7)
//                             pos
//                         );
//                     }
//                     param_candidate.head_rows(reg_p) = param_t.t();
//                 } else { 
//                     arma::mat param_unprox_reg_copy = param_unprox_reg_view;
//                     if (penalty == 1) {
//                         
//                         proximalFlat2(param_unprox_reg_copy, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
//                     } else if (penalty == 2) {
//                         proximalGraph2(param_unprox_reg_copy, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
//                     } else if (penalty == 3) {
//                         proximalTree2(param_unprox_reg_copy, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
//                     }else if (penalty == 4) { // SCAD
//                         proximalSCAD(
//                             param_unprox_reg_copy,
//                             lam1 * learning_rate, // The lambda parameter
//                             lam2,                 // The 'a' parameter (e.g., 3.7)
//                             pos
//                         );
//                     }
//                     param_candidate.head_rows(reg_p) = param_unprox_reg_copy;
//                 }
//                 if (reg_p < p) {
//                     param_candidate.tail_rows(p - reg_p) = param_unprox.tail_rows(p - reg_p);
//                 }
//                 
//                 double loss_candidate = eval_multinom_loss(X, Y, offset, K, param_candidate, p, n);
//                 double quadratic_approx_rhs = arma::dot(svr_grad_current, param_candidate - param);
//                 
//                 if (loss_candidate <= loss_current - quadratic_approx_rhs + (0.5 / current_lr) * arma::accu(arma::pow(param_candidate - param, 2))) {
//                     param = param_candidate;
//                     break; 
//                 }
//                 
//                 current_lr *= beta;
//                 
//                 if (ls_iter == max_line_search_iter - 1) {
//                     break;
//                 }
//             }
//         }
//         
//         counter_outer += 1;
//         niter_inner = static_cast<int>(ceil(static_cast<double>(niter_inner) * 1.25));
//         
//         double norm_old = arma::norm(param_old, "fro");
//         diff = arma::norm(param - param_old, "fro") / (norm_old + 1e-10);
//         
//         if (verbose) {
//             Rcpp::Rcout << "\nIteration " << counter_outer << ", Rel. Change: " << diff << "\n";
//         }
//         
//         if (diff < tolerance || counter_outer >= maxit) {
//             break;
//         }
//     } 
//     
//     arma::sp_mat param_sp(param);
//     Rcpp::List result = Rcpp::List::create(
//         Rcpp::Named("Estimates") = param,
//         Rcpp::Named("Sparse Estimates") = param_sp
//     );
//     return result;
// }



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
                    proximalFlat2(param_t, K, reg_p, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate);
                } else if (penalty == 2) {
                    proximalGraph2(param_t, K, reg_p, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                } else if (penalty == 3) {
                    proximalTree2(param_t, K, reg_p, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
                }
                param.head_rows(reg_p) = param_t.t(); 
            } else { 
                arma::mat param_reg_copy = param_reg_view; // Explicit copy
                if (penalty == 1) {
                    proximalFlat2(param_reg_copy, reg_p, K, regul, grp_id, ncores, lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate );
                } else if (penalty == 2) {
                    proximalGraph2(param_reg_copy, reg_p, K, regul, grp, grpV, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate);
                } else if (penalty == 3) {
                    proximalTree2(param_reg_copy, reg_p, K, regul, grp, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate);
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
        Rcpp::warning("This PCD implementation primarily supports penalties handled by proximalFlat2. Graph/Tree penalties might not behave as expected without specialized PCD updates.");
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
                        proximalFlat2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, current_grp_id_for_prox, ncores,
                                      lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
                    } else if (penalty_code == 2) {
                        proximalGraph2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, grpV_mat, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    } else if (penalty_code == 3) {
                        proximalTree2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    }
                    param.row(j_feat) = param_j_block_for_prox_op.t();
                } else {
                    param_j_block_for_prox_op = Bj_unprox; // 1 x K_classes matrix
                    n_rows_prox = 1;         // Dimension of param_j_block_for_prox_op
                    n_cols_prox = K_classes; // Dimension of param_j_block_for_prox_op
                    
                    if (penalty_code == 1) {
                        proximalFlat2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, current_grp_id_for_prox, ncores,
                                      lam1 * learning_rate, lam2 * learning_rate, lam3 * learning_rate, pos);
                    } else if (penalty_code == 2) {
                        proximalGraph2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, grpV_mat, etaG, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
                    } else if (penalty_code == 3) {
                        proximalTree2(param_j_block_for_prox_op, n_rows_prox, n_cols_prox, regul, grp_mat, etaG, own_var, N_own_var, ncores, lam1 * learning_rate, lam2 * learning_rate, pos);
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
    // Rcpp::Rcout << "Warning: proximalFlat2 called but using placeholder logic." << std::endl; // Placeholder
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
//                     // else if(penalty_code==2) proximalGraph2(param_j_block_for_prox_op,nr,nc,regul,grp_mat,grpV_mat,etaG,ncores,eff_lam1,eff_lam2,pos);
//                     // else if(penalty_code==3) proximalTree2(param_j_block_for_prox_op,nr,nc,regul,grp_mat,etaG,own_var,N_own_var,ncores,eff_lam1,eff_lam2,pos);
//                     param.row(j_feat) = param_j_block_for_prox_op.t();
//                 } else { 
//                     param_j_block_for_prox_op = Bj_unprox; int nr=1, nc=K_classes;
//                     if(penalty_code==1) proximalFlat3(param_j_block_for_prox_op,nr,nc,regul,current_grp_id_for_prox,ncores,eff_lam1,eff_lam2,eff_lam3,pos);
//                     // else if(penalty_code==2) proximalGraph2(param_j_block_for_prox_op,nr,nc,regul,grp_mat,grpV_mat,etaG,ncores,eff_lam1,eff_lam2,pos);
//                     // else if(penalty_code==3) proximalTree2(param_j_block_for_prox_op,nr,nc,regul,grp_mat,etaG,own_var,N_own_var,ncores,eff_lam1,eff_lam2,pos);
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

//' @title Asymmetric Multinomial Logistic Regression (Corrected)
 //' @description Fits a penalized multinomial logistic regression model using a reference-class
 //' parameterization. The optimization follows the core algorithm of glmnet: Iteratively
 //' Reweighted Least Squares (IRLS) combined with Cyclical Coordinate Descent (CCD).
 //'
 //' @param X Predictor matrix.
 //' @param Y Response vector of class labels (integers from 0 to K-1, where 0 is the reference class).
 //' @param offset_vec A vector of offsets to be included in the linear predictor.
 //' @param K_classes The total number of classes (including the reference class).
 //' @param reg_p The number of predictors to be regularized.
 //' @param lam1 The L1 regularization parameter (lambda).
 //' @param lam2 The L2 regularization parameter (alpha).
 //' @param tolerance Convergence tolerance for the outer loop.
 //' @param kkt_abs_check_tol Absolute tolerance for the KKT check.
 //' @param maxit Maximum number of outer IRLS iterations.
 //' @param max_ccd_passes_active_set Maximum number of CCD passes over the active set per IRLS iteration.
 //' @param pos Logical flag for positivity constraints on coefficients.
 //' @return A list containing the estimated coefficients.
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