library(mvtnorm)
library(boot)
library(truncnorm)
library(Matrix)
#library(tictoc)
library(readr)
library(dplyr)

datagen <- function(n) {
  lambda1 = 0
  lambda2 = 0.6
  gamma1 = 0.6
  gamma2 = 0.4
  
  W_1 <- rbeta(n, 2, 3)
  W_2 <- rbeta(n, 2, 3)
  W_3 <- rbeta(n, 2, 3)
  A <- rbinom(n, 1, plogis(0.5 * W_1 + 0.5 * W_2 - 1))
  Z_1 <- rtruncnorm(n, a = -1, b = 1, mean = -0.4 + 0 * A + 0.2 * (W_3) ** 2)
  Z_2 <- rtruncnorm(n, a = -1, b = 1, mean = 0.2 - 0 * A + 0.5 * sin(W_2))
  M_1 <- rtruncnorm(n, a = -1, b = 1, mean = -0.5 + lambda1 * Z_1 + lambda2 * A + 0.4 * W_2 + 0.2 * W_3)
  M_2 <- rtruncnorm(n, a = -1, b = 1, mean = -0.5 + lambda1 * Z_2 + lambda2 * A + 0.4 * W_1 + 0.2 * W_3)
  Y <- rnorm(n, mean = 0.2 * M_1 + 0.2 * M_2 + gamma1 * Z_1 / 2 + gamma1 * Z_2 / 2 + gamma2 * A - 0.5 * cos(W_1) - 1.5)
  data <- data.frame(W_1, W_2, W_3, A, Z_1, Z_2, M_1, M_2, Y)
  return(data)
}

# data0 <- datagen(5e6)
# 
# lambda1 = 0.4
# lambda2 = 0.6
# gamma1 = 0.6
# gamma2 = 0.4
# 
# a_1 = 0
# a_2 = 1
# a_3 = 1
# a_4 = 1
# 
# data0 <- data0 %>%
#   mutate(prob_A = case_when(a_1 == 0 ~ 1 - plogis(0.5 * W_1 + 0.5 * W_2 - 1),
#                             TRUE ~ plogis(0.5 * W_1 + 0.5 * W_2 - 1))) %>%
#   mutate(alpha_part1 = as.numeric(A == a_1) / prob_A) %>%
#   mutate(prob_M_denom_1 = dtruncnorm(M_1, a = -1, b = 1, mean = -0.5 + lambda1 * Z_1 + lambda2 * A + 0.4 * W_2 + 0.2 * W_3)) %>%
#   mutate(prob_M_num_1 = dtruncnorm(M_1, a = -1, b = 1, mean = -0.5 + lambda1 * (-0.4 + 0.5 * a_4 + 0.5 * (W_3) ** 2) + lambda2 * a_3 + 0.4 * W_2 + 0.2 * W_3), sd = sqrt(lambda1 ** 2 + 1)) %>%
#   mutate(prob_M_denom_2 = dtruncnorm(M_2, a = -1, b = 1, mean = -0.5 + lambda1 * Z_2 + lambda2 * A + 0.4 * W_1 + 0.2 * W_3)) %>%
#   mutate(prob_M_num_2 = dtruncnorm(M_2, a = -1, b = 1, mean = -0.5 + lambda1 * (0.2 - 0.5 * a_4 + 0.5 * sin(W_2)) + lambda2 * a_3 + 0.4 * W_1 + 0.2 * W_3), sd = sqrt(lambda1 ** 2 + 1)) %>%
#   mutate(alpha_part2 = prob_M_num_1 * prob_M_num_2 / (prob_M_denom_1 * prob_M_denom_2)) %>% 
#   mutate(prob_Z_num_1 = dtruncnorm(Z_1, a = -1, b = 1, mean = -0.4 + 0.5 * a_2 + 0.5 * (W_3) ** 2)) %>%
#   mutate(prob_Z_denom_1 = dtruncnorm(Z_1, a = -1, b = 1, mean = -0.4 + 0.5 * A + 0.5 * (W_3) ** 2)) %>%
#   mutate(prob_Z_num_2 = dtruncnorm(Z_2, a = -1, b = 1, mean = 0.2 - 0.5 * a_2 + 0.5 * sin(W_2))) %>%
#   mutate(prob_Z_denom_2 = dtruncnorm(Z_2, a = -1, b = 1, mean = 0.2 - 0.5 * A + 0.5 * sin(W_2))) %>%
#   mutate(alpha_part3 = prob_Z_num_1 * prob_Z_num_2 / (prob_Z_denom_1 * prob_Z_denom_2)) %>% 
#   mutate(true_alpha3 = alpha_part1 * alpha_part2 * alpha_part3)
# 
# mean(data0$true_alpha3 * data0$Y)

n <- 2000

label_id <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))

print(label_id)

data_simu <- datagen(n)
# 
# data_all_supp_Z <- data.frame(W_1 = NULL, W_2 = NULL, W_3 = NULL, A = NULL, Z_1 = NULL, Z_2 = NULL, 
#                               M_1 = NULL, M_2 = NULL, Y = NULL, label = NULL)
# data_all_supp_M <- data.frame(W_1 = NULL, W_2 = NULL, W_3 = NULL, A = NULL, Z_1 = NULL, Z_2 = NULL, 
#                               M_1 = NULL, M_2 = NULL, Y = NULL, label = NULL)
# 
# 
# #W_1 = simulation_test_one_batch$W_1
# W_1 = data_simu$W_1
# W_2 = data_simu$W_2
# W_3 = data_simu$W_3
# #W_3 = simulation_test_one_batch$W_3
# A = data_simu$A
# Z_1 = data_simu$Z_1
# Z_2 = data_simu$Z_2
# M_1 = data_simu$M_1
# M_2 = data_simu$M_2
# Y = data_simu$Y
# 
# D <- dist(cbind(W_1, W_2, W_3, A))
# D <- D / max(D)
# d  <- as.vector(t(as.matrix(D)))
# 
# #tic()
# rows <- c(as.numeric(gl(n, n, n^2)), as.numeric(gl(n, n, n^2)) + n)
# cols <- c(1:(n^2), unlist(lapply(0:(n-2), function(j) j + seq(1, n^2, n))), (0:(n-1)*(n+1) + 1))
# A <- sparseMatrix(i = rows, j = cols, x = 1)
# b <-  Matrix(sparseVector(i = 1:(2*n - 1), x = 1, length = 2*n), ncol = 1)
# 
# ## sol <- simplex(d, A3 = t(A), b3 = b)
# ## sol <- solveLP(d, bvec = b, Amat = A, const.dir = rep('=', length(b)), lpSolve = TRUE)
# 
# library(Rglpk)
# sol <- Rglpk_solve_LP(d, A, dir = rep("==", nrow(b)), rhs = b)
# #toc()
# 
# P <- matrix(sol$solution, n, n)
# Z_1 <- P %*% Z_1
# Z_2 <- P %*% Z_2
# 
# W_1 = data_simu$W_1
# W_2 = data_simu$W_2
# W_3 = data_simu$W_3
# #W_3 = simulation_test_one_batch$W_3
# A = data_simu$A
# M_1 = data_simu$M_1
# M_2 = data_simu$M_2
# Y = data_simu$Y
# data_simu_supp_Z <- data.frame(W_1 = W_1, W_2 = W_2, W_3 = W_3, A = A, Z_1 = Z_1,
#                                Z_2 = Z_2, M_1 = M_1, M_2 = M_2, Y = Y, label = as.integer(label_id))
# 
# M_1 <- P %*% M_1
# M_2 <- P %*% M_2
# 
# Z_1 = data_simu$Z_1
# Z_2 = data_simu$Z_2
# #W_3 = simulation_test_one_batch$W_3
# A = data_simu$A
# Y = data_simu$Y
# 
# data_simu_supp_M <- data.frame(W_1 = W_1, W_2 = W_2, W_3 = W_3, A = A, Z_1 = Z_1,
#                                Z_2 = Z_2, M_1 = M_1, M_2 = M_2, Y = Y, label = as.integer(label_id))

write_csv(data_simu, file = paste0("~/Projects/RieszLearning/make_data_nonnull_NDENIE_", n, "/", "simulation_test_all_batches_", n, "_cont_multi_Z_M_Y_recantingtwins_array_", as.integer(label_id), ".csv"))
# write_csv(data_simu_supp_Z, file = paste0("~/Projects/RieszLearning/make_data_nonnull_NDENIE_", n, "/", "simulation_test_all_batches_", n, "_cont_multi_Z_M_Y_recantingtwins_supp_Z_array_", as.integer(label_id), ".csv"))
# write_csv(data_simu_supp_M, file = paste0("~/Projects/RieszLearning/make_data_nonnull_NDENIE_", n, "/", "simulation_test_all_batches_", n, "_cont_multi_Z_M_Y_recantingtwins_supp_M_array_", as.integer(label_id), ".csv"))
# 
