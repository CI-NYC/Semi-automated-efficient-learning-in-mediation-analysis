# Num: 3
# Path 3 in the recanting twins
# Contr of (0, 0, 1, 1) and (0, 0, 1, 0)

library(mlr3extralearners)
library(mlr3superlearner)
library(data.table)
library(tidyverse)
library(Rglpk)

label = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
N = 2000

filename = paste0("~/Projects/RieszLearning/make_data_nonnull_", N, "/simulation_test_all_batches_", N, "_cont_multi_Z_M_Y_recantingtwins_supp_Z_array_", label, ".csv")
combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10_supp_Z <- read_csv(filename)

filename = paste0("~/Projects/RieszLearning/results_0010/N_", N, "_seq_", label, ".csv")
combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 <- read_csv(filename)

attr_models = c("lightgbm", "mean", "earth", "nnet")

a_1 <- 0
a_2 <- 0
a_3 <- 1
a_4 <- 0


batch_1 <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10
batch_1_supp_Z <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10_supp_Z 

fit_4 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3", "Y"), with = F],
                          target = "Y",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_4 <- predict(fit_4, batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_1 <- batch_1 %>%
  mutate(A = a_1)

batch_1_a_2 <- batch_1 %>%
  mutate(A = a_2)

batch_1_a_3 <- batch_1 %>%
  mutate(A = a_3)

batch_1_a_4 <- batch_1 %>%
  mutate(A = a_4)

# Fit theta_3_Z and b_3_Z

batch_1_a_1_supp_Z <- batch_1_a_1 %>%
  mutate(Z_1 = batch_1_supp_Z$Z_1) %>% 
  mutate(Z_2 = batch_1_supp_Z$Z_2)

b_4 <- predict(fit_4, batch_1_a_1_supp_Z[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_4_Z <- predict(fit_4, batch_1_a_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_supp_Z$b_4 = b_4

fit_3_Z <- mlr3superlearner(data = batch_1_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3", "b_4"), with = F],
                            target = "b_4",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)


theta_3_Z <- predict(fit_3_Z, batch_1_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_2_supp_Z <- batch_1_a_2 %>%
  mutate(Z_1 = batch_1_supp_Z$Z_1) %>% 
  mutate(Z_2 = batch_1_supp_Z$Z_2)

b_3_Z <- predict(fit_3_Z, batch_1_a_2_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Fit theta_2_Z and b_2_Z
batch_1$b_3_Z <- b_3_Z

fit_2_Z <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3", "b_3_Z"), with = F],
                            target = "b_3_Z",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)

theta_2_Z <- predict(fit_2_Z, batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_2_Z <- predict(fit_2_Z, batch_1_a_3[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Fit theta_1_Z and b_1_Z

batch_1$b_2_Z <- b_2_Z

fit_1_Z <- mlr3superlearner(data = batch_1[, c("A", "W_1", "W_2", "W_3", "b_2_Z"), with = F],
                            target = "b_2_Z",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)

theta_1_Z <- predict(fit_1_Z, batch_1[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_1_Z <- predict(fit_1_Z, batch_1_a_4[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Summarize
Y <- batch_1$Y
alpha_4 <- batch_1$alpha6
alpha_1_Z <- batch_1$alpha1
alpha_2_Z <- batch_1$alpha3
alpha_3_Z <- batch_1$alpha5

esti1 <- alpha_4 * (Y - theta_4) +
  alpha_3_Z * (b_4_Z - theta_3_Z) + 
  alpha_2_Z * (b_3_Z - theta_2_Z) + 
  alpha_1_Z * (b_2_Z - theta_1_Z) + 
  b_1_Z

filename = paste0("~/Projects/RieszLearning/results_0011/N_", N, "_seq_", label, ".csv")
combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 <- read_csv(filename)

a_1 <- 0
a_2 <- 0
a_3 <- 1
a_4 <- 1

attr_models = c("lightgbm", "mean", "earth", "nnet")

batch_1 <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 
batch_1_supp_Z <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10_supp_Z

fit_4 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3", "Y"), with = F],
                          target = "Y",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_4 <- predict(fit_4, batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_1 <- batch_1 %>%
  mutate(A = a_1)

batch_1_a_2 <- batch_1 %>%
  mutate(A = a_2)

batch_1_a_3 <- batch_1 %>%
  mutate(A = a_3)

batch_1_a_4 <- batch_1 %>%
  mutate(A = a_4)

# Fit theta_3_Z and b_3_Z

batch_1_a_1_supp_Z <- batch_1_a_1 %>%
  mutate(Z_1 = batch_1_supp_Z$Z_1) %>% 
  mutate(Z_2 = batch_1_supp_Z$Z_2)

b_4 <- predict(fit_4, batch_1_a_1_supp_Z[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_4_Z <- b_4

batch_1_supp_Z$b_4 = b_4

fit_3_Z <- mlr3superlearner(data = batch_1_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3", "b_4"), with = F],
                            target = "b_4",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)


theta_3_Z <- predict(fit_3_Z, batch_1_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_2_supp_Z <- batch_1_a_2 %>%
  mutate(Z_1 = batch_1_supp_Z$Z_1) %>% 
  mutate(Z_2 = batch_1_supp_Z$Z_2)

b_3_Z <- predict(fit_3_Z, batch_1_a_2_supp_Z[, c("A", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Fit theta_2_Z and b_2_Z
batch_1$b_3_Z <- b_3_Z

fit_2_Z <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3", "b_3_Z"), with = F],
                            target = "b_3_Z",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)

theta_2_Z <- predict(fit_2_Z, batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_2_Z <- predict(fit_2_Z, batch_1_a_3[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)


# Fit theta_1_Z and b_1_Z

batch_1$b_2_Z <- b_2_Z

fit_1_Z <- mlr3superlearner(data = batch_1[, c("A", "W_1", "W_2", "W_3", "b_2_Z"), with = F],
                            target = "b_2_Z",
                            library = attr_models,
                            outcome_type = "continuous",
                            folds = 10)

theta_1_Z <- predict(fit_1_Z, batch_1[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_1_Z <- predict(fit_1_Z, batch_1_a_4[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Summarize
Y <- batch_1$Y
alpha_4 <- batch_1$alpha6
alpha_1_Z <- batch_1$alpha1
alpha_2_Z <- batch_1$alpha3
alpha_3_Z <- batch_1$alpha5

esti2 <- alpha_4 * (Y - theta_4) +
  alpha_3_Z * (b_4_Z - theta_3_Z) + 
  alpha_2_Z * (b_3_Z - theta_2_Z) + 
  alpha_1_Z * (b_2_Z - theta_1_Z) + 
  b_1_Z

res = mean(esti2 - esti1)
res_upper = res + qnorm(0.975) * sd(esti2 - esti1) / sqrt(N)
res_lower = res - qnorm(0.975) * sd(esti2 - esti1) / sqrt(N)

data = data.frame(res = res, res_upper = res_upper, res_lower = res_lower)

save(data, file=paste0("/gpfs/home/ll4245/Projects/RieszLearning/superlearner_3/", as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')),".RData"))