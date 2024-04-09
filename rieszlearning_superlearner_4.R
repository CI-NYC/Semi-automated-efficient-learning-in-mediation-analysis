# Num: 4
# Path 4 in the recanting twins
# Contr of (0, 1, 0) and (0, 0, 0)

library(mlr3extralearners)
library(mlr3superlearner)
library(data.table)
library(tidyverse)
library(Rglpk)

label = as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
N = 2000

filename = paste0("~/Projects/RieszLearning/results_000/N_", N, "_seq_", label, ".csv")
combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 <- read_csv(filename)

a_1 = 0
a_2 = 0
a_3 = 0

res_summary <- data.frame(res = NULL, res_upper = NULL, res_lower = NULL)

attr_models = c("lightgbm", "mean", "earth", "nnet")

batch_1 <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 

# Fit theta_3 and b_3
fit_3 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3", "Y"), with = F],
                          target = "Y",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_3 <- predict(fit_3, batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_1 <- batch_1 %>%
  mutate(A = a_1)

batch_1_a_2 <- batch_1 %>%
  mutate(A = a_2)

batch_1_a_3 <- batch_1 %>%
  mutate(A = a_3)

b_3 <- predict(fit_3, batch_1_a_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1$b_3 = b_3

fit_2 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3", "b_3"), with = F],
                          target = "b_3",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_2 <- predict(fit_2, batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_2 <- predict(fit_2, batch_1_a_2[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1$b_2 <- b_2

fit_1 <- mlr3superlearner(data = batch_1[, c("A", "W_1", "W_2", "W_3", "b_2"), with = F],
                          target = "b_2",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_1 <- predict(fit_1, batch_1[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_1 <- predict(fit_1, batch_1_a_3[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Summarize
Y <- batch_1$Y
alpha_3 <- batch_1$alpha3
alpha_2 <- batch_1$alpha2
alpha_1 <- batch_1$alpha1

esti1 <- alpha_3 * (Y - theta_3) +
  alpha_2 * (b_3 - theta_2) + 
  alpha_1 * (b_2 - theta_1) + b_1

filename = paste0("~/Projects/RieszLearning/results_010/N_", N, "_seq_", label, ".csv")
combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 <- read_csv(filename)

a_1 = 0
a_2 = 1
a_3 = 0

res_summary <- data.frame(res = NULL, res_upper = NULL, res_lower = NULL)

attr_models = c("lightgbm", "mean", "earth", "nnet")

batch_1 <- combined_alpha_results_RDE_RIE_cont_M_cont_Z_a_10 

# Fit theta_3 and b_3
fit_3 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3", "Y"), with = F],
                          target = "Y",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_3 <- predict(fit_3, batch_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1_a_1 <- batch_1 %>%
  mutate(A = a_1)

batch_1_a_2 <- batch_1 %>%
  mutate(A = a_2)

batch_1_a_3 <- batch_1 %>%
  mutate(A = a_3)

b_3 <- predict(fit_3, batch_1_a_1[, c("A", "Z_1", "Z_2", "M_1", "M_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1$b_3 = b_3

fit_2 <- mlr3superlearner(data = batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3", "b_3"), with = F],
                          target = "b_3",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_2 <- predict(fit_2, batch_1[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_2 <- predict(fit_2, batch_1_a_2[, c("A", "Z_1", "Z_2", "W_1", "W_2", "W_3"), with = F], discrete = F)

batch_1$b_2 <- b_2

fit_1 <- mlr3superlearner(data = batch_1[, c("A", "W_1", "W_2", "W_3", "b_2"), with = F],
                          target = "b_2",
                          library = attr_models,
                          outcome_type = "continuous",
                          folds = 10)

theta_1 <- predict(fit_1, batch_1[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

b_1 <- predict(fit_1, batch_1_a_3[, c("A", "W_1", "W_2", "W_3"), with = F], discrete = F)

# Summarize
Y <- batch_1$Y
alpha_3 <- batch_1$alpha3
alpha_2 <- batch_1$alpha2
alpha_1 <- batch_1$alpha1

esti2 <- alpha_3 * (Y - theta_3) +
  alpha_2 * (b_3 - theta_2) + 
  alpha_1 * (b_2 - theta_1) + b_1

res = mean(esti2 - esti1)
res_upper = res + qnorm(0.975) * sd(esti2 - esti1) / sqrt(N)
res_lower = res - qnorm(0.975) * sd(esti2 - esti1) / sqrt(N)

data = data.frame(res = res, res_upper = res_upper, res_lower = res_lower)

save(data, file=paste0("/gpfs/home/ll4245/Projects/RieszLearning/superlearner_4/", as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID')),".RData"))
