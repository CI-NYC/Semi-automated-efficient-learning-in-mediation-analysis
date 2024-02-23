library(readr)
library(dplyr)
simulation_test_one_batch <- read_csv("~/Ivan_Diaz/RieszLearning/data/simulation_test_one_batch_1000_version_RDE_RIE_4.csv")

simulation_test_one_batch

#W_1 = simulation_test_one_batch$W_1
W_2 = simulation_test_one_batch$W_2
#W_3 = simulation_test_one_batch$W_3
A = simulation_test_one_batch$A
Z = simulation_test_one_batch$Z
M = simulation_test_one_batch$M
Y = simulation_test_one_batch$Y

#prob_A = 1 / (1 + exp(-(-1 + W_1 / 3 + W_2 / 4)))
prob_A = 1 / (1 + exp(-(-0.25 + W_2 / 4)))
prob_Z = 1 / (1 + exp(-(-0.25 + A / 2)))
#prob_Z = pnorm(Z, mean = -0.25 + A / 2)
prob_M = 1 / (1 + exp(-(-0.25 + A / 2 - Z / 2 + W_2 / 4)))
#prob_Y = 1 / (1 + exp(-(-1 + M / 2 + A / 4 - Z / 2 + W_1 / 4 - W_3 / 4)))
prob_Y = 1 / (1 + exp(-(-0.25 + M / 2 + A / 4 - Z / 2 + W_2 / 4)))


alpha1 <- 1 / (prob_A)
alpha1 <- case_when(alpha_results$A == 0 ~ 0,
                    TRUE ~ alpha1)

alpha2 <- 1 / (prob_A)
alpha2 <- case_when(alpha_results$A == 0 ~ 0,
                    TRUE ~ alpha2)

prob_Z_A1 = 1 / (1 + exp(-(-0.25 + 1 / 2)))

alpha3_ratio_1_num <- case_when(simulation_test_one_batch$Z == 1 ~ prob_Z_A1,
                                TRUE ~ 1 - prob_Z_A1)
alpha3_ratio_1_denom <- case_when(simulation_test_one_batch$Z == 1 ~ prob_Z,
                                  TRUE ~ 1 - prob_Z)

alpha3 <- alpha2 * alpha3_ratio_1_num / alpha3_ratio_1_denom

prob_M_Z1 = 1 / (1 + exp(-(-0.25 + 0 / 2 - 1 / 2 + W_2 / 4)))
prob_M_Z0 = 1 / (1 + exp(-(-0.25 + 0 / 2 + W_2 / 4)))

# P(M|A, W)
prob_M_non_Z_A0 = prob_M_Z1 * prob_Z_A1 + prob_M_Z0 * (1 - prob_Z_A1)

prob_M_Z1 = 1 / (1 + exp(-(-0.25 + A / 2 - 1 / 2 + W_2 / 4)))
prob_M_Z0 = 1 / (1 + exp(-(-0.25 + A / 2 + W_2 / 4)))

prob_M_non_Z = prob_M_Z1 * prob_Z_A1 + prob_M_Z0 * (1 - prob_Z_A1)

alpha4_ratio_num <- case_when(simulation_test_one_batch$M == 1 ~ prob_M_non_Z_A0,
                              TRUE ~ 1 - prob_M_non_Z_A0)
alpha4_ratio_denom <- case_when(simulation_test_one_batch$M == 1 ~ prob_M_non_Z,
                              TRUE ~ 1 - prob_M_non_Z)

alpha4 <- alpha1 * alpha4_ratio_num / alpha4_ratio_denom


alpha5_ratio_num <- case_when(simulation_test_one_batch$M == 1 ~ prob_M_non_Z,
                              TRUE ~ 1 - prob_M_non_Z)
alpha5_ratio_denom <- case_when(simulation_test_one_batch$M == 1 ~ prob_M,
                                TRUE ~ 1 - prob_M)

alpha5 <- alpha1 * alpha5_ratio_num / alpha5_ratio_denom

simulation_true_dataset <- read_csv("~/Ivan_Diaz/RieszLearning/data/simulation_test_all_batches_1000_version_RDE_RIE_2_true_dataset.csv")

simulation_true_dataset <- simulation_true_dataset |>
  mutate(prob_A = 1 / (1 + exp(-(-0.25 + W_1 / 4)))) |>
  mutate(alpha1_true = case_when(A == 1 ~ 1 / prob_A,
                                 TRUE ~ 0)) |>
  
  mutate(prob_A = 1 / (1 + exp(-(-0.25 + W_1 / 4)))) |>
  mutate(alpha2_true = case_when(A == 0 ~ 1 / (1 - prob_A),
                                 TRUE ~ 0)) |>
  
  mutate(prob_A = 1 / (1 + exp(-(-0.25 + W_1 / 4)))) |>
  mutate(prob_Z = 1 / (1 + exp(-(-0.25 + A / 2)))) |> 
  mutate(prob_Z_A1 = 1 / (1 + exp(-(-0.25 + 1 / 2)))) |> 
  mutate(alpha3_true = alpha2_true * case_when(Z == 1 ~ prob_Z_A1 / prob_Z,
                                               TRUE ~ (1 - prob_Z_A1) / (1 - prob_Z))) |>
  
  mutate(prob_A = 1 / (1 + exp(-(-0.25 + W_1 / 4)))) |>
  mutate(prob_M = dnorm(M, mean = -0.25 + A / 2 - Z / 2 + W_1 ** 2 / 4),
         prob_M_Z1_A1 = dnorm(M, mean = -0.25 + 1 / 2 - 1 / 2 + W_1 ** 2 / 4),
         prob_M_Z0_A1 = dnorm(M, mean = -0.25 + 1 / 2 + W_1 ** 2 / 4),
         prob_Z_A1 = 1 / (1 + exp(-(-0.25 + 1 / 2))),
         prob_M_non_Z_A1 = prob_M_Z1_A1 * prob_Z_A1 + prob_M_Z0_A1 * (1 - prob_Z_A1),
         prob_M_Z1_A0 = dnorm(M, mean = -0.25 - 1 / 2 + W_1 ** 2 / 4),
         prob_M_Z0_A0 = dnorm(M, mean = -0.25 + W_1 ** 2 / 4),
         prob_Z_A0 = 1 / (1 + exp(-(-0.25))),
         prob_M_non_Z_A0 = prob_M_Z1_A0 * prob_Z_A0 + prob_M_Z0_A0 * (1 - prob_Z_A0)) |>
  mutate(prob_M_Z1_A = dnorm(M, mean = -0.25 + A / 2 - 1 / 2 + W_1 ** 2 / 4),
         prob_M_Z0_A = dnorm(M, mean = -0.25 + A / 2 + W_1 ** 2 / 4),
         prob_Z_A = 1 / (1 + exp(-(-0.25 + A / 2))),
         prob_M_non_Z_A = prob_M_Z1_A * prob_Z_A + prob_M_Z0_A * (1 - prob_Z_A)) |>
  mutate(alpha4_true = alpha1_true * prob_M_non_Z_A0 / prob_M_non_Z_A) |>
  
  mutate(alpha5_true = alpha1_true * prob_M_non_Z_A0 / prob_M) |>
  
  select(c(A, Z, M, W_1, Y, alpha5_true, label))

alpha_results <- read_csv("~/Ivan_Diaz/RieszLearning/data/combined_alpha_results.csv")

library(ggplot2)

alpha_results = data.frame(alpha1_pred = alpha_results$alpha1, alpha2_pred = alpha_results$alpha2,
                                alpha3_pred = alpha_results$alpha3, alpha4_pred = alpha_results$alpha4,
                           alpha5_pred = alpha_results$alpha5, alpha1_true = alpha1, alpha2_true = alpha2, alpha3_true = alpha3, 
                           alpha4_true = alpha4, alpha5_true = alpha5)

# alpha1
alpha_results_A0 = alpha_results |>
  filter(alpha1_true == 0)

alpha_results_A1 = alpha_results |>
  filter(alpha1_true != 0)

ggplot(alpha_results_A0, aes(x=alpha1_true, y=alpha1_pred)) + geom_boxplot() +
  ylim(-0.05, 0.2) + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(alpha_results_A0, aes(x=alpha2_true, y=alpha2_pred)) + geom_boxplot() +
  ylim(-0.05, 0.2) + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(alpha_results_A0, aes(x=alpha3_true, y=alpha3_pred)) + geom_boxplot() +
  ylim(-0.05, 0.2) + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))


ggplot(alpha_results_A1, aes(x=alpha1_true, y=alpha1_pred)) + geom_point() + 
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(alpha_results_A1, aes(x=alpha2_true, y=alpha2_pred)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(alpha_results_A1, aes(x=alpha3_true, y=alpha3_pred)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

index = which((alpha_results$alpha3_pred - alpha_results$alpha3_true) < -1.5)
batch = simulation_test_one_batch[index,]


simulation_test_one_batch <- cbind(simulation_test_one_batch, alpha_results)
  
data_all_2 <- data.frame()
for (i in 1:200){
  data_all_temp <- data_all |>
    filter(label == i)
  
  alpha_results_temp <- alpha_results |>
    filter(label == i)
  
  data_all_temp <- cbind(data_all_temp, alpha_results_temp)
  data_all_temp <- data_all_temp |>
    select(c(W_2, A, Z, M, Y, alpha1, alpha2, alpha3, alpha4, alpha5))
  data_all_2 <- rbind(data_all_2, data_all_temp)
}

simulation_test_one_batch_summary <- simulation_test_one_batch |>
  group_by(W_2, A, Z, M) |>
  summarise(alpha1_pred_mean = mean(alpha1_pred), alpha1_true_mean = mean(alpha1_true),
            alpha2_pred_mean = mean(alpha2_pred), alpha2_true_mean = mean(alpha2_true),
            alpha3_pred_mean = mean(alpha3_pred), alpha3_true_mean = mean(alpha3_true),
            alpha4_pred_mean = mean(alpha4_pred), alpha4_true_mean = mean(alpha4_true),
            alpha5_pred_mean = mean(alpha5_pred), alpha5_true_mean = mean(alpha5_true),
            count_points = n())

simulation_test_one_batch_summary <- alpha_results |>
  group_by(W_2, A, Z, M) |>
  summarise(alpha1_pred_mean = mean(alpha1),
            alpha2_pred_mean = mean(alpha2),
            alpha3_pred_mean = mean(alpha3), 
            alpha4_pred_mean = mean(alpha4),
            alpha5_pred_mean = mean(alpha5),
            count_points = n())

mean((simulation_test_one_batch_summary$alpha3_pred_mean - simulation_test_one_batch_summary$alpha3_true_mean)**2)

# alpha2

ggplot(alpha_results, aes(x=alpha2_true, y=alpha2_pred)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(alpha_results_A1, aes(x=alpha2_true, y=alpha2_pred)) + geom_boxplot() + 
  ylim(-3.5, -0.5) + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

cor(alpha_results$alpha2_true, alpha_results$alpha2_pred)



# Test

summary_results <- data.frame(sample_size = 0, W_2 = 0, A = 0, Z = 0, prob = 0)
summary_results[1,] <- c(1000, 1, 1, NA, 0.207)
summary_results[2,] <- c(1000, 1, 0, NA, 0.126)
summary_results[3,] <- c(1000, 0, 1, NA, 0.155)
summary_results[4,] <- c(1000, 0, 0, NA, 0.122)
summary_results[5,] <- c(1000, 1, 1, 0, 0.210)
summary_results[6,] <- c(1000, 1, 0, 0, 0.124)
summary_results[7,] <- c(1000, 0, 1, 0, 0.153)
summary_results[8,] <- c(1000, 0, 0, 0, 0.127)
summary_results[9,] <- c(1000, 1, 1, 1, 0.190)
summary_results[10,] <- c(1000, 1, 0, 1, 0.143)
summary_results[11,] <- c(1000, 0, 1, 1, 0.167)
summary_results[12,] <- c(1000, 0, 0, 1, 0.089)

summary_results[1,] <- c(10000, 1, 1, NA, 0.218)
summary_results[2,] <- c(10000, 1, 0, NA, 0.131)
summary_results[3,] <- c(10000, 0, 1, NA, 0.205)
summary_results[4,] <- c(10000, 0, 0, NA, 0.114)
summary_results[5,] <- c(10000, 1, 1, 0, 0.214)
summary_results[6,] <- c(10000, 1, 0, 0, 0.132)
summary_results[7,] <- c(10000, 0, 1, 0, 0.203)
summary_results[8,] <- c(10000, 0, 0, 0, 0.116)
summary_results[9,] <- c(10000, 1, 1, 1, 0.242)
summary_results[10,] <- c(10000, 1, 0, 1, 0.126)
summary_results[11,] <- c(10000, 0, 1, 1, 0.221)
summary_results[12,] <- c(10000, 0, 0, 1, 0.098)



# 

n <- 10000
W <- rbinom(n, 1, plogis(1))
A <- rbinom(n, 1, plogis(1 + W))
Z <- rbinom(n, 1, plogis(1 - A + W))
M <- rbinom(n, 1, plogis(-1 + A + Z*W))
data <- data.frame(W, A, Z, M)

library(tidyverse)
## P(M=1 | A=a, Z=z)
data %>% group_by(A, W) %>% summarise(Pmaz = mean(M))

data %>% group_by(A, W) %>% mutate(Zstar = sample(Z)) %>%
  group_by(A, W, Zstar) %>% summarise(Pmaz = mean(M))

data %>% group_by(A, W) %>% summarise(Pmaz = mean(M))

data %>% group_by(A, W) %>% mutate(Zstar = sample(Z)) %>%
  group_by(A, W, Zstar) %>% summarise(Pmaz = mean(M))



combined_alpha_results_RDE_RIE_cont_M_a_10 <- read_csv("Ivan_Diaz/RieszLearning/data/combined_alpha_results_RDE_RIE_cont_M_a_10.csv")

combined_alpha_results_RDE_RIE_cont_M_a_10_A1 <- combined_alpha_results_RDE_RIE_cont_M_a_10 |>
  filter(A == 1)

combined_alpha_results_RDE_RIE_cont_M_a_10_A0 <- combined_alpha_results_RDE_RIE_cont_M_a_10 |>
  filter(A == 0)

ggplot(combined_alpha_results_RDE_RIE_cont_M_a_10_A1, aes(x=alpha1_true, y=alpha1)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(combined_alpha_results_RDE_RIE_cont_M_a_10_A1, aes(x=alpha4_true, y=alpha4)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(combined_alpha_results_RDE_RIE_cont_M_a_10_A1, aes(x=alpha5_true, y=alpha5)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(combined_alpha_results_RDE_RIE_cont_M_a_10_A0, aes(x=alpha2_true, y=alpha2)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))

ggplot(combined_alpha_results_RDE_RIE_cont_M_a_10_A0, aes(x=alpha3_true, y=alpha3)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, colour = "red") + 
  theme(axis.text=element_text(size=15), axis.title=element_text(size=15))


combined_alpha_results_RDE_RIE_cont_M_a_10



combined_alpha_results_RDE_RIE_cont_M_a_10_estimate <- combined_alpha_results_RDE_RIE_cont_M_a_10 |>
  mutate(esti = alpha5_true * Y) |>
  group_by(label) |>
  summarise(true_esti = mean(esti))

combined_alpha_results_RDE_RIE_cont_M_a_10_estimate

combined_alpha_results_RDE_RIE_cont_M_a_10_estimate_pred <- read_csv("Ivan_Diaz/RieszLearning/data/combined_alpha_results_esti_RDE_RIE_cont_M_a_10.csv")

combined_alpha_results_RDE_RIE_cont_M_a_10_estimate <- combined_alpha_results_RDE_RIE_cont_M_a_10_estimate |>
  left_join(combined_alpha_results_RDE_RIE_cont_M_a_10_estimate_pred, by = "label")

combined_alpha_results_RDE_RIE_cont_M_a_10_estimate <- combined_alpha_results_RDE_RIE_cont_M_a_10_estimate |>
  mutate(bias = pred + 0.1352368) |>
  mutate(cover = case_when(pred_upper > -0.1352368 & pred_lower <= -0.1352368 ~ 1,
                           TRUE ~ 0))
combined_alpha_results_RDE_RIE_cont_M_a_10_estimate <- combined_alpha_results_RDE_RIE_cont_M_a_10_estimate |>
  select(label, true_esti, pred, pred_upper, pred_lower, bias, cover)
write_csv(combined_alpha_results_RDE_RIE_cont_M_a_10_estimate, file = "Ivan_Diaz/RieszLearning/data/estimates_cont_M_a_10.csv")
