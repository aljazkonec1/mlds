library(rstanarm)
library(ggplot2)
library(cowplot)

#load data

shots_data <- read.csv("dataset.csv")
#normalize data
shots_data$Angle <- (shots_data$Angle - mean(shots_data$Angle)) / sd(shots_data$Angle)
shots_data$Distance <- (shots_data$Distance - mean(shots_data$Distance)) / sd(shots_data$Distance)
summary(shots_data)




model <- rstanarm::stan_glm(Made ~  Angle + Distance, data = shots_data, family = binomial(link = "logit"), 
                            chains = 5, iter = 2500, warmup = 500)
samples <- as.data.frame(model)
summary(samples)
sd(samples$Angle)
sd(samples$Distance)

set.seed(42) 
subset_shots_data <- shots_data[sample(nrow(shots_data), 50), ]

subset_model <- rstanarm::stan_glm(Made ~ Angle + Distance, data = subset_shots_data, family = binomial(link = "logit"), 
                                   chains = 5, iter = 2500, warmup = 500)


subset_samples <- as.data.frame(subset_model)
summary(subset_samples)
sd(subset_samples$Angle)
sd(subset_samples$Distance)

y_limits <- range(c(samples$Distance, subset_samples$Distance))
x_limits <- range(c(samples$Angle, subset_samples$Angle))


angle_distance <- ggplot(samples, aes(x = Angle, y = Distance)) +
    geom_point(alpha = 0.4) +
    stat_density_2d(aes(fill = ..level..), geom = "polygon", color = "white", alpha = 0.2) +
    scale_fill_viridis_c() +
    labs(title = "Posterior Samples of Angle and Distance Coefficients",
       x = "Angle Coefficient",
       y = "Distance Coefficient")+
    theme(legend.position = "none")+
    theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 16), 
        axis.title.y = element_text(size = 16), 
        axis.text = element_text(size = 14))+
      ylim(y_limits) + xlim(x_limits)
angle_distance
ggsave("report/figures/angle_distance_plot.pdf", plot= angle_distance, width= 9, height = 16, units = "in")




subset_angle_plot <- ggplot(subset_samples, aes(x = Angle, y = Distance)) +
    geom_point(alpha = 0.4) +
    stat_density_2d(aes(fill = ..level..), geom = "polygon", color = "white", alpha = 0.2) +
    scale_fill_viridis_c() +
    labs(title = "Subsampled posterior of Angle and Distance Coefficients",
       x = "Angle Coefficient",
       y = "Distance Coefficient")+
    theme(legend.position = "none")+
    theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 16), 
        axis.title.y = element_text(size = 16), 
        axis.text = element_text(size = 14))+
          ylim(y_limits)+ xlim(x_limits)
subset_angle_plot

ggsave("report/figures/subsample_angle_distance_plot.pdf", plot= subset_angle_plot, width= 9, height = 16, units = "in")


combined_plot <- plot_grid(angle_distance, subset_angle_plot, ncol = 1, nrow = 2)
combined_plot
ggsave("report/figures/combined_plot_vertical.pdf", plot = combined_plot, width = 9, height = 16)

combined_plot <- plot_grid(angle_distance, subset_angle_plot)
# ggsave("report/figures/combined_plot_horizontal.pdf", plot = combined_plot, width = 16, height = 9)


# importance of features
samples$Angle
samples$Distance
abs(samples$Distance) > abs(samples$Angle)
mean (abs(samples$Distance) > abs(samples$Angle))



# Shot success based on increasing angle 
prob_angle_negative <- mean(samples$Angle < 0)
prob_angle_negative
