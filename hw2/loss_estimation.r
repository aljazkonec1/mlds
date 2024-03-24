library(ggplot2)
library(patchwork)

toy_data <- function(n, seed = NULL) {
set.seed(seed)
x <- matrix(rnorm(8 * n), ncol = 8)
z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
y <- runif(n) > 1 / (1 + exp(-z))
return (data.frame(x = x, y = y))
}

log_loss <- function(y, p) {
    -(y * log(p) + (1 - y) * log(1 - p))
}


df_dgp <- toy_data(100000, 0)

# the variance drops by sqrt(n) as n increases, sqrt(100000) = 316
#which means when we devide we get at most to third decimal
data <- toy_data(50,0)

h <- glm(y ~ ., data = data, family = binomial(link=logit))
h
pred <- predict(h, df_dgp, type = "response")
true_risk <- mean(log_loss(df_dgp$y, pred))
true_risk

means <- c()
std_errors <- c()

contains_true_risk <- c()

for ( x in 1:1000){
    data <- toy_data(50)

    pred <- predict(h, data, type = "response")
    loss <- log_loss(data$y, pred)

    means <- c(means, mean(loss))
    
    m <- mean(loss)
    std_error <- sd(loss)/sqrt(50)
    std_errors <- c(std_errors, std_error)
    ci.lower <- m - 1.96 * std_error
    ci.upper <- m + 1.96 * std_error

    contains_true_risk <- c(contains_true_risk, (ci.lower <= true_risk) & (ci.upper >= true_risk))

}

sum(contains_true_risk)/1000

mean_observed_risk <- mean(means)
mean_observed_risk
true_risk

mean_difference <- abs(round(mean( means - true_risk), 4))
mean_difference


differences <- means - true_risk
density <- density(differences)
plot(density)

always_05_risk <- mean(log_loss(df_dgp$y, rep(0.5, nrow(df_dgp))))
always_05_risk

median_standard_error <- median(std_errors)
median_standard_error


## oversetimation of the deployed models risk

diff_in_risk <- c()

for(i in 1:50){
    dataset1 <- toy_data(50)
    dataset2 <- toy_data(50)

    h1 <- glm(y ~ ., data = dataset1, family = binomial(link=logit))
    combined <- rbind(dataset1, dataset2)
    h2 <- glm(y ~ ., data = combined, family = binomial(link=logit))

    pred1 <- predict(h1, df_dgp, type = "response")
    pred2 <- predict(h2, df_dgp, type = "response")

    risk1 <- mean(log_loss(df_dgp$y, pred1))
    risk2 <- mean(log_loss(df_dgp$y, pred2))

    diff_in_risk <- c(diff_in_risk, risk1 - risk2)
}

summary(diff_in_risk)



## loss estimator variability due to split variability

data <- toy_data(100, 0)

h0 <- glm(y ~ ., data = data, family = binomial(link=logit))
true_risk <- mean(log_loss(df_dgp$y, predict(h0, df_dgp, type = "response")))
true_risk

contains_true_risk <- c()
means <- c()
std_errors <- c()

for (i in 1:1000) {

    t <- sample(1:100, 50)

    training <- data[t, ]
    testing <- data[-t, ]

    h <- glm(y ~ ., data = training, family = binomial(link=logit))
    pred <- predict(h, testing, type = "response")
    loss <- log_loss(testing$y, pred)
    m <- mean(loss)
    means <- c(means, m)
    
    std_error <- sd(loss)/sqrt(50)
    std_errors <- c(std_errors, std_error)

    ci.lower <- m - 1.96 * std_error
    ci.upper <- m + 1.96 * std_error

    contains_true_risk <- c(contains_true_risk, (ci.lower <= true_risk) & (ci.upper >= true_risk))

}

mean(means - true_risk)
median(std_errors)


mean(contains_true_risk)

density <- density(means - true_risk)
plot(density)


## cross-validation

crossvalidation <- function(data, n, nfolds) {
    fold_size <- floor(n/nfolds)
    
    s <- sample(1:n, n)
    losses <- c()

    for (i in 1:nfolds) {
        test <- data[s[((i-1)*fold_size + 1):(i*fold_size)], ]
        train <- data[s[-(((i-1)*fold_size + 1):(i*fold_size))], ]

        h <- glm(y ~ ., data = train, family = binomial(link=logit))
        pred <- predict(h, test, type = "response")
        loss <- log_loss(test$y, pred)
        losses <- c(losses, loss)
    } 

    return(losses[order(s)])
}


repeat500times <- function(n, nfolds, rep= FALSE){
    ctr <- c()
    means <- c()
    std_errors <- c()

    for (i in 1:500) {
        data <- toy_data(n, i)

        h0 <- glm(y ~ ., data = data, family = binomial(link=logit))
        true_risk <- mean(log_loss(df_dgp$y, predict(h0, df_dgp, type = "response")))

        if (rep){
            all_losses <- numeric(n)
            for (j in 1:20){
                losses <- crossvalidation(data, n, nfolds)
                all_losses <- all_losses + losses
             }
            losses <- all_losses/20
        }
        else {
            losses <- crossvalidation(data, n, nfolds)

        }

        mean_loss <- mean(losses)
        se <- sd(losses)/sqrt(n)

        ci_lower <- mean_loss - 1.96 * se
        ci_upper <- mean_loss + 1.96 * se

        ctr <- c(ctr, (ci_lower <= true_risk) & (ci_upper >= true_risk))
        means <- c(means, mean_loss - true_risk)
        std_errors <- c(std_errors, se)
    }
    return (list(ctr,means, std_errors))
}


# 2-fold CV

name <- list("2-fold" = 2, "4-fold" = 4, "10-fold" = 10, "10-fold-20-rep"= 10, "loocv" = 100)
densities <- list()

for (cv in names(name)) {

    if (cv == "10-fold-20-rep"){
        l <- repeat500times(100, name[[cv]], rep = TRUE)
        } else {
        l <- repeat500times(100, name[[cv]])
    }
    ctr <- l[[1]]
    means <- l[[2]]
    std_errors <- l[[3]]

    print(paste0("Number of folds: ", cv))
    print(paste0("Mean difference: ", mean(round(means, 4))))
    print(paste0("Median standard error: ", median(round(std_errors, 4))))

    print(paste0("Percentage of 95CI that conatin true risk proxy: ", mean(ctr)*100))
    d <- density(means)
    densities[[cv]] <- d

}

plots <- list()
for (cv in names(densities)) {
    # plot(densities[[cv]], main = cv, asp=1)
    d <- densities[[cv]]
    df <- data.frame(x = d$x, y = d$y)
    plots[[cv]] <-ggplot(df, aes(x, y)) + 
                geom_line() +
                ggtitle(cv) + 
                theme(aspect.ratio=1) +
                ylab("Density") +
                xlab("est_risk - true_risk")
}   

plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]]+ plot_layout(ncol = 3, nrow = 2) 
# plots[[1]] + plots[[2]] + plots[[3]] + plot_layout(ncol = 3, nrow = 2) 
