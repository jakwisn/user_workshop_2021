# Judges are often presented with two sets of scores from the Compas system -- one that classifies people into High,
# Medium and Low risk, and a corresponding decile score.
# There is a clear downward trend in the decile scores as those scores increase for white defendants.


################# imports #################

library(magrittr)
library(ggplot2)
library(dplyr)
library(DALEX)
library(fairmodels)
library(gbm)
library(ranger)

fairmodels::compas

compas2 <- read.csv('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv')
head(compas2)

df <- dplyr::select(compas2, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, juv_misd_count, v_decile_score,
                    days_b_screening_arrest, decile_score, two_year_recid, c_jail_in, c_jail_out) %>%
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A') %>%
  mutate(jail_time = as.numeric(difftime(c_jail_out, c_jail_in, units = c('days'))))

head(df)
df <- df[ -c(4:5, 13:14)]


ggplot(df[c('race', 'decile_score')], aes(x=decile_score)) + geom_bar() + facet_grid(~race)

head(df)

# Here we change the order of the
df$two_year_recid <- ifelse(df$two_year_recid == 1, 0, 1)


################# Fairness Check #################

# Classification task - will defendants become recidivist?

lr_model <- glm(two_year_recid ~., data = df, family = binomial())
lr_explainer <- DALEX::explain(lr_model, data = df, y = df$two_year_recid)

# lets check the performance
model_performance(lr_explainer)

# let's do fairness check and quickly check if the model is fair
fairness_check(lr_explainer, protected = df$race, privileged = 'Caucasian')

# assigning to variable
fobject <- fairness_check(lr_explainer, protected = df$race, privileged = 'Caucasian')
plot(fobject)


# Insides
fobject$groups_confusion_matrices
fobject$groups_data
fobject$fairness_check_data

fobject$parity_loss_metric_data
fobject$privileged
fobject$protected
fobject$cutoff
fobject$epsilon

# what if we want to check more than one model? That is fine but first...

df <- df[df$race %in% c('Caucasian', 'African-American'), ]
head(df)

lr_model <- glm(two_year_recid ~., data = df, family = binomial())
lr_explainer <- DALEX::explain(lr_model, data = df, y = df$two_year_recid)

fairness_check(lr_explainer, protected = df$race, privileged = 'Caucasian')

fobject <- fairness_check(lr_explainer, protected = df$race, privileged = 'Caucasian')

plot(fobject)

# what happens if i tweak this?
fobject <- fairness_check(lr_explainer,
                          protected = df$race,
                          privileged = 'Caucasian',
                          epsilon = 0.6)

plot(fobject)

# ok but lets revert now
fobject <- fairness_check(lr_explainer, protected = df$race, privileged = 'Caucasian')


# now we are ready, let's add another model

head(df)
rf_model <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df,
                           probability = TRUE,
                           seed = 123)

rf_explainer <- DALEX::explain(rf_model, data = df, y = df$two_year_recid)

model_performance(rf_explainer)

# we have a few options to compare the models


# 1. explainer and fairness object
## 1.1
fobject2 <- fairness_check(rf_explainer, fobject, # with fobject
                           protected = df$race,
                           privileged = "Caucasian")
## 1.2 (recommended)
fobject2 <- fairness_check(rf_explainer, fobject) # with fobject

# 2. fairness objects
fobject2 <- fairness_check(rf_explainer,
                           protected = df$race,
                           privileged = "Caucasian")

fobject2 <- fairness_check(fobject, fobject2)
plot(fobject2)


# 3. 2 explainers
fobject2 <- fairness_check(rf_explainer, lr_explainer,
                           protected = df$race,
                           privileged = "Caucasian")

plot(fobject2)

# more than 2 is ok.

# what if the race in the data is the case?
df2 <- df[c('two_year_recid', 'age', 'c_charge_degree', 'priors_count', 'decile_score')]

rf_model2 <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df2,
                           probability = TRUE)

rf_explainer2 <- DALEX::explain(rf_model2, data = df2, y = df2$two_year_recid)
model_performance(rf_explainer2)

fobject3 <- fairness_check(rf_explainer2, fobject2,
                           protected = df$race,
                           privileged = "Caucasian")

fobject3 <- fairness_check(rf_explainer2, fobject2,
                           protected = df$race,
                           privileged = "Caucasian",
                           label = 'ranger_without_race')

plot(fobject3)

################# Other Plots #################

# shows raw scores of metrics
fobject3 %>% metric_scores() %>% plot()

# parity loss
fobject3$parity_loss_metric_data


fobject3 %>% fairness_radar() %>% plot()
?fairness_radar
fobject3 %>% fairness_radar(fairness_metrics = c("TPR", "STP", "FPR")) %>% plot()

# all metrics?

fobject3 %>% fairness_heatmap() %>% plot()

# summarised metrics?
# default metrics - those in fairness_check
fobject3 %>% stack_metrics() %>% plot()


# metric and performance? No problem
fobject3 %>% performance_and_fairness(fairness_metric = 'FPR', performance_metric = 'accuracy') %>%
  plot()

# hard to remember? No problem!

?plot_fairmodels()

fobject3 %>% plot_fairmodels("stack_metrics")

# okay we have bias? What can we do?
# A lot!


################# Mitigation methods #################

##### Pre processing

# Let's construct a model previously used.

fobject <- fairness_check(rf_explainer, protected = df$race, privileged = 'Caucasian')

# resampling
indices <- resample(protected = df$race, df$two_year_recid)
df_resampled <- df[indices,]

rf_model_resampled <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df_resampled,
                           probability = TRUE)

rf_explainer_resampled <- DALEX::explain(rf_model_resampled,
                               data = df,
                               y = df$two_year_recid,
                               label = 'resampled')


fobject <- fairness_check(fobject, rf_explainer_resampled)


# reweight

weights <- reweight(protected = as.factor(df$race), y=df$two_year_recid)

rf_model_reweighted <- ranger::ranger(as.factor(two_year_recid) ~.,
                                     data=df,
                                     case.weights = weights,
                                     probability = TRUE)

rf_explainer_reweighted <- DALEX::explain(rf_model_reweighted,
                                         data = df,
                                         y = df$two_year_recid,
                                         label = 'reweighted')


fobject <- fairness_check(fobject, rf_explainer_reweighted)


##### Post-Processing

# ROC pivot
rf_explainer_roc <- roc_pivot(rf_explainer, df$race, "Caucasian", theta = 0.05)


fobject <- fairness_check(fobject, rf_explainer_roc, label = "roc")

plot(fobject)

# Cutoff manipulation

rf_explainer %>%
  fairness_check(protected = df$race, privileged = 'Caucasian') %>%
  ceteris_paribus_cutoff("African-American") %>%
  plot()

fobject <- fairness_check(fobject, rf_explainer,
                          cutoff = list('African-American'=0.4),
                          label = 'ranger_cutoff')

plot(fobject)

# checking FPR and accuracy
fobject %>% performance_and_fairness(fairness_metric = 'FPR', performance_metric = 'accuracy') %>% plot()

################# Exercise #################

# As for the exercise check fairness for the same models but with protected vector equal to df$sex


