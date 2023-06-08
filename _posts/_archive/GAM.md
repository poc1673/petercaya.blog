---
layout: post
title: "GAM and Trees"
date:   2021-05-21
categories: programming
---

```r
rm(list = ls())
library(pacman)
p_load(data.table,caret,ggplot2,plotly,mgcv,rpart,magrittr,precrec,MLmetrics,partykit,gam,rmarkdown,knitr,broom)
set.seed(13)
with.nonlinear <- data.table(twoClassSim(10000,linearVars = 6,ordinal = F ) )
```

# Introduction

A few months ago, I came across [this](https://stackoverflow.com/questions/41692017/decision-trees-combined-with-logistic-regression) article on StackExchange on employing shallow decision trees as a feature engineering step for logistic regression. This technique attempts to account for potential nonlinear relationships in dependent variables. After reading the post and considering other arguments, my impression was that performing this introductory step removes useful information on the features and could decrease the flexibility of the model. Depending on the application, it may be more useful to either transform the data to remove the nonlinearities, or to account for them in the specification of the model.

To investigate further, I will first assess the effectiveness of decision trees as a feature engineering strategy by comparing them to logistic regression and decision trees alone. Afterwards, I will discuss the generalized additive model (GAM) framework and how it also addresses this issue before performing a brief comparison of the overall results.

# Motivation for using decision trees as an input

Logistic regression and decision trees are the first two modeling methodologies used to introduce classification models. Each method has its own pitfall. Logistic regression assumes all variables are described by a linear function, IE the model has to take the form:


<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i&space;x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i&space;x_i" title="\hat{y} = \beta_0 + \sum_{i=1}^K \beta_i x_i" /></a>

Decision trees don't make an assumption on the distribution of variables, they simply create a new branch in the decision tree based on a criteria like Gini impurity. However, decision trees are prone to overfitting on data and can be unstable to changes in the training data

To try to remedy the downsides of these two methods, several sources have suggested using a decision tree as an intermediate step which helps remove potential nonlinearity in the model. In its most simple form, the process is as follows:

1.  Fit a shallow decision tree, <a href="https://www.codecogs.com/eqnedit.php?latex=T(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(X)" title="T(X)" /></a>   for the training data, x. This tree will have N terminal nodes.
2.  The N categorical variables denoted <a href="https://www.codecogs.com/eqnedit.php?latex=C_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_n" title="C_n" /></a> are included as features in the logistic regression specification.
3.  The logistic regression is fit using the modified set of data.




# Alternatives to remedy nonlinearity in the data:

There are two simple alternatives to handling nonlinearity in data. The first choice is to simply transform the raw data into to attempt to give it a linear relationship with the dependent variable. This strategy isn't always an option, though, and its appropriateness varies based on the domain. The second way to eliminate it is to change the model specification to a method which can cope with nonlinear data.

## Generalized additive models

There are several methodologies that can be used to cope with nonlinear data, but one the one I have chosen for this exercise is the generalized additive model. Generalized additive models are a framework originally introduced in "[Generalized Additive Models](https://web.stanford.edu/~hastie/Papers/gam.pdf)" by Trevor Hastie and Robert Tibshirani. The authors went on to write [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) which is where I first encountered it.

In brief, it takes the concept of the generalized linear model that logistic regression is based off of, and relaxes the assumption of linear basis functions. The results is a model specification like that described earlier.

The equation for the GAM equivalent for logistic regression would be:

<a href="https://www.codecogs.com/eqnedit.php?latex=ln\Big(&space;\frac{\mu(X)}{1-\mu(X)}&space;\Big)=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i(x_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln\Big(&space;\frac{\mu(X)}{1-\mu(X)}&space;\Big)=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i(x_i)" title="ln\Big( \frac{\mu(X)}{1-\mu(X)} \Big)= \beta_0 + \sum_{i=1}^K \beta_i(x_i)" /></a>

The notation above substitutes <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i(\bullet)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i(\bullet)" title="\beta_i(\bullet)" /></a> for <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i" title="\beta_i" /></a> because we have changed the term operating on each dependent variable to an arbitrary smooth function. With this in mind, we could think of a logistic regression model as an additive model where:

<a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i(x)&space;=&space;c_ix_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i(x)&space;=&space;c_ix_i" title="\beta_i(x) = c_ix_i" /></a>

For some regression coefficient <a href="https://www.codecogs.com/eqnedit.php?latex=c_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_i" title="c_i" /></a>. It is clear that this offers us additional flexibility in how we treat the variables that we are modeling, so we will use this specification in our numerical experiment as a challenger to the tree+logistic regression specification.

## Numerical Experiment:

For a numerical experiment, I'm using a synthetically created set of data. I am choosing this method over a real data-set because it allows me to control for the functional form of the features to ensure that some of them are truly nonlinear. I have linked to a post on another blog where the author has evaluated the decision tree + logistic regression methodology on a banking data-set.

To test the results, we will first fit a standard CART decision tree and a logistic regression model using the same sets of variables. Then, a CART decision tree with a max depth of 3 will be used to generate a set of class labels which will included in the training data for a new logistic regression model. Then, the results will be compared using ROC and precision-recall methods.

We begin by fitting a simple CART decision tree using the *rpart* package on all meaningful variables. Next, a simple logistic regression model is fit on all variables in the data. Finally, we fit another simple CART decision tree using with a limited depth of three. This is then used to create a feature set which is used as part of the training process for the model:


```r
train.index <- createDataPartition(y = with.nonlinear$Class, p = .8,list =F , time = 1)
training_data <- with.nonlinear[train.index]
testing_data <- with.nonlinear[-train.index]
simple.cart <- rpart(data = training_data,formula = Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3)
glm.mod <- glm(data = training_data,formula =  Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3,family = "binomial")
# Generate the combined model:
tree.for.feed <- rpart(data = training_data,
                       formula = Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3,
                        control = rpart.control(  maxdepth = 3))
training_data$labels <- factor(tree.for.feed$where)
glm.w.tree <- glm(data = training_data,formula =  Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6 +Nonlinear1 +Nonlinear2 +Nonlinear3+ labels,family = "binomial")
glm.w.only.tree <- glm(data = training_data,formula =  Class~labels,family = "binomial")
```

We can see that the tree + logistic regression model has


```r
tidy(glm.w.tree)
```

```
## Warning: `...` is not empty.
## 
## We detected these problematic arguments:
## * `needs_dots`
## 
## These dots only exist to allow future extensions and should be empty.
## Did you misspecify an argument?
```

```
## # A tibble: 15 x 5
##    term        estimate std.error statistic  p.value
##    <chr>          <dbl>     <dbl>     <dbl>    <dbl>
##  1 (Intercept)  -0.656     0.0786    -8.35  6.71e-17
##  2 Linear1      -0.0236    0.0252    -0.937 3.49e- 1
##  3 Linear2      -0.625     0.0429   -14.6   4.62e-48
##  4 Linear3       0.525     0.0416    12.6   1.49e-36
##  5 Linear4      -0.449     0.0297   -15.1   1.31e-51
##  6 Linear5       0.309     0.0258    12.0   4.46e-33
##  7 Linear6      -0.143     0.0254    -5.62  1.86e- 8
##  8 Nonlinear1    0.483     0.0443    10.9   1.25e-27
##  9 Nonlinear2    0.391     0.0873     4.48  7.55e- 6
## 10 Nonlinear3    0.293     0.0878     3.34  8.46e- 4
## 11 labels5      -0.0257    0.127     -0.202 8.40e- 1
## 12 labels6       0.153     0.100      1.53  1.25e- 1
## 13 labels9       0.0247    0.122      0.203 8.39e- 1
## 14 labels10      0.379     0.167      2.27  2.31e- 2
## 15 labels11      0.264     0.0976     2.70  6.87e- 3
```

Next, we train the GAM model:


```r
formatted.results <-  ifelse(training_data$Class == "Class2",1,0)
gam.model <- gam(data = training_data,formula = Class~s(Linear1)+s(Linear2)+s(Linear3)+s(Linear4)+s(Linear5)+s(Linear6)+s(Nonlinear1) +s(Nonlinear2) +s(Nonlinear3),family = binomial)
```

The GAM model's plot function allows us to see the smoothing functions that were fit by the model (corresponding to $\beta_i(x_i))$. If we inspect the form of the GAM, we can see that the smoothing functions catch the nonlinearity in the data. The linear variables appear as follows:


```r
plot(gam.model)
```
![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-2.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-3.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-4.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-5.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-6.png)<!-- -->

The nonlinear variables are modeled in the function as follows:

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-7.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-8.png)<!-- -->![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-5-9.png)<!-- -->


# Performance Comparison

We begin with the synthetic data-set described earlier. We will view the performance of the models in-sample and out-of-sample together to see how well the model performs in general, and to see how large the drop in performance is out-of-sample:


```r
simple.cart.eval <- evalmod(scores = predict(object = tree.for.feed )[,2], labels = formatted.results)
glm.mod.eval <- evalmod(scores = predict(object = glm.mod,type = "response"), labels = formatted.results)
mixed.for.eval <- evalmod(scores = predict(object = glm.w.tree,type = "response"), labels = formatted.results)
glm.w.only.tree.for.eval<- evalmod(scores = predict(object = glm.w.only.tree,type = "response"), labels = formatted.results)
gam.train.preds <- evalmod(scores = predict(object = gam.model,type = "response"), labels =  ifelse(training_data$Class == "Class2",1,0))
```


```r
formatted.results <-  ifelse(testing_data$Class == "Class2",1,0)
testing_data$labels =factor( predict(as.party(tree.for.feed), testing_data ,type = "node") )
simple.cart.test.eval <- evalmod(scores = predict(object = tree.for.feed,newdata = testing_data )[,2], labels = formatted.results)
glm.mod.test.eval <- evalmod(scores = predict(object = glm.mod,type = "response",newdata = testing_data), labels = formatted.results)
mixed.for.test.eval <- evalmod(scores = predict(object = glm.w.tree,type = "response",newdata = testing_data), labels = formatted.results)
glm.w.only.tree.test.for.eval<- evalmod(scores = predict(object = glm.w.only.tree,type = "response",newdata = testing_data), labels = formatted.results)
gam.test.preds <- evalmod(scores = predict(object = gam.model,newdata = testing_data,type = "response"), labels = formatted.results)
```


```r
format.auc.results <- function(x){
  return( data.table(   ROC.AUC = auc(x)[4][,1][1] , PR.AUC = auc(x)[4][,1][2]  )     )}
training_results <- lapply(X = list(simple.cart.eval,glm.mod.eval,mixed.for.eval,gam.train.preds),format.auc.results) %>% rbindlist
testing_results <- lapply(X = list(simple.cart.test.eval,glm.mod.test.eval,mixed.for.test.eval,gam.test.preds),format.auc.results) %>% rbindlist
names(training_results) <- paste("Training ", names(training_results),sep = "") %>% gsub(pattern = "\\.",replacement = " " )
names(testing_results) <- paste("Testing ", names(testing_results),sep = "") %>% gsub(pattern = "\\.",replacement = " " )
curve_results <- cbind(training_results,testing_results)
curve_results$Model <- c("CART","Logistic Regression", "Tree+Logistic Regression","GAM")
curve_results <- curve_results[,c(5,1,2,3,4)]
kable(curve_results)
```



|Model                    | Training ROC AUC| Training PR AUC| Testing ROC AUC| Testing PR AUC|
|:------------------------|----------------:|---------------:|---------------:|--------------:|
|CART                     |        0.6762069|       0.6189382|       0.6476973|      0.5873447|
|Logistic Regression      |        0.7623476|       0.7203193|       0.7355131|      0.6916757|
|Tree+Logistic Regression |        0.7635627|       0.7214351|       0.7341228|      0.6912453|
|GAM                      |        0.7685433|       0.7279184|       0.7368863|      0.6921735|

The results above show that the GAM model performs the best for all metrics for both the in-sample and out-of-sample sets. The mixed version of the model using CART and logistic regression performs marginally better than just using logistic regression. It's notable that the GAM outperforms the CART+logistic regression model by a greater margin than the CART+logistic regression model outperforms the logistic regression model. However, the overall effectiveness of the three models is still close:


```r
# Create information for multiplot:
# Training data
training.for.mmdata <- data.frame(predict(object = tree.for.feed )[,2],
                                  predict(object = glm.mod,type = "response"),
                                  predict(object = glm.w.tree,type = "response"),
                                  predict(object = gam.model,type = "response")  ) 
training_mdat <- mmdata(scores = training.for.mmdata,labels = ifelse(training_data$Class == "Class2",1,0),modnames = c("CART","Logistic Regression", "Tree w/ GLM", "GAM"))

testing.for.mmdata <- data.frame(predict(object = tree.for.feed,newdata = testing_data )[,2],
                                 predict(object = glm.mod,type = "response",newdata = testing_data),
                                 predict(object = glm.w.tree,type = "response",newdata = testing_data),
                                  predict(object = gam.model,newdata = testing_data,type = "response")  ) 
testing_mdat <- mmdata(scores = testing.for.mmdata,labels = ifelse(testing_data$Class == "Class2",1,0),modnames = c("CART","Logistic Regression", "Tree w/ GLM", "GAM"))
```


```r
autoplot(evalmod(training_mdat),curvetype = c("ROC"))+theme(legend.position = "bottom") +ggtitle("ROC Curve - Training Data")
```

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-10-1.png)


```r
autoplot(evalmod(training_mdat),curvetype = c("PR"))+theme(legend.position = "bottom") +ggtitle("Precision Recall - Training Data")
```

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-11-1.png)



```r
# Testing data
autoplot(evalmod(testing_mdat),curvetype = c("ROC")) +theme(legend.position = "bottom") +ggtitle("ROC Curve - Testing Data")
```

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-12-1.png)<!-- -->


```r
autoplot(evalmod(testing_mdat),curvetype = c("PR"))+theme(legend.position = "bottom") +ggtitle("Precision Recall - Testing Data")
```

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/figure-html/unnamed-chunk-13-1.png)<!-- -->

# Conclusion:

The justification for combining CART decision trees with logistic regression is that it makes the model more capable of coping with nonlinear independent variables. This issue can be addressed by using a generalized additive model instead because it allows for arbitrary smooth basis functions in the model equation. I used a synthetic data-set containing nonlinear features to fit and compare a logistic regression model, a CART+logistic regression model, and a GAM to compare the results. These results showed that there was a small different in performance between the logistic regression model and the CART+logistic regression model. GAM outperformed both. However, the margin of difference between the results based on ROC and PR AUC isn't large enough to say that any one methodology "failed". It does show that one can outperform CART+logistic with the more natural formulation of the GAM.

More investigation could be done on the topic, but I wanted to keep this terse enough to be easily covered in one blog post. Additional steps for fitting each model could have been taken to fine-tune their results. The models also weren't compared on a real-life data-set. Finally, it would be interesting to know how accurate the logistic regression model could be made with appropriate feature transformations.

# Further reading.

I'm summarizing some of the sources I used to review and prepare for this post:

**GAM**

-   Kim Larsen's [post](https://multithreaded.stitchfix.com/blog/2015/07/30/gam/) on the Stitch Fix blog has probably the best overall introduction to GAMs I've seen.

-   [ESL](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

-   Hastie and Tibshirani's [original paper](https://web.stanford.edu/~hastie/Papers/gam.pdf)

**Decision Trees + Logistic Regression**

-   SAS's [whitepaper](https://support.sas.com/resources/papers/proceedings/proceedings/sugi30/074-30.pdf)on the topic

-   [Using Classification Tree Outcomes to Enhance Logistic Regression Models](https://www.lexjansen.com/wuss/2004/data_analysis/c_das_using_classification_t.pdf)

-   The StackExchange [post](https://stackoverflow.com/questions/41692017/decision-trees-combined-with-logistic-regression) that inspired this post.

-   Andrzej Szymanski, PhD has a good [blog post](https://towardsdatascience.com/combining-logistic-regression-and-decision-tree-1adec36a4b3f) that this post was supposed to contrast. He uses a banking data-set for his numerical experiment. The results were similar to this post; a marginal improvement for tree+logistic regression over normal logistic regression.
