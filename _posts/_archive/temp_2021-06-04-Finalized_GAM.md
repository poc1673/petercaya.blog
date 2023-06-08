---
layout: post
title: "GAM and Trees"
date:   2021-05-21
categories: programming
---
 One common pitfall of building a statistical model is ensuring that the modeling method being used is appropriate to the structure of the data. The most obvious of these is the existence of nonlinear data in a data-set. Common strategies like logistic regression assume a linear relationship. One strategy I came across a few months ago in [this](https://stackoverflow.com/questions/41692017/decision-trees-combined-with-logistic-regression) article on StackExchange discussed the use of shallow decision trees as a feature engineering step for logistic regression. The justification for this strategy is that it uses trees as a feature engineering step to transform the nonlinear data into dummy variables that can then be used in the logistic regression specification.

After reading this and other posts, my impression was that performing this introductory step may introduce the drawback of reducing the flexibility of the model. Depending on the application, it may be more useful to either transform the data to remove the nonlinearities, or to account for them in the specification of the model. To investigate further, I decided to benchmark this methodology against logistic regression and decision trees alone. Additionally, I also decided to compare this strategy to an alternative that may be more structurally appropriate for including nonlinear relationships in a model.


# Motivation

Logistic regression and decision trees are generally the first two modeling methodologies one is introduced to for creating classification models. Each has its own pitfall. Regression models assume all variables are described by a linear function because the model takes the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i&space;x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i&space;x_i" title="\hat{y} = \beta_0 + \sum_{i=1}^K \beta_i x_i" /></a>

Decision trees don't make an assumption on the distribution of variables, they simply create a new branch in the decision tree based on a criteria like Gini impurity. However, decision trees are prone to overfitting on data and can be unstable to changes in the training data. The chart below illustrates the probability results for logistic regression and decision trees over the range of a dependent variable in a one variable model.

To try to remedy the downsides of these two methods, several sources have suggested using a decision tree as an intermediate step which helps remove potential nonlinearity in the model. In its most simple form, the process is as follows:
<script type="application/json" data-for="htmlwidget-5bbe4e3958604c4fb255">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"Data":["Adult","Adult","Adult","Banking","Banking","Banking","Synthetic","Synthetic","Synthetic"],"Model":["GAM","GLM + Tree","Logistic","GAM","GLM + Tree","Logistic","GAM","GLM + Tree","Logistic"],"In-Sample_ROC.AUC":[0.918691547544777,0.911856188271863,0.909657290793314,0.792889304268533,0.784127284964812,0.770466656009576,0.763191172281337,0.759465352175627,0.758834525823933],"In-Sample_PR.AUC":[0.798908275410571,0.779958817410901,0.771659003379546,0.452683537235139,0.441022359993118,0.43302213658738,0.783699853220669,0.78077234347527,0.780398639923388],"Out-of-Sample_ROC.AUC":[0.918691547544777,0.911856188271863,0.909657290793314,0.792889304268533,0.784127284964812,0.770466656009576,0.763191172281337,0.759465352175627,0.750122906426567],"Out-of-Sample_PR.AUC":[0.798908275410571,0.779958817410901,0.771659003379546,0.452683537235139,0.441022359993118,0.43302213658738,0.783699853220669,0.78077234347527,0.779401960629445]},"columns":[{"accessor":"Data","name":"Data","type":"character"},{"accessor":"Model","name":"Model","type":"character"},{"accessor":"In-Sample_ROC.AUC","name":"ROC.AUC","type":"numeric","format":{"cell":{"digits":3},"aggregated":{"digits":3}}},{"accessor":"In-Sample_PR.AUC","name":"PR.AUC","type":"numeric","format":{"cell":{"digits":3},"aggregated":{"digits":3}}},{"accessor":"Out-of-Sample_ROC.AUC","name":"ROC.AUC","type":"numeric","format":{"cell":{"digits":3},"aggregated":{"digits":3}}},{"accessor":"Out-of-Sample_PR.AUC","name":"PR.AUC","type":"numeric","format":{"cell":{"digits":3},"aggregated":{"digits":3}}}],"columnGroups":[{"name":"In-Sample","columns":["In-Sample_ROC.AUC","In-Sample_PR.AUC"]},{"name":"Out-of-Sample","columns":["Out-of-Sample_ROC.AUC","Out-of-Sample_PR.AUC"]}],"pivotBy":["Data"],"defaultSortDesc":true,"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"defaultExpanded":true,"dataKey":"9a12cb0b6d2ee574a378055d39edc78e","key":"9a12cb0b6d2ee574a378055d39edc78e"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>1.  Fit a shallow decision tree, <a href="https://www.codecogs.com/eqnedit.php?latex=T(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(X)" title="T(X)" /></a> for the training data, X. This tree will have N terminal nodes.
2.  N categorical variables denoted <a href="https://www.codecogs.com/eqnedit.php?latex=C_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_n" title="C_n" /></a> are included as features in the logistic regression specification.
3.  The logistic regression is fit using the modified set of data.

There are two simple alternatives to handling nonlinearity in data. The first choice is to simply transform the raw data into to attempt to give it a linear relationship with the dependent variable. This strategy isn't always an option, though, and its appropriateness varies based on the domain. The second way to eliminate it is to change the model specification to a method which can cope with nonlinear data.

There are two simple alternatives to handling nonlinearity in data. The first choice is to simply transform the raw data into to attempt to give it a linear relationship with the dependent variable. This strategy isn't always an option, though, and its appropriateness varies based on the domain. The second way to eliminate it is to change the model specification to a method which can cope with nonlinear data.

## Generalized additive models

There are several methodologies that can be used to cope with nonlinear data, but one the one I have chosen for this exercise is the generalized additive model. Generalized additive models are a framework originally introduced in "[Generalized Additive Models](https://web.stanford.edu/~hastie/Papers/gam.pdf)" by Trevor Hastie and Robert Tibshirani. The authors went on to write [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) which is where I first encountered it.

In brief, it takes the concept of the generalized linear model that logistic regression is based off of, and relaxes the assumption of linear basis functions. The results is a model specification like that described earlier.

The equation for the GAM equivalent for logistic regression would be:

<a href="https://www.codecogs.com/eqnedit.php?latex=ln\Big(&space;\frac{\mu(X)}{1-\mu(X)}&space;\Big)=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i(x_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln\Big(&space;\frac{\mu(X)}{1-\mu(X)}&space;\Big)=&space;\beta_0&space;&plus;&space;\sum_{i=1}^K&space;\beta_i(x_i)" title="ln\Big( \frac{\mu(X)}{1-\mu(X)} \Big)= \beta_0 + \sum_{i=1}^K \beta_i(x_i)" /></a>

The notation above substitutes <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i(\bullet)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i(\bullet)" title="\beta_i(\bullet)" /></a> for <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i" title="\beta_i" /></a> because we have changed the term operating on each dependent variable to an arbitrary smooth function. With this in mind, we could think of a logistic regression model as an additive model where:

<a href="https://www.codecogs.com/eqnedit.php?latex=\beta_i(x)&space;=&space;c_ix_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_i(x)&space;=&space;c_ix_i" title="\beta_i(x) = c_ix_i" /></a>

For some regression coefficient <a href="https://www.codecogs.com/eqnedit.php?latex=c_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_i" title="c_i" /></a>. It is clear that this offers us additional flexibility in how we treat the variables that we are modeling, so we will use this specification in our numerical experiment as a challenger to the tree+logistic regression specification.

# Data Analysis

I used three different data-sets to compare the three modeling methodologies. The first data-set is made up of synthetic data containing features which were linearly and nonlinearly related to the dependent variable. The second data set is the banking survey data-set available [here](https://github.com/AndrzejSzymanski/TDS/blob/master/banking.csv). I chose this data-set since it was used in a previous blog post on mixed tree and logistic regression methods available [here](https://towardsdatascience.com/combining-logistic-regression-and-decision-tree-1adec36a4b3f). The third data-set is the [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) data-set from UCI.



```r
data.list <- list("Synthetic" = with.nonlinear   ,
                  "Banking" =  fread("https://raw.githubusercontent.com/AndrzejSzymanski/TDS-LR-DT/master/banking_cleansed.csv")  ,
                  "Adult" = fread("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")  )
names(data.list[[3]]) <- c("age","workclass","fnlwgt","education","education-num","marital-status","occupation", "relationship", "race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income-class")

# Change variables to factors for banking data-set
banking.names.for.factors <- names(data.list$Banking)[apply(data.list$Banking,MARGIN = 2, function(x){length(unique(x))})==2]
data.list$Banking[,
                  names(data.list$Banking)[apply(data.list$Banking,MARGIN = 2, function(x){
                    length(unique(x))})==2] := lapply(X = .SD,FUN = factor),
                  .SDcols = banking.names.for.factors]

# Change variables to factors for Adult data-set:
adult.names.for.factors <- names(data.list$Adult)[sapply(X = 1:ncol(data.list$Adult),function(x){is.character( data.list$Adult[[x]]  )})]
data.list$Adult[,names(data.list$Adult)[sapply(X = 1:ncol(data.list$Adult),
                                                 function(x){is.character( data.list$Adult[[x]]  )})]:= lapply(X = .SD,FUN = factor),
                  .SDcols = adult.names.for.factors]


data.list$Adult[,names(data.list$Adult)[sapply(X = 1:ncol(data.list$Adult),
                                                 function(x){is.integer( data.list$Adult[[x]]  )})]:= lapply(X = .SD,FUN = as.numeric ),
                  .SDcols = names(data.list$Adult)[sapply(X = 1:ncol(data.list$Adult),
                                                 function(x){is.integer( data.list$Adult[[x]]  )})]]






training.data <- list()
test.data <- list()
for( i in 1:length(data.list)){
  train_inds <- sample(x = 1:nrow(data.list[[i]]) ,size = .8*nrow(data.list[[i]]))
  training.data[[i]] <- data.list[[i]][train_inds]
  test.data[[i]] <-   data.list[[i]][-train_inds]  }
names(training.data)<- names(data.list)
names(test.data)<- names(data.list)
```

The high level information on the data I'm testing is summarized below:


```r
sum.func <- function(x,col,model.name){ 
  dims <- data.table( "Measure" = c("Observations","Factors")  , "Count" = dim(x))
  factors <- data.table( "Minority Class Percent",  min(x[,.N,by = col]$N)/sum(x[,.N,by = col]$N)  )
  names(factors) <- c("Measure","Count")
  for.return <- data.table(rbind(dims,factors)   )
  names(for.return)[2] <- model.name
  # factors$Measure <- paste("Class = ",factors$Measure,sep = "")
  return(  for.return    )  }
dep.vars <- c("Class", "y", "Income-class")
summaries <- lapply(X = 1:length(training.data),
                    FUN = function(x){sum.func(training.data[[x]], dep.vars[x] ,
                                                                           model.name = names(training.data)[x])   })
summaries[[1]]$Banking <- summaries[[2]]$Banking
summaries[[1]]$Adult <- summaries[[3]]$Adult
kable(summaries[[1]],digits = 3)
```



|Measure                | Synthetic|   Banking|     Adult|
|:----------------------|---------:|---------:|---------:|
|Observations           |   8000.00| 32950.000| 26048.000|
|Factors                |     12.00|    29.000|    15.000|
|Minority Class Percent |      0.46|     0.114|     0.239|


Before engaging in our comparison of methods, it's useful to understand which variables have a nonlinear relationship with the dependent variable. One very simple way to do this is to chart the relationship between the dependent and independent variable to see if a nonlinear relationship is obvious to the eye of the reviewer. In the case of the implementation here, I am using a simple GAM with one variable which takes the form:

ADD IN INFORMATION ON HOW THE GAM FUNCTION WORKS HERE

To attempt to identify the relationship of each independent variable to the dependent variable, we plot the distribution of positive instances over the range of each variable. We will approximate the relationship between the dependent and independent variables using a smooth function to determine whether each relationship is linear or nonlinear below.


```r
source("plot_function.r")

training.data$Synthetic[,Class := ifelse(Class == "Class1", 1,0)]
test.data$Synthetic[,Class := ifelse(Class == "Class1", 1,0)]

training.data$Adult[,"Class" := ifelse(`Income-class` == ">50K", 1,0)]
test.data$Adult[,"Class" := ifelse(`Income-class` == ">50K", 1,0)]

synthetic.plot <- lapply(X = names(training.data$Synthetic)[names(training.data$Synthetic)!="Class"],
                         function(x){nonlinear.viz(dt = training.data$Synthetic,dep.var = "Class",x)})

banking.plot <- lapply(X = c("age","previous","euribor3m","cons_conf_idx","cons_price_idx","nr_employed", "emp_var_rate"   ,   "pdays"),
                         function(x){nonlinear.viz(dt = training.data$Banking,dep.var = "y",x)})

names(training.data$Adult) <- gsub(names(training.data$Adult),pattern = "-",replacement = "_")
names(test.data$Adult) <- names(training.data$Adult)

adult.plot <- lapply(X = c("age","education_num","capital_gain","capital_loss","hours_per_week"),
                         function(x){nonlinear.viz(dt = training.data$Adult,
                                                   dep.var = "Class",x )})
```
## Synthetic data-set

The synthetic data-set that's being used is the produce of the __ function from the caret library. This function is straightforward: A data-set is generated with a binary outcome and a set of variables which are either linearly related, or nonlinearly related. This data-set is useful for our tests because it allows us to compare the algorithms without being concerned with information regarding the domain of the problem. The relationship between variables is summarized in the table below:


| Feature    | Relationship |
|------------|--------------|
| Linear2    | Linear       |
| Linear3    | Linear       |
| Linear4    | Linear       |
| Linear5    | Linear       |
| Linear6    | Linear       |
| Nonlinear1 | Nonlinear    |
| Nonlinear2 | Nonlinear    |
| Nonlinear3 | Nonlinear            |

I've plotted the nonlinear variables for the synthetic data against the likelihood of th eclass in the data-set using the below:

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="210.33096mm"
   height="296.48361mm"
   viewBox="0 0 210.33097 296.48361"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="synthetic nonlinear.svg"
   inkscape:export-filename="C:\Users\USER\Dropbox\GAM versus tree plus logistic regression\Plots\Synthetic\synthetic nonlinear.png"
   inkscape:export-xdpi="96"
   inkscape:export-ydpi="96">
  <defs
     id="defs2">
    <rect
       x="2.9399648"
       y="4.8108516"
       width="200.71942"
       height="15.189149"
       id="rect875" />
    <rect
       x="3.7417734"
       y="3.474504"
       width="195.10677"
       height="16.525497"
       id="rect865" />
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath999">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path997" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1011">
      <path
         d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
         id="path1009" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1409">
      <path
         d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
         id="path1407" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1789">
      <path
         d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
         id="path1787" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4"
     inkscape:cx="158.88042"
     inkscape:cy="574.80771"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     inkscape:document-rotation="0"
     showgrid="false"
     showguides="true"
     inkscape:guide-bbox="true"
     inkscape:window-width="2880"
     inkscape:window-height="1526"
     inkscape:window-x="2869"
     inkscape:window-y="-11"
     inkscape:window-maximized="1">
    <sodipodi:guide
       position="-205.86615,289.6053"
       orientation="1,0"
       id="guide835" />
    <sodipodi:guide
       position="105.19867,301.09788"
       orientation="-1,0"
       id="guide837"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="0.1986676,281.09789"
       orientation="0,1"
       id="guide839"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="35.745515,97.097876"
       orientation="0,1"
       id="guide855"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="22.382038,6.0978755"
       orientation="0,1"
       id="guide859"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="229.2426,196.13094"
       orientation="0,1"
       id="guide861"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="-34.011834,163.59788"
       orientation="0,1"
       id="guide957"
       inkscape:label=""
       inkscape:locked="false"
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0.19866687,-4.6142695)">
    <text
       xml:space="preserve"
       id="text863"
       style="font-size:5.64444px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect865);" />
    <text
       xml:space="preserve"
       style="font-size:3.57804px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';stroke-width:0.264583"
       x="15.234363"
       y="10.690781"
       id="text871"><tspan
         sodipodi:role="line"
         id="tspan869"
         x="15.234363"
         y="10.690781"
         style="stroke-width:0.264583" /></text>
    <text
       xml:space="preserve"
       id="text873"
       style="font-size:9.83693px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect875);"
       transform="matrix(1.6900339,0,0,1.6102871,-3.4911386,-9.1477965)"><tspan
         x="2.9394531"
         y="13.053821"><tspan
           style="font-size:6.84308px">Nonlinear Variables for Synthetic Data</tspan><tspan
           style="shape-inside:url(#rect875)">
</tspan></tspan></text>
    <rect
       style="fill:#f9fff7;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="rect889-0"
       width="105"
       height="92"
       x="105"
       y="20" />
    <rect
       style="fill:#f9fff7;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="rect919"
       width="12.027128"
       height="1.925097e-12"
       x="133.10022"
       y="20" />
    <rect
       style="fill:#f9fff7;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="rect889-3"
       width="105"
       height="92"
       x="0"
       y="112" />
    <rect
       style="fill:#f9fff7;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="rect889-08"
       width="105"
       height="92"
       x="105"
       y="112" />
    <g
       id="g967"
       inkscape:label=" Nonlinear1 "
       transform="matrix(0.24445101,0,0,-0.18253968,-1.4043381,111.36216)">
      <g
         id="g969" />
      <g
         id="g971" />
      <g
         id="g973" />
      <g
         id="g975" />
      <g
         id="g977" />
      <g
         id="g979" />
      <g
         id="g981" />
      <g
         id="g983" />
      <g
         id="g985" />
      <g
         id="g987" />
      <g
         id="g989" />
      <g
         id="g991" />
      <g
         id="g993">
        <g
           id="g995"
           clip-path="url(#clipPath999)">
          <path
             d="M 21.771428,-3.4942256 H 885.77143 V 500.50577 H 21.771428 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1001" />
        </g>
      </g>
      <g
         id="g1003" />
      <g
         id="g1005">
        <g
           id="g1007"
           clip-path="url(#clipPath1011)">
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1013" />
          <path
             d="M 49.56,106.89 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1015" />
          <path
             d="M 49.56,192.99 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1017" />
          <path
             d="M 49.56,279.1 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1019" />
          <path
             d="M 49.56,365.21 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1021" />
          <path
             d="M 49.56,451.31 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1023" />
          <path
             d="m 178.07,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1025" />
          <path
             d="m 362.02,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1027" />
          <path
             d="m 545.96,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1029" />
          <path
             d="m 729.91,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1031" />
          <path
             d="M 49.56,63.84 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1033" />
          <path
             d="M 49.56,149.94 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1035" />
          <path
             d="M 49.56,236.05 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1037" />
          <path
             d="M 49.56,322.15 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1039" />
          <path
             d="M 49.56,408.26 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1041" />
          <path
             d="m 86.1,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1043" />
          <path
             d="m 270.04,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1045" />
          <path
             d="m 453.99,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1047" />
          <path
             d="m 637.93,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1049" />
          <path
             d="m 821.88,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1051" />
          <path
             d="m 86.34,393.02 9.3,-3.99 9.31,-3.99 9.31,-4 9.31,-3.99 9.31,-3.98 9.31,-3.97 9.31,-3.94 9.31,-3.92 9.31,-3.88 9.31,-3.85 9.3,-3.83 9.31,-3.83 9.31,-3.84 9.31,-3.88 9.31,-3.92 9.31,-3.99 9.31,-4.07 9.31,-4.17 9.31,-4.29 9.31,-4.42 9.31,-4.56 9.3,-4.73 9.31,-4.9 9.31,-5.09 9.31,-5.29 9.31,-5.5 9.31,-5.74 9.31,-5.96 9.31,-6.15 9.31,-6.32 9.31,-6.47 9.3,-6.6 9.31,-6.68 9.31,-6.76 9.31,-6.81 9.31,-6.82 9.31,-6.82 9.31,-6.78 9.31,-6.71 9.31,-6.63 9.31,-6.51 9.31,-6.37 9.3,-6.2 9.31,-6 9.31,-5.77 9.31,-5.52 9.31,-5.26 9.31,-4.96 9.31,-4.64 9.31,-4.3 9.31,-3.94 9.31,-3.55 9.3,-3.14 9.31,-2.73 9.31,-2.34 9.31,-1.96 9.31,-1.6 9.31,-1.23 9.31,-0.9 9.31,-0.57 9.31,-0.26 9.31,0.05 9.31,0.3 9.3,0.53 9.31,0.69 9.31,0.82 9.31,0.89 9.31,0.92 9.31,0.9 9.31,0.84 9.31,0.72 9.31,0.62 9.31,0.52 9.3,0.44 9.31,0.37 9.31,0.31 9.31,0.28 9.31,0.25 9.31,0.23"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1055" />
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1057" />
        </g>
      </g>
      <g
         id="g1059" />
      <g
         id="g1061" />
      <g
         id="g1063" />
      <g
         id="g1065" />
      <g
         id="g1067" />
      <g
         id="g1069" />
      <g
         id="g1071" />
      <g
         id="g1073" />
      <g
         id="g1075" />
      <g
         id="g1077" />
      <g
         id="g1079" />
      <g
         id="g1081" />
      <g
         id="g1083" />
      <g
         id="g1085" />
      <g
         id="g1087" />
      <g
         id="g1089">
        <text
           transform="matrix(1,0,0,-1,23.23,59.89)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1093"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1091">0.45</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,145.99)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1097"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1095">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,232.1)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1101"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1099">0.55</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,318.21)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1105"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1103">0.60</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,404.31)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1109"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1107">0.65</tspan></text>
      </g>
      <g
         id="g1111" />
      <g
         id="g1113">
        <path
           d="m 46.82,63.84 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1115" />
        <path
           d="m 46.82,149.94 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1117" />
        <path
           d="m 46.82,236.05 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1119" />
        <path
           d="m 46.82,322.15 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1121" />
        <path
           d="m 46.82,408.26 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1123" />
      </g>
      <g
         id="g1125" />
      <g
         id="g1127" />
      <g
         id="g1129" />
      <g
         id="g1131" />
      <g
         id="g1133" />
      <g
         id="g1135" />
      <g
         id="g1137" />
      <g
         id="g1139" />
      <g
         id="g1141" />
      <g
         id="g1143">
        <path
           d="m 86.1,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1145" />
        <path
           d="m 270.04,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1147" />
        <path
           d="m 453.99,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1149" />
        <path
           d="m 637.93,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1151" />
        <path
           d="m 821.88,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1153" />
      </g>
      <g
         id="g1155" />
      <g
         id="g1179" />
      <g
         id="g1181" />
      <g
         id="g1183" />
      <g
         id="g1185" />
      <g
         id="g1187" />
      <g
         id="g1195" />
      <g
         id="g1197">
        <text
           transform="matrix(0,1,1,0,16.97,217.7)"
           style="font-variant:normal;font-weight:normal;font-size:16.7004px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1201"><tspan
             x="0 11.552 20.448 29.344 32.896 36.448002 45.344002 54.240002 63.136002 68.463997"
             y="0"
             sodipodi:role="line"
             id="tspan1199"
             style="font-size:16.7004px">Nonlinear1</tspan></text>
      </g>
      <g
         id="g1203" />
      <g
         id="g1205" />
      <g
         id="g1207" />
      <g
         id="g1209" />
      <g
         id="g1211" />
      <g
         id="g1219" />
      <g
         id="g1221" />
      <g
         id="g1223" />
      <g
         id="g1225" />
      <g
         id="g1227" />
      <g
         id="g1229" />
    </g>
    <g
       id="g1371"
       inkscape:label=" Nonlinear2 "
       transform="matrix(0.2427778,0,0,-0.18200154,-0.06874581,203.64342)">
      <g
         id="g1373" />
      <g
         id="g1375" />
      <g
         id="g1377" />
      <g
         id="g1379" />
      <g
         id="g1381" />
      <g
         id="g1383" />
      <g
         id="g1385" />
      <g
         id="g1387" />
      <g
         id="g1389" />
      <g
         id="g1391" />
      <g
         id="g1393" />
      <g
         id="g1395" />
      <g
         id="g1397">
        <path
           d="M 0,0 H 864 V 504 H 0 Z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1399" />
      </g>
      <g
         id="g1401" />
      <g
         id="g1403">
        <g
           id="g1405"
           clip-path="url(#clipPath1409)">
          <path
             d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1411" />
          <path
             d="M 55.68,101.79 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1413" />
          <path
             d="M 55.68,210.25 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1415" />
          <path
             d="M 55.68,318.71 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1417" />
          <path
             d="M 55.68,427.18 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1419" />
          <path
             d="m 183.33,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1421" />
          <path
             d="m 365.82,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1423" />
          <path
             d="m 548.31,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1425" />
          <path
             d="m 730.8,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1427" />
          <path
             d="M 55.68,47.56 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1429" />
          <path
             d="M 55.68,156.02 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1431" />
          <path
             d="M 55.68,264.48 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1433" />
          <path
             d="M 55.68,372.95 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1435" />
          <path
             d="m 92.09,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1437" />
          <path
             d="m 274.58,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1439" />
          <path
             d="m 457.07,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1441" />
          <path
             d="m 639.55,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1443" />
          <path
             d="m 822.04,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1445" />
          <path
             d="m 92.17,343.54 9.24,-4.03 9.24,-4.03 9.24,-4.03 9.24,-4.03 9.24,-4.02 9.24,-4.03 9.23,-4.02 9.24,-4.02 9.24,-4.01 9.24,-4.01 9.24,-3.99 9.24,-3.99 9.24,-3.98 9.23,-3.96 9.24,-3.95 9.24,-3.94 9.24,-3.91 9.24,-3.89 9.24,-3.87 9.24,-3.85 9.24,-3.81 9.23,-3.76 9.24,-3.73 9.24,-3.67 9.24,-3.62 9.24,-3.55 9.24,-3.48 9.24,-3.42 9.23,-3.34 9.24,-3.25 9.24,-3.18 9.24,-3.09 9.24,-3 9.24,-2.9 9.24,-2.8 9.24,-2.71 9.23,-2.6 9.24,-2.5 9.24,-2.4 9.24,-2.29 9.24,-2.19 9.24,-2.09 9.24,-1.99 9.23,-1.88 9.24,-1.78 9.24,-1.67 9.24,-1.58 9.24,-1.48 9.24,-1.38 9.24,-1.28 9.24,-1.2 9.23,-1.1 9.24,-1.02 9.24,-0.92 9.24,-0.85 9.24,-0.76 9.24,-0.7 9.24,-0.62 9.23,-0.56 9.24,-0.5 9.24,-0.45 9.24,-0.39 9.24,-0.35 9.24,-0.31 9.24,-0.28 9.24,-0.24 9.23,-0.21 9.24,-0.19 9.24,-0.17 9.24,-0.16 9.24,-0.14 9.24,-0.14 9.24,-0.13 9.23,-0.12 9.24,-0.12 9.24,-0.11 9.24,-0.11 9.24,-0.1 9.24,-0.11"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1449" />
          <path
             d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1451" />
        </g>
      </g>
      <g
         id="g1453" />
      <g
         id="g1455" />
      <g
         id="g1457" />
      <g
         id="g1459" />
      <g
         id="g1461" />
      <g
         id="g1463" />
      <g
         id="g1465" />
      <g
         id="g1467" />
      <g
         id="g1469" />
      <g
         id="g1471" />
      <g
         id="g1473" />
      <g
         id="g1475" />
      <g
         id="g1477" />
      <g
         id="g1479" />
      <g
         id="g1481" />
      <g
         id="g1483">
        <text
           transform="matrix(1,0,0,-1,23.23,43.61)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1487"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1485">0.500</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,152.07)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1491"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1489">0.525</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,260.53)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1495"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1493">0.550</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,369)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1499"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1497">0.575</tspan></text>
      </g>
      <g
         id="g1501" />
      <g
         id="g1503">
        <path
           d="m 52.94,47.56 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1505" />
        <path
           d="m 52.94,156.02 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1507" />
        <path
           d="m 52.94,264.48 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1509" />
        <path
           d="m 52.94,372.95 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1511" />
      </g>
      <g
         id="g1513" />
      <g
         id="g1515" />
      <g
         id="g1517" />
      <g
         id="g1519" />
      <g
         id="g1521" />
      <g
         id="g1523" />
      <g
         id="g1525" />
      <g
         id="g1527" />
      <g
         id="g1529" />
      <g
         id="g1531">
        <path
           d="m 92.09,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1533" />
        <path
           d="m 274.58,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1535" />
        <path
           d="m 457.07,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1537" />
        <path
           d="m 639.55,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1539" />
        <path
           d="m 822.04,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1541" />
      </g>
      <g
         id="g1543" />
      <g
         id="g1567" />
      <g
         id="g1569" />
      <g
         id="g1571" />
      <g
         id="g1573" />
      <g
         id="g1575" />
      <g
         id="g1583" />
      <g
         id="g1585">
        <text
           transform="matrix(0,1,1,0,16.97,217.7)"
           style="font-variant:normal;font-weight:normal;font-size:16.7826px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1589"><tspan
             x="0 11.552 20.448 29.344 32.896 36.448002 45.344002 54.240002 63.136002 68.463997"
             y="0"
             sodipodi:role="line"
             id="tspan1587"
             style="font-size:16.7826px">Nonlinear2</tspan></text>
      </g>
      <g
         id="g1591" />
      <g
         id="g1593" />
      <g
         id="g1595" />
      <g
         id="g1597" />
      <g
         id="g1599" />
      <g
         id="g1607" />
      <g
         id="g1609" />
      <g
         id="g1611" />
      <g
         id="g1613" />
      <g
         id="g1615" />
      <g
         id="g1617" />
    </g>
    <g
       id="g1751"
       inkscape:label=" Nonlinear3 "
       transform="matrix(0.24284309,0,0,-0.18184135,-0.06874581,295.29146)">
      <g
         id="g1753" />
      <g
         id="g1755" />
      <g
         id="g1757" />
      <g
         id="g1759" />
      <g
         id="g1761" />
      <g
         id="g1763" />
      <g
         id="g1765" />
      <g
         id="g1767" />
      <g
         id="g1769" />
      <g
         id="g1771" />
      <g
         id="g1773" />
      <g
         id="g1775" />
      <g
         id="g1777">
        <path
           d="M 0,0 H 864 V 504 H 0 Z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1779" />
      </g>
      <g
         id="g1781" />
      <g
         id="g1783">
        <g
           id="g1785"
           clip-path="url(#clipPath1789)">
          <path
             d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1791" />
          <path
             d="M 55.68,80.14 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1793" />
          <path
             d="M 55.68,188.93 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1795" />
          <path
             d="M 55.68,297.73 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1797" />
          <path
             d="M 55.68,406.52 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1799" />
          <path
             d="m 183.37,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1801" />
          <path
             d="m 365.88,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1803" />
          <path
             d="m 548.4,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1805" />
          <path
             d="m 730.91,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1807" />
          <path
             d="M 55.68,134.54 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1809" />
          <path
             d="M 55.68,243.33 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1811" />
          <path
             d="M 55.68,352.12 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1813" />
          <path
             d="M 55.68,460.92 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1815" />
          <path
             d="m 92.11,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1817" />
          <path
             d="m 274.63,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1819" />
          <path
             d="m 457.14,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1821" />
          <path
             d="m 639.66,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1823" />
          <path
             d="m 822.17,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1825" />
          <path
             d="m 92.17,334.1 9.24,-4.42 9.24,-4.42 9.24,-4.44 9.24,-4.44 9.24,-4.46 9.24,-4.48 9.23,-4.49 9.24,-4.51 9.24,-4.54 9.24,-4.57 9.24,-4.58 9.24,-4.6 9.24,-4.62 9.23,-4.63 9.24,-4.63 9.24,-4.64 9.24,-4.64 9.24,-4.64 9.24,-4.64 9.24,-4.62 9.24,-4.61 9.23,-4.58 9.24,-4.56 9.24,-4.53 9.24,-4.49 9.24,-4.44 9.24,-4.4 9.24,-4.34 9.23,-4.27 9.24,-4.19 9.24,-4.09 9.24,-3.99 9.24,-3.87 9.24,-3.74 9.24,-3.6 9.24,-3.45 9.23,-3.3 9.24,-3.12 9.24,-2.94 9.24,-2.75 9.24,-2.56 9.24,-2.35 9.24,-2.13 9.23,-1.9 9.24,-1.68 9.24,-1.46 9.24,-1.25 9.24,-1.05 9.24,-0.87 9.24,-0.68 9.24,-0.52 9.23,-0.35 9.24,-0.2 9.24,-0.05 9.24,0.09 9.24,0.23 9.24,0.37 9.24,0.52 9.23,0.65 9.24,0.78 9.24,0.91 9.24,1.05 9.24,1.17 9.24,1.28 9.24,1.4 9.24,1.49 9.23,1.6 9.24,1.68 9.24,1.76 9.24,1.84 9.24,1.9 9.24,1.96 9.24,2.01 9.23,2.06 9.24,2.1 9.24,2.12 9.24,2.15 9.24,2.16 9.24,2.16"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1829" />
          <path
             d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1831" />
        </g>
      </g>
      <g
         id="g1833" />
      <g
         id="g1835" />
      <g
         id="g1837" />
      <g
         id="g1839" />
      <g
         id="g1841" />
      <g
         id="g1843" />
      <g
         id="g1845" />
      <g
         id="g1847" />
      <g
         id="g1849" />
      <g
         id="g1851" />
      <g
         id="g1853" />
      <g
         id="g1855" />
      <g
         id="g1857" />
      <g
         id="g1859" />
      <g
         id="g1861" />
      <g
         id="g1863">
        <text
           transform="matrix(1,0,0,-1,23.23,130.59)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1867"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1865">0.525</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,239.38)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1871"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1869">0.550</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,348.17)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1875"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1873">0.575</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,456.97)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1879"><tspan
             x="0 6.1160002 9.1739998 15.29 21.406"
             y="0"
             sodipodi:role="line"
             id="tspan1877">0.600</tspan></text>
      </g>
      <g
         id="g1881" />
      <g
         id="g1883">
        <path
           d="m 52.94,134.54 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1885" />
        <path
           d="m 52.94,243.33 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1887" />
        <path
           d="m 52.94,352.12 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1889" />
        <path
           d="m 52.94,460.92 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1891" />
      </g>
      <g
         id="g1893" />
      <g
         id="g1895" />
      <g
         id="g1897" />
      <g
         id="g1899" />
      <g
         id="g1901" />
      <g
         id="g1903" />
      <g
         id="g1905" />
      <g
         id="g1907" />
      <g
         id="g1909" />
      <g
         id="g1911">
        <path
           d="m 92.11,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1913" />
        <path
           d="m 274.63,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1915" />
        <path
           d="m 457.14,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1917" />
        <path
           d="m 639.66,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1919" />
        <path
           d="m 822.17,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1921" />
      </g>
      <g
         id="g1923" />
      <g
         id="g1925">
        <text
           transform="matrix(1,0,0,-1,81.41,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1929"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1927">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,263.92,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1933"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1931">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,446.44,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1937"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1935">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,628.95,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1941"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1939">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,811.47,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1945"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             sodipodi:role="line"
             id="tspan1943">1.00</tspan></text>
      </g>
      <g
         id="g1947" />
      <g
         id="g1949" />
      <g
         id="g1951" />
      <g
         id="g1953" />
      <g
         id="g1955" />
      <g
         id="g1957"
         transform="matrix(2.7470315,0,0,2.6132843,-822.53667,-54.940363)">
        <text
           transform="scale(0.89451884,-1.1179194)"
           style="font-variant:normal;font-weight:normal;font-size:14.3123px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:0.894519"
           id="text1961"
           x="488.82062"
           y="-8.0506697"><tspan
             x="488.82062"
             sodipodi:role="line"
             id="tspan1959"
             y="-8.0506697"
             style="stroke-width:0.894519">Class</tspan></text>
      </g>
      <g
         id="g1963" />
      <g
         id="g1965">
        <text
           transform="matrix(0,1,1,0,16.97,217.7)"
           style="font-variant:normal;font-weight:normal;font-size:16.7877px;font-family:Helvetica;-inkscape-font-specification:Helvetica;letter-spacing:0px;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1969"><tspan
             x="0 11.552 20.448 29.344 32.896 36.448002 45.344002 54.240002 63.136002 68.463997"
             y="0"
             sodipodi:role="line"
             id="tspan1967"
             style="font-size:16.7877px;">Nonlinear3</tspan></text>
      </g>
      <g
         id="g1971" />
      <g
         id="g1973" />
      <g
         id="g1975" />
      <g
         id="g1977" />
      <g
         id="g1979" />
      <g
         id="g1987" />
      <g
         id="g1989" />
      <g
         id="g1991" />
      <g
         id="g1993" />
      <g
         id="g1995" />
      <g
         id="g1997" />
    </g>
  </g>
</svg>

We can see that the variables are as we would expect: In all cases, there is a generally negative nonlinear relationship exists between the likelihood of an instance being a positive class, and the likelihood of a negative class.

## Banking data-set

The banking data-set contains __ different variables. Seven of these __ variables are continuous variables with the remaining __ being binary variables. The feature the relationship to the dependent variable are show in the table below:

| Feature        | Relationship |
|----------------|--------------|
| age            | Nonlinear    |
| cons_conf_idx  | Nonlinear    |
| cons_price_idx | Nonlinear    |
| emp_var_rate   | Nonlinear    |
| euribor3m      | Nonlinear    |
| nr_employed    | Nonlinear    |
| pdays          | Nonlinear    |


As with the synthetic data, I've also produced a set of tables showing the relationship between the continuous feature variables and the dependent variable (in this case, y):

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   width="296.05814mm"
   height="204.5423mm"
   viewBox="0 0 296.05814 204.5423"
   version="1.1"
   id="svg8">
  <defs
     id="defs2">
    <rect
       x="3.0749428"
       y="9.5960531"
       width="155.01633"
       height="17.060036"
       id="rect874" />
    <rect
       x="2.9399648"
       y="4.8108516"
       width="200.71942"
       height="15.189149"
       id="rect875" />
    <rect
       x="3.7417734"
       y="3.474504"
       width="195.10677"
       height="16.525497"
       id="rect865" />
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath920">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path918" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath932">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path930" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1336">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path1334" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1348">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path1346" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1743">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path1741" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1755">
      <path
         d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
         id="path1753" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2105">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path2103" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2117">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path2115" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2482">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path2480" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2494">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path2492" />
    </clipPath>
  </defs>
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     transform="translate(0.1322915,-11.113787)">
    <text
       xml:space="preserve"
       id="text863"
       style="font-size:5.64444px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect865);" />
    <text
       xml:space="preserve"
       style="font-size:3.57804px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';stroke-width:0.264583"
       x="15.234363"
       y="10.690781"
       id="text871"><tspan
         id="tspan869"
         x="15.234363"
         y="10.690781"
         style="stroke-width:0.264583" /></text>
    <text
       xml:space="preserve"
       id="text873"
       style="font-size:9.83693px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect875);"
       transform="matrix(1.6900339,0,0,1.6102871,-2.154791,-4.0696755)" />
    <text
       xml:space="preserve"
       id="text872"
       style="font-size:8.46667px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect874);"><tspan
         x="3.0742188"
         y="16.690709"><tspan
           style="shape-inside:url(#rect874)">Nonlinear Variables for Banking Data</tspan></tspan></text>
    <g
       id="g888"
       transform="matrix(0.17297209,0,0,-0.12641282,-0.1322915,90.368142)">
      <g
         id="g890" />
      <g
         id="g892" />
      <g
         id="g894" />
      <g
         id="g896" />
      <g
         id="g898" />
      <g
         id="g900" />
      <g
         id="g902" />
      <g
         id="g904" />
      <g
         id="g906" />
      <g
         id="g908" />
      <g
         id="g910" />
      <g
         id="g912" />
      <g
         id="g914"
         transform="matrix(0.52132547,0,0,0.35995501,0,322.58267)">
        <g
           id="g916"
           clip-path="url(#clipPath920)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path922" />
        </g>
      </g>
      <g
         id="g924" />
      <g
         id="g926">
        <g
           id="g928"
           clip-path="url(#clipPath932)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path934" />
          <path
             d="M 43.45,95 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path936" />
          <path
             d="M 43.45,203.05 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path938" />
          <path
             d="M 43.45,311.1 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path940" />
          <path
             d="M 43.45,419.15 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path942" />
          <path
             d="m 184.52,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path944" />
          <path
             d="m 328.01,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path946" />
          <path
             d="m 471.49,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path948" />
          <path
             d="m 614.98,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path950" />
          <path
             d="m 758.47,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path952" />
          <path
             d="M 43.45,40.97 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path954" />
          <path
             d="M 43.45,149.02 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path956" />
          <path
             d="M 43.45,257.08 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path958" />
          <path
             d="M 43.45,365.13 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path960" />
          <path
             d="M 43.45,473.18 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path962" />
          <path
             d="m 112.78,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path964" />
          <path
             d="m 256.26,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path966" />
          <path
             d="m 399.75,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path968" />
          <path
             d="m 543.24,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path970" />
          <path
             d="m 686.72,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path972" />
          <path
             d="m 830.21,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path974" />
          <path
             d="m 80.5,332.75 9.38,-18.35 9.38,-18.44 9.38,-18.43 9.38,-18.3 9.37,-18.05 9.38,-17.69 9.38,-17.21 9.38,-16.62 9.38,-15.92 9.38,-15.12 9.38,-14.22 9.38,-13.16 9.38,-11.93 9.38,-10.61 9.38,-9.27 9.38,-7.99 9.38,-6.78 9.38,-5.7 9.38,-4.72 9.38,-3.88 9.38,-3.14 9.37,-2.49 9.38,-1.91 9.38,-1.24 9.38,-0.48 9.38,0.47 9.38,1.69 9.38,3.46 9.38,6.29 9.38,11.19 9.38,20.09 9.38,35.2 9.38,54.05 9.38,68.28 9.38,67.74 9.38,53.06 9.38,34.56 9.38,17.2 9.37,0.98 9.38,-16.64 9.38,-39.17 9.38,-64.7 9.38,-78.82 9.38,-71.48 9.38,-49.77 9.38,-28.59 9.38,-14.29 9.38,-5.45 9.38,1.3 9.38,8.29 9.38,18.22 9.38,33.5 9.38,52.68 9.38,66.59 9.38,63.71 9.38,45.29 9.37,25.84 9.38,10.28 9.38,-1.91 9.38,-11.78 9.38,-19.71 9.38,-25.48 9.38,-28.51 9.38,-28.32 9.38,-25 9.38,-19.45 9.38,-14.02 9.38,-9.38 9.38,-5.42 9.38,-1.98 9.38,1.03 9.38,3.76 9.38,6.22 9.37,8.48 9.38,10.51 9.38,12.28 9.38,13.75 9.38,14.82 9.38,15.47"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path978" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path980" />
        </g>
      </g>
      <g
         id="g982" />
      <g
         id="g984" />
      <g
         id="g986" />
      <g
         id="g988" />
      <g
         id="g990" />
      <g
         id="g992" />
      <g
         id="g994" />
      <g
         id="g996" />
      <g
         id="g998" />
      <g
         id="g1000" />
      <g
         id="g1002" />
      <g
         id="g1004" />
      <g
         id="g1006" />
      <g
         id="g1008" />
      <g
         id="g1010" />
      <g
         id="g1012">
        <text
           transform="matrix(1,0,0,-1,23.23,37.02)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1016"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1014">0.0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,145.07)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1020"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1018">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,253.13)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1024"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1022">0.4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,361.18)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1028"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1026">0.6</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,469.23)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1032"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1030">0.8</tspan></text>
      </g>
      <g
         id="g1034" />
      <g
         id="g1036">
        <path
           d="m 40.71,40.97 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1038" />
        <path
           d="m 40.71,149.02 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1040" />
        <path
           d="m 40.71,257.08 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1042" />
        <path
           d="m 40.71,365.13 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1044" />
        <path
           d="m 40.71,473.18 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1046" />
      </g>
      <g
         id="g1048" />
      <g
         id="g1050" />
      <g
         id="g1052" />
      <g
         id="g1054" />
      <g
         id="g1056" />
      <g
         id="g1058" />
      <g
         id="g1060" />
      <g
         id="g1062" />
      <g
         id="g1064" />
      <g
         id="g1066">
        <path
           d="m 112.78,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1068" />
        <path
           d="m 256.26,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1070" />
        <path
           d="m 399.75,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1072" />
        <path
           d="m 543.24,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1074" />
        <path
           d="m 686.72,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1076" />
        <path
           d="m 830.21,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1078" />
      </g>
      <g
         id="g1080" />
      <g
         id="g1082">
        <text
           transform="matrix(1,0,0,-1,106.51,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1086"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1084">−2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,249.99,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1090"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1088">−1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,396.69,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1094"><tspan
             x="0"
             y="0"
             id="tspan1092">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,540.18,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1098"><tspan
             x="0"
             y="0"
             id="tspan1096">1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,683.66,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1102"><tspan
             x="0"
             y="0"
             id="tspan1100">2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,827.15,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1106"><tspan
             x="0"
             y="0"
             id="tspan1104">3</tspan></text>
      </g>
      <g
         id="g1108" />
      <g
         id="g1110" />
      <g
         id="g1112" />
      <g
         id="g1114" />
      <g
         id="g1116" />
      <g
         id="g1118">
        <text
           transform="matrix(1,0,0,-1,446.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1122"><tspan
             x="0"
             y="0"
             id="tspan1120">y</tspan></text>
      </g>
      <g
         id="g1124" />
      <g
         id="g1126">
        <text
           transform="matrix(0,1,1,0,16.97,205.24)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1130"><tspan
             x="0 8 16.896 25.792 33.792 42.688 50.688 59.584 68.480003 72.928001 81.823997 85.375999 94.272003"
             y="0"
             id="tspan1128">cons_conf_idx</tspan></text>
      </g>
      <g
         id="g1132" />
      <g
         id="g1134" />
      <g
         id="g1136" />
      <g
         id="g1138" />
      <g
         id="g1140" />
      <g
         id="g1142">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1146"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 206.56 212.12 223.24001 234.36 245.48 251.03999 261.04001 272.16 283.28 293.28 304.39999 314.39999 325.51999 336.64001 342.20001 353.32001 357.76001 368.88"
             y="0"
             id="tspan1144">Relationship between y and cons_conf_idx</tspan></text>
      </g>
      <g
         id="g1148" />
      <g
         id="g1150" />
      <g
         id="g1152" />
      <g
         id="g1154" />
      <g
         id="g1156" />
      <g
         id="g1158" />
    </g>
    <g
       id="g1304"
       transform="matrix(0.171875,0,0,-0.125,-0.1322915,215.65609)">
      <g
         id="g1306" />
      <g
         id="g1308" />
      <g
         id="g1310" />
      <g
         id="g1312" />
      <g
         id="g1314" />
      <g
         id="g1316" />
      <g
         id="g1318" />
      <g
         id="g1320" />
      <g
         id="g1322" />
      <g
         id="g1324" />
      <g
         id="g1326" />
      <g
         id="g1328" />
      <g
         id="g1330">
        <g
           id="g1332"
           clip-path="url(#clipPath1336)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1338" />
        </g>
      </g>
      <g
         id="g1340" />
      <g
         id="g1342">
        <g
           id="g1344"
           clip-path="url(#clipPath1348)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1350" />
          <path
             d="M 43.45,61.49 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1352" />
          <path
             d="M 43.45,153.13 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1354" />
          <path
             d="M 43.45,244.77 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1356" />
          <path
             d="M 43.45,336.41 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1358" />
          <path
             d="M 43.45,428.04 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1360" />
          <path
             d="m 59.59,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1362" />
          <path
             d="m 226.73,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1364" />
          <path
             d="m 393.88,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1366" />
          <path
             d="m 561.03,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1368" />
          <path
             d="m 728.18,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1370" />
          <path
             d="M 43.45,107.31 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1372" />
          <path
             d="M 43.45,198.95 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1374" />
          <path
             d="M 43.45,290.59 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1376" />
          <path
             d="M 43.45,382.22 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1378" />
          <path
             d="M 43.45,473.86 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1380" />
          <path
             d="m 143.16,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1382" />
          <path
             d="m 310.31,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1384" />
          <path
             d="m 477.45,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1386" />
          <path
             d="m 644.6,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1388" />
          <path
             d="m 811.75,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1390" />
          <path
             d="m 80.5,319.32 9.38,20.95 9.38,20.65 9.38,19.25 9.38,16.72 9.37,13.04 9.38,8.26 9.38,2.43 9.38,-4.4 9.38,-11.75 9.38,-17.97 9.38,-22.59 9.38,-25.53 9.38,-26.79 9.38,-26.49 9.38,-24.84 9.38,-22.15 9.38,-18.71 9.38,-15.53 9.38,-13.28 9.38,-11.85 9.38,-11.07 9.37,-10.86 9.38,-10.92 9.38,-10.67 9.38,-10.03 9.38,-9.1 9.38,-7.98 9.38,-6.74 9.38,-5.42 9.38,-4.04 9.38,-2.61 9.38,-1.11 9.38,0.26 9.38,1.39 9.38,2.28 9.38,2.99 9.38,3.47 9.38,3.69 9.37,3.61 9.38,3.19 9.38,2.38 9.38,1.19 9.38,-0.38 9.38,-2.27 9.38,-4.34 9.38,-6.45 9.38,-8.37 9.38,-9.15 9.38,-8.16 9.38,-6.03 9.38,-3.18 9.38,0.38 9.38,5.19 9.38,10.57 9.38,15.12 9.38,17.36 9.37,14.91 9.38,8.48 9.38,2.03 9.38,-3.1 9.38,-6.4 9.38,-7.78 9.38,-7.54 9.38,-6.01 9.38,-3.91 9.38,-2.07 9.38,-0.46 9.38,0.99 9.38,2.38 9.38,3.73 9.38,5.09 9.38,6.48 9.38,7.93 9.37,9.42 9.38,10.98 9.38,12.55 9.38,14.11 9.38,15.62 9.38,17"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1394" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1396" />
        </g>
      </g>
      <g
         id="g1398" />
      <g
         id="g1400" />
      <g
         id="g1402" />
      <g
         id="g1404" />
      <g
         id="g1406" />
      <g
         id="g1408" />
      <g
         id="g1410" />
      <g
         id="g1412" />
      <g
         id="g1414" />
      <g
         id="g1416" />
      <g
         id="g1418" />
      <g
         id="g1420" />
      <g
         id="g1422" />
      <g
         id="g1424" />
      <g
         id="g1426" />
      <g
         id="g1428">
        <text
           transform="matrix(1,0,0,-1,23.23,103.36)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1432"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1430">0.1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,195)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1436"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1434">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,286.64)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1440"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1438">0.3</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,378.28)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1444"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1442">0.4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,469.91)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1448"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1446">0.5</tspan></text>
      </g>
      <g
         id="g1450" />
      <g
         id="g1452">
        <path
           d="m 40.71,107.31 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1454" />
        <path
           d="m 40.71,198.95 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1456" />
        <path
           d="m 40.71,290.59 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1458" />
        <path
           d="m 40.71,382.22 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1460" />
        <path
           d="m 40.71,473.86 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1462" />
      </g>
      <g
         id="g1464" />
      <g
         id="g1466" />
      <g
         id="g1468" />
      <g
         id="g1470" />
      <g
         id="g1472" />
      <g
         id="g1474" />
      <g
         id="g1476" />
      <g
         id="g1478" />
      <g
         id="g1480" />
      <g
         id="g1482">
        <path
           d="m 143.16,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1484" />
        <path
           d="m 310.31,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1486" />
        <path
           d="m 477.45,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1488" />
        <path
           d="m 644.6,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1490" />
        <path
           d="m 811.75,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1492" />
      </g>
      <g
         id="g1494" />
      <g
         id="g1496">
        <text
           transform="matrix(1,0,0,-1,136.89,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1500"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1498">−2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,304.04,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1504"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1502">−1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,474.4,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1508"><tspan
             x="0"
             y="0"
             id="tspan1506">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,641.54,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1512"><tspan
             x="0"
             y="0"
             id="tspan1510">1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,808.69,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1516"><tspan
             x="0"
             y="0"
             id="tspan1514">2</tspan></text>
      </g>
      <g
         id="g1518" />
      <g
         id="g1520" />
      <g
         id="g1522" />
      <g
         id="g1524" />
      <g
         id="g1526" />
      <g
         id="g1528">
        <text
           transform="matrix(1,0,0,-1,446.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1532"><tspan
             x="0"
             y="0"
             id="tspan1530">y</tspan></text>
      </g>
      <g
         id="g1534" />
      <g
         id="g1536">
        <text
           transform="matrix(0,1,1,0,16.97,202.91)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1540"><tspan
             x="0 8 16.896 25.792 33.792 42.688 51.584 57.152 60.703999 68.704002 77.599998 86.496002 90.047997 98.944"
             y="0"
             id="tspan1538">cons_price_idx</tspan></text>
      </g>
      <g
         id="g1542" />
      <g
         id="g1544" />
      <g
         id="g1546" />
      <g
         id="g1548" />
      <g
         id="g1550" />
      <g
         id="g1552">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1556"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 206.56 212.12 223.24001 234.36 245.48 251.03999 261.04001 272.16 283.28 293.28 304.39999 315.51999 322.48001 326.92001 336.92001 348.04001 359.16 363.60001 374.72"
             y="0"
             id="tspan1554">Relationship between y and cons_price_idx</tspan></text>
      </g>
      <g
         id="g1558" />
      <g
         id="g1560" />
      <g
         id="g1562" />
      <g
         id="g1564" />
      <g
         id="g1566" />
      <g
         id="g1568" />
    </g>
    <g
       id="g1711"
       transform="matrix(0.171875,0,0,-0.125,-0.1322915,152.65608)">
      <g
         id="g1713" />
      <g
         id="g1715" />
      <g
         id="g1717" />
      <g
         id="g1719" />
      <g
         id="g1721" />
      <g
         id="g1723" />
      <g
         id="g1725" />
      <g
         id="g1727" />
      <g
         id="g1729" />
      <g
         id="g1731" />
      <g
         id="g1733" />
      <g
         id="g1735" />
      <g
         id="g1737">
        <g
           id="g1739"
           clip-path="url(#clipPath1743)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1745" />
        </g>
      </g>
      <g
         id="g1747" />
      <g
         id="g1749">
        <g
           id="g1751"
           clip-path="url(#clipPath1755)">
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1757" />
          <path
             d="M 49.56,111.87 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1759" />
          <path
             d="M 49.56,223.78 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1761" />
          <path
             d="M 49.56,335.69 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1763" />
          <path
             d="M 49.56,447.6 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1765" />
          <path
             d="m 258.77,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1767" />
          <path
             d="m 499.46,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1769" />
          <path
             d="m 740.14,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1771" />
          <path
             d="M 49.56,55.92 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1773" />
          <path
             d="M 49.56,167.83 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1775" />
          <path
             d="M 49.56,279.74 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1777" />
          <path
             d="M 49.56,391.64 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1779" />
          <path
             d="m 138.43,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1781" />
          <path
             d="m 379.11,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1783" />
          <path
             d="m 619.8,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1785" />
          <path
             d="m 86.34,247.17 9.3,24.59 9.31,22.11 9.31,16.58 9.31,8.46 9.31,-2.02 9.31,-15.27 9.31,-31.48 9.31,-44.27 9.31,-45.25 9.31,-38.64 9.3,-29.31 9.31,-20.26 9.31,-13.13 9.31,-8.18 9.31,-4.98 9.31,-3.01 9.31,-1.8 9.31,-1.02 9.31,-0.49 9.31,-0.07 9.31,0.39 9.3,1.06 9.31,2.32 9.31,5.08 9.31,11.93 9.31,30.04 9.31,72.5 9.31,109.46 9.31,81.59 9.31,41.74 9.31,17.76 9.3,4.08 9.31,-5.68 9.31,-15.07 9.31,-25.52 9.31,-36.45 9.31,-44.91 9.31,-47.06 9.31,-42.49 9.31,-35.3 9.31,-27.82 9.31,-21.19 9.3,-15.81 9.31,-11.69 9.31,-8.58 9.31,-6.27 9.31,-4.52 9.31,-3.21 9.31,-2.15 9.31,-1.29 9.31,-0.55 9.31,0.15 9.3,0.83 9.31,1.25 9.31,0.89 9.31,0.23 9.31,-0.38 9.31,-0.88 9.31,-1.27 9.31,-1.55 9.31,-1.7 9.31,-1.75 9.31,-1.72 9.3,-1.62 9.31,-1.48 9.31,-1.31 9.31,-1.11 9.31,-0.91 9.31,-0.7 9.31,-0.47 9.31,-0.23 9.31,0.03 9.31,0.32 9.3,0.69 9.31,1.12 9.31,1.58 9.31,2.07 9.31,2.57 9.31,3.04"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1789" />
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1791" />
        </g>
      </g>
      <g
         id="g1793" />
      <g
         id="g1795" />
      <g
         id="g1797" />
      <g
         id="g1799" />
      <g
         id="g1801" />
      <g
         id="g1803" />
      <g
         id="g1805" />
      <g
         id="g1807" />
      <g
         id="g1809" />
      <g
         id="g1811" />
      <g
         id="g1813" />
      <g
         id="g1815" />
      <g
         id="g1817" />
      <g
         id="g1819" />
      <g
         id="g1821" />
      <g
         id="g1823">
        <text
           transform="matrix(1,0,0,-1,23.23,51.97)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1827"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan1825">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,163.88)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1831"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan1829">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,275.79)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1835"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan1833">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,387.7)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1839"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan1837">0.75</tspan></text>
      </g>
      <g
         id="g1841" />
      <g
         id="g1843">
        <path
           d="m 46.82,55.92 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1845" />
        <path
           d="m 46.82,167.83 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1847" />
        <path
           d="m 46.82,279.74 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1849" />
        <path
           d="m 46.82,391.64 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1851" />
      </g>
      <g
         id="g1853" />
      <g
         id="g1855" />
      <g
         id="g1857" />
      <g
         id="g1859" />
      <g
         id="g1861" />
      <g
         id="g1863" />
      <g
         id="g1865" />
      <g
         id="g1867" />
      <g
         id="g1869" />
      <g
         id="g1871">
        <path
           d="m 138.43,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1873" />
        <path
           d="m 379.11,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1875" />
        <path
           d="m 619.8,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1877" />
      </g>
      <g
         id="g1879" />
      <g
         id="g1881">
        <text
           transform="matrix(1,0,0,-1,132.16,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1885"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1883">−2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,372.84,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1889"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan1887">−1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,616.74,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1893"><tspan
             x="0"
             y="0"
             id="tspan1891">0</tspan></text>
      </g>
      <g
         id="g1895" />
      <g
         id="g1897" />
      <g
         id="g1899" />
      <g
         id="g1901" />
      <g
         id="g1903" />
      <g
         id="g1905">
        <text
           transform="matrix(1,0,0,-1,450.04,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1909"><tspan
             x="0"
             y="0"
             id="tspan1907">y</tspan></text>
      </g>
      <g
         id="g1911" />
      <g
         id="g1913">
        <text
           transform="matrix(0,1,1,0,16.97,207.31)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1917"><tspan
             x="0 8.8959999 22.224001 31.120001 40.015999 47.616001 56.512001 61.84 70.736 75.903999 84.800003 89.248001"
             y="0"
             id="tspan1915">emp_var_rate</tspan></text>
      </g>
      <g
         id="g1919" />
      <g
         id="g1921" />
      <g
         id="g1923" />
      <g
         id="g1925" />
      <g
         id="g1927" />
      <g
         id="g1929">
        <text
           transform="matrix(1,0,0,-1,49.56,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1933"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 206.56 212.12 223.24001 234.36 245.48 251.03999 262.16 278.82001 289.94 301.06 310.56 321.67999 328.34 339.45999 345.92001 357.04001 362.60001"
             y="0"
             id="tspan1931">Relationship between y and emp_var_rate</tspan></text>
      </g>
      <g
         id="g1935" />
      <g
         id="g1937" />
      <g
         id="g1939" />
      <g
         id="g1941" />
      <g
         id="g1943" />
      <g
         id="g1945" />
    </g>
    <g
       id="g2073"
       transform="matrix(0.171875,0,0,-0.125,147.42583,117.68858)">
      <g
         id="g2075" />
      <g
         id="g2077" />
      <g
         id="g2079" />
      <g
         id="g2081" />
      <g
         id="g2083" />
      <g
         id="g2085" />
      <g
         id="g2087" />
      <g
         id="g2089" />
      <g
         id="g2091" />
      <g
         id="g2093" />
      <g
         id="g2095" />
      <g
         id="g2097" />
      <g
         id="g2099">
        <g
           id="g2101"
           clip-path="url(#clipPath2105)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2107" />
        </g>
      </g>
      <g
         id="g2109" />
      <g
         id="g2111">
        <g
           id="g2113"
           clip-path="url(#clipPath2117)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path2119" />
          <path
             d="M 43.45,71.43 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2121" />
          <path
             d="M 43.45,179.59 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2123" />
          <path
             d="M 43.45,287.76 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2125" />
          <path
             d="M 43.45,395.93 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2127" />
          <path
             d="m 195.79,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2129" />
          <path
             d="m 386.45,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2131" />
          <path
             d="m 577.11,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2133" />
          <path
             d="m 767.77,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2135" />
          <path
             d="M 43.45,125.51 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2137" />
          <path
             d="M 43.45,233.68 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2139" />
          <path
             d="M 43.45,341.84 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2141" />
          <path
             d="M 43.45,450.01 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2143" />
          <path
             d="m 100.46,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2145" />
          <path
             d="m 291.12,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2147" />
          <path
             d="m 481.78,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2149" />
          <path
             d="m 672.44,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2151" />
          <path
             d="m 80.5,257.66 9.38,-24.3 9.38,-23.19 9.38,-21.6 9.38,-19.65 9.37,-17.48 9.38,-15.26 9.38,-13.06 9.38,-10.97 9.38,-9.04 9.38,-7.39 9.38,-6.04 9.38,-4.95 9.38,-4.07 9.38,-3.34 9.38,-2.77 9.38,-2.3 9.38,-1.93 9.38,-1.63 9.38,-1.39 9.38,-1.16 9.38,-0.99 9.37,-0.84 9.38,-0.71 9.38,-0.61 9.38,-0.54 9.38,-0.48 9.38,-0.38 9.38,-0.23 9.38,-0.03 9.38,0.21 9.38,0.52 9.38,0.88 9.38,1.31 9.38,1.84 9.38,2.45 9.38,3.19 9.38,4.05 9.38,5.11 9.37,6.4 9.38,7.97 9.38,9.9 9.38,12.27 9.38,14.95 9.38,17.52 9.38,19.71 9.38,21.2 9.38,21.72 9.38,21.11 9.38,19.27 9.38,16.27 9.38,12.42 9.38,8.76 9.38,5.63 9.38,3.07 9.38,1.07 9.38,-0.37 9.37,-1.23 9.38,-1.54 9.38,-1.3 9.38,-0.87 9.38,-0.5 9.38,-0.22 9.38,-0.02 9.38,0.1 9.38,0.16 9.38,0.11 h 9.38 l 9.38,-0.16 9.38,-0.32 9.38,-0.47 9.38,-0.59 9.38,-0.7 9.38,-0.81 9.37,-0.89 9.38,-0.97 9.38,-1.02 9.38,-1.06 9.38,-1.1 9.38,-1.1"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2155" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2157" />
        </g>
      </g>
      <g
         id="g2159" />
      <g
         id="g2161" />
      <g
         id="g2163" />
      <g
         id="g2165" />
      <g
         id="g2167" />
      <g
         id="g2169" />
      <g
         id="g2171" />
      <g
         id="g2173" />
      <g
         id="g2175" />
      <g
         id="g2177" />
      <g
         id="g2179" />
      <g
         id="g2181" />
      <g
         id="g2183" />
      <g
         id="g2185" />
      <g
         id="g2187" />
      <g
         id="g2189">
        <text
           transform="matrix(1,0,0,-1,23.23,121.56)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2193"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2191">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,229.73)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2197"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2195">0.4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,337.89)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2201"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2199">0.6</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,446.06)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2205"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2203">0.8</tspan></text>
      </g>
      <g
         id="g2207" />
      <g
         id="g2209">
        <path
           d="m 40.71,125.51 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2211" />
        <path
           d="m 40.71,233.68 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2213" />
        <path
           d="m 40.71,341.84 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2215" />
        <path
           d="m 40.71,450.01 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2217" />
      </g>
      <g
         id="g2219" />
      <g
         id="g2221" />
      <g
         id="g2223" />
      <g
         id="g2225" />
      <g
         id="g2227" />
      <g
         id="g2229" />
      <g
         id="g2231" />
      <g
         id="g2233" />
      <g
         id="g2235" />
      <g
         id="g2237">
        <path
           d="m 100.46,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2239" />
        <path
           d="m 291.12,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2241" />
        <path
           d="m 481.78,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2243" />
        <path
           d="m 672.44,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2245" />
      </g>
      <g
         id="g2247" />
      <g
         id="g2249">
        <text
           transform="matrix(1,0,0,-1,94.19,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2253"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan2251">−2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,288.06,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2257"><tspan
             x="0"
             y="0"
             id="tspan2255">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,478.72,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2261"><tspan
             x="0"
             y="0"
             id="tspan2259">2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,669.38,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2265"><tspan
             x="0"
             y="0"
             id="tspan2263">4</tspan></text>
      </g>
      <g
         id="g2267" />
      <g
         id="g2269" />
      <g
         id="g2271" />
      <g
         id="g2273" />
      <g
         id="g2275" />
      <g
         id="g2277">
        <text
           transform="matrix(1,0,0,-1,446.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2281"><tspan
             x="0"
             y="0"
             id="tspan2279">y</tspan></text>
      </g>
      <g
         id="g2283" />
      <g
         id="g2285">
        <text
           transform="matrix(0,1,1,0,16.97,243.03)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2289"><tspan
             x="0 8.8959999 17.792"
             y="0"
             id="tspan2287">age</tspan></text>
      </g>
      <g
         id="g2291" />
      <g
         id="g2293" />
      <g
         id="g2295" />
      <g
         id="g2297" />
      <g
         id="g2299" />
      <g
         id="g2301">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2305"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 206.56 212.12 223.24001 234.36 245.48 251.03999 262.16 273.28"
             y="0"
             id="tspan2303">Relationship between y and age</tspan></text>
      </g>
      <g
         id="g2307" />
      <g
         id="g2309" />
      <g
         id="g2311" />
      <g
         id="g2313" />
      <g
         id="g2315" />
      <g
         id="g2317" />
    </g>
    <g
       id="g2450"
       transform="matrix(0.171875,0,0,-0.125,147.42583,184.65609)">
      <g
         id="g2452" />
      <g
         id="g2454" />
      <g
         id="g2456" />
      <g
         id="g2458" />
      <g
         id="g2460" />
      <g
         id="g2462" />
      <g
         id="g2464" />
      <g
         id="g2466" />
      <g
         id="g2468" />
      <g
         id="g2470" />
      <g
         id="g2472" />
      <g
         id="g2474" />
      <g
         id="g2476">
        <g
           id="g2478"
           clip-path="url(#clipPath2482)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2484" />
        </g>
      </g>
      <g
         id="g2486" />
      <g
         id="g2488">
        <g
           id="g2490"
           clip-path="url(#clipPath2494)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path2496" />
          <path
             d="M 43.45,111.55 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2498" />
          <path
             d="M 43.45,219 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2500" />
          <path
             d="M 43.45,326.46 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2502" />
          <path
             d="M 43.45,433.91 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2504" />
          <path
             d="m 144.4,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2506" />
          <path
             d="m 346.8,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2508" />
          <path
             d="m 549.2,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2510" />
          <path
             d="m 751.61,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2512" />
          <path
             d="M 43.45,57.83 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2514" />
          <path
             d="M 43.45,165.28 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2516" />
          <path
             d="M 43.45,272.73 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2518" />
          <path
             d="M 43.45,380.18 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2520" />
          <path
             d="m 245.6,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2522" />
          <path
             d="m 448,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2524" />
          <path
             d="m 650.41,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2526" />
          <path
             d="m 852.81,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2528" />
          <path
             d="m 80.5,313.68 9.38,3.36 9.38,3.4 9.38,3.46 9.38,3.58 9.37,3.71 9.38,3.88 9.38,4.09 9.38,4.32 9.38,4.59 9.38,4.56 9.38,3.14 9.38,0.15 9.38,-4.45 9.38,-10.69 9.38,-17.45 9.38,-19.19 9.38,-14.32 9.38,-6.03 9.38,0.88 9.38,6.14 9.38,9.77 9.37,11.74 9.38,11.99 9.38,10.46 9.38,7.19 9.38,2.7 9.38,-1.68 9.38,-5.69 9.38,-9.38 9.38,-12.69 9.38,-15.57 9.38,-17.93 9.38,-19.64 9.38,-20.64 9.38,-20.87 9.38,-20.38 9.38,-19.25 9.38,-17.63 9.37,-15.81 9.38,-13.94 9.38,-12.13 9.38,-10.42 9.38,-8.86 9.38,-7.47 9.38,-6.25 9.38,-5.2 9.38,-4.3 9.38,-3.54 9.38,-2.92 9.38,-2.39 9.38,-1.96 9.38,-1.61 9.38,-1.32 9.38,-1.08 9.38,-0.88 9.38,-0.73 9.37,-0.59 9.38,-0.45 9.38,-0.32 9.38,-0.19 9.38,-0.04 9.38,0.11 9.38,0.32 9.38,0.62 9.38,1.12 9.38,2 9.38,3.69 9.38,6.84 9.38,10.65 9.38,12.42 9.38,11.22 9.38,7.86 9.38,3.38 9.37,-1.07 9.38,-4.63 9.38,-6.88 9.38,-7.76 9.38,-7.58 9.38,-6.68"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2532" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2534" />
        </g>
      </g>
      <g
         id="g2536" />
      <g
         id="g2538" />
      <g
         id="g2540" />
      <g
         id="g2542" />
      <g
         id="g2544" />
      <g
         id="g2546" />
      <g
         id="g2548" />
      <g
         id="g2550" />
      <g
         id="g2552" />
      <g
         id="g2554" />
      <g
         id="g2556" />
      <g
         id="g2558" />
      <g
         id="g2560" />
      <g
         id="g2562" />
      <g
         id="g2564" />
      <g
         id="g2566">
        <text
           transform="matrix(1,0,0,-1,23.23,53.88)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2570"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2568">0.0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,161.33)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2574"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2572">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,268.78)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2578"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2576">0.4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,376.23)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2582"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan2580">0.6</tspan></text>
      </g>
      <g
         id="g2584" />
      <g
         id="g2586">
        <path
           d="m 40.71,57.83 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2588" />
        <path
           d="m 40.71,165.28 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2590" />
        <path
           d="m 40.71,272.73 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2592" />
        <path
           d="m 40.71,380.18 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2594" />
      </g>
      <g
         id="g2596" />
      <g
         id="g2598" />
      <g
         id="g2600" />
      <g
         id="g2602" />
      <g
         id="g2604" />
      <g
         id="g2606" />
      <g
         id="g2608" />
      <g
         id="g2610" />
      <g
         id="g2612" />
      <g
         id="g2614">
        <path
           d="m 245.6,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2616" />
        <path
           d="m 448,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2618" />
        <path
           d="m 650.41,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2620" />
        <path
           d="m 852.81,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2622" />
      </g>
      <g
         id="g2624" />
      <g
         id="g2626">
        <text
           transform="matrix(1,0,0,-1,239.33,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2630"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan2628">−2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,441.73,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2634"><tspan
             x="0 6.4239998"
             y="0"
             id="tspan2632">−1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,647.35,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2638"><tspan
             x="0"
             y="0"
             id="tspan2636">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,849.75,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2642"><tspan
             x="0"
             y="0"
             id="tspan2640">1</tspan></text>
      </g>
      <g
         id="g2644" />
      <g
         id="g2646" />
      <g
         id="g2648" />
      <g
         id="g2650" />
      <g
         id="g2652" />
      <g
         id="g2654">
        <text
           transform="matrix(1,0,0,-1,446.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2658"><tspan
             x="0"
             y="0"
             id="tspan2656">y</tspan></text>
      </g>
      <g
         id="g2660" />
      <g
         id="g2662">
        <text
           transform="matrix(0,1,1,0,16.97,210.54)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2666"><tspan
             x="0 8.8959999 14.224 23.120001 32.015999 45.344002 54.240002 57.792 66.208 73.888 82.783997"
             y="0"
             id="tspan2664">nr_employed</tspan></text>
      </g>
      <g
         id="g2668" />
      <g
         id="g2670" />
      <g
         id="g2672" />
      <g
         id="g2674" />
      <g
         id="g2676" />
      <g
         id="g2678">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2682"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 206.56 212.12 223.24001 234.36 245.48 251.03999 262.16 268.82001 279.94 291.06 307.72 318.84 323.28 333.79999 343.39999 354.51999"
             y="0"
             id="tspan2680">Relationship between y and nr_employed</tspan></text>
      </g>
      <g
         id="g2684" />
      <g
         id="g2686" />
      <g
         id="g2688" />
      <g
         id="g2690" />
      <g
         id="g2692" />
      <g
         id="g2694" />
    </g>
  </g>
</svg>


We can see that in each of the example charts above, there is a pronounced nonlinear relationship between the dependent and independent variable that would be difficult to compensate for using a linear basis function.

## Adult data-set

As with the banking data-set, the adult data-set has many variables which are binary variables along with a set of continuous variables (in this case, five). They are summarized in the table below:

| Feature        | Relationship |
|----------------|--------------|
| age            | Nonlinear    |
| capital_gain   | Nonlinear    |
| capital_loss   | Nonlinear    |
| education_num  | Near linear  |
| hours_per_week | Nonlinear    |

I've also created the chart below which illustrates the relationship between each variable and the dependent variable:

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   width="297.0918mm"
   height="209.88162mm"
   viewBox="0 0 297.09181 209.88162"
   version="1.1"
   id="svg8">
  <defs
     id="defs2">
    <rect
       x="2.9399648"
       y="4.8108516"
       width="200.71942"
       height="15.189149"
       id="rect875" />
    <rect
       x="3.7417734"
       y="3.474504"
       width="195.10677"
       height="16.525497"
       id="rect865" />
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath999">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path997" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1011">
      <path
         d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
         id="path1009" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1409">
      <path
         d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
         id="path1407" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1789">
      <path
         d="m 55.68,38.48 h 802.84 v 435.8 H 55.68 Z"
         id="path1787" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1870">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path1868" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath1882">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path1880" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3067">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path3065" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3079">
      <path
         d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
         id="path3077" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3468">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path3466" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3480">
      <path
         d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
         id="path3478" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3860">
      <path
         d="M 0,0 H 864 V 504 H 0 Z"
         id="path3858" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3872">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path3870" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4255">
      <path
         d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
         id="path4253" />
    </clipPath>
  </defs>
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     transform="translate(0.29045691,-4.6142695)">
    <text
       xml:space="preserve"
       id="text863"
       style="font-size:5.64444px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect865);" />
    <text
       xml:space="preserve"
       style="font-size:3.57804px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';stroke-width:0.264583"
       x="15.234363"
       y="10.690781"
       id="text871"><tspan
         id="tspan869"
         x="15.234363"
         y="10.690781"
         style="stroke-width:0.264583" /></text>
    <text
       xml:space="preserve"
       id="text873"
       style="font-size:9.83693px;line-height:1.25;font-family:Garamond;-inkscape-font-specification:'Garamond, Normal';white-space:pre;shape-inside:url(#rect875);"
       transform="matrix(1.6900339,0,0,1.6102871,-3.4911386,-9.1477965)"><tspan
         x="2.9394531"
         y="13.053821"><tspan
           style="font-size:6.84308px">Nonlinear Variables for Adult Data</tspan><tspan
           style="shape-inside:url(#rect875)">
</tspan></tspan></text>
    <rect
       style="fill:#f9fff7;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="rect919"
       width="12.027128"
       height="1.925097e-12"
       x="133.10022"
       y="20" />
    <g
       id="g1838"
       transform="matrix(0.171875,0,0,-0.12566071,-0.19866687,87.947274)">
      <g
         id="g1840" />
      <g
         id="g1842" />
      <g
         id="g1844" />
      <g
         id="g1846" />
      <g
         id="g1848" />
      <g
         id="g1850" />
      <g
         id="g1852" />
      <g
         id="g1854" />
      <g
         id="g1856" />
      <g
         id="g1858" />
      <g
         id="g1860" />
      <g
         id="g1862" />
      <g
         id="g1864">
        <g
           id="g1866"
           clip-path="url(#clipPath1870)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1872" />
        </g>
      </g>
      <g
         id="g1874" />
      <g
         id="g1876">
        <g
           id="g1878"
           clip-path="url(#clipPath1882)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path1884" />
          <path
             d="M 43.45,105.91 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1886" />
          <path
             d="M 43.45,201.64 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1888" />
          <path
             d="M 43.45,297.38 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1890" />
          <path
             d="M 43.45,393.11 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1892" />
          <path
             d="m 288.58,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1894" />
          <path
             d="m 542.34,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1896" />
          <path
             d="m 796.1,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1898" />
          <path
             d="M 43.45,58.04 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1900" />
          <path
             d="M 43.45,153.78 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1902" />
          <path
             d="M 43.45,249.51 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1904" />
          <path
             d="M 43.45,345.25 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1906" />
          <path
             d="M 43.45,440.98 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1908" />
          <path
             d="m 161.7,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1910" />
          <path
             d="m 415.46,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1912" />
          <path
             d="m 669.22,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1914" />
          <path
             d="m 80.5,58.59 9.38,0.45 9.38,0.82 9.38,1.46 9.38,2.5 9.37,4.14 9.38,6.55 9.38,9.76 9.38,13.61 9.38,17.44 9.38,20.61 9.38,22.81 9.38,23.85 9.38,23.76 9.38,22.75 9.38,21.19 9.38,19.43 9.38,17.81 9.38,16.56 9.38,15.46 9.38,14.28 9.38,13.1 9.37,11.95 9.38,10.86 9.38,9.88 9.38,9.02 9.38,8.3 9.38,7.69 9.38,7.05 9.38,6.35 9.38,5.59 9.38,4.76 9.38,3.89 9.38,2.95 9.38,1.97 9.38,0.93 9.38,-0.15 9.38,-1.3 9.38,-2.48 9.37,-3.72 9.38,-5.01 9.38,-6.31 9.38,-7.65 9.38,-9 9.38,-10.34 9.38,-11.45 9.38,-12.23 9.38,-12.72 9.38,-12.88 9.38,-12.78 9.38,-12.41 9.38,-11.83 9.38,-11.02 9.38,-10.1 9.38,-9.16 9.38,-8.22 9.38,-7.32 9.37,-6.42 9.38,-5.55 9.38,-4.67 9.38,-3.82 9.38,-2.97 9.38,-2.2 9.38,-1.62 9.38,-1.24 9.38,-1.03 9.38,-1 9.38,-1.15 9.38,-1.47 9.38,-1.96 9.38,-2.57 9.38,-3.19 9.38,-3.69 9.38,-4.09 9.37,-4.41 9.38,-4.63 9.38,-4.76 9.38,-4.82 9.38,-4.79 9.38,-4.7"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1918" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path1920" />
        </g>
      </g>
      <g
         id="g1922" />
      <g
         id="g1924" />
      <g
         id="g1926" />
      <g
         id="g1928" />
      <g
         id="g1930" />
      <g
         id="g1932" />
      <g
         id="g1934" />
      <g
         id="g1936" />
      <g
         id="g1938" />
      <g
         id="g1940" />
      <g
         id="g1942" />
      <g
         id="g1944" />
      <g
         id="g1946" />
      <g
         id="g1948" />
      <g
         id="g1950" />
      <g
         id="g1952">
        <text
           transform="matrix(1,0,0,-1,23.23,54.09)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1956"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1954">0.0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,149.83)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1960"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1958">0.1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,245.56)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1964"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1962">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,341.3)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1968"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1966">0.3</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,437.03)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text1972"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan1970">0.4</tspan></text>
      </g>
      <g
         id="g1974" />
      <g
         id="g1976">
        <path
           d="m 40.71,58.04 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1978" />
        <path
           d="m 40.71,153.78 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1980" />
        <path
           d="m 40.71,249.51 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1982" />
        <path
           d="m 40.71,345.25 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1984" />
        <path
           d="m 40.71,440.98 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path1986" />
      </g>
      <g
         id="g1988" />
      <g
         id="g1990" />
      <g
         id="g1992" />
      <g
         id="g1994" />
      <g
         id="g1996" />
      <g
         id="g1998" />
      <g
         id="g2000" />
      <g
         id="g2002" />
      <g
         id="g2004" />
      <g
         id="g2006">
        <path
           d="m 161.7,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2008" />
        <path
           d="m 415.46,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2010" />
        <path
           d="m 669.22,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path2012" />
      </g>
      <g
         id="g2014" />
      <g
         id="g2016">
        <text
           transform="matrix(1,0,0,-1,155.58,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2020"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan2018">25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,409.34,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2024"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan2022">50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,663.1,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2028"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan2026">75</tspan></text>
      </g>
      <g
         id="g2030" />
      <g
         id="g2032" />
      <g
         id="g2034" />
      <g
         id="g2036" />
      <g
         id="g2038" />
      <g
         id="g2040">
        <text
           transform="matrix(1,0,0,-1,430.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2044"><tspan
             x="0 11.552 15.104 24 32"
             y="0"
             id="tspan2042">Class</tspan></text>
      </g>
      <g
         id="g2046" />
      <g
         id="g2048">
        <text
           transform="matrix(0,1,1,0,16.97,243.03)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2052"><tspan
             x="0 8.8959999 17.792"
             y="0"
             id="tspan2050">age</tspan></text>
      </g>
      <g
         id="g2054" />
      <g
         id="g2056" />
      <g
         id="g2058" />
      <g
         id="g2060" />
      <g
         id="g2062" />
      <g
         id="g2064"
         transform="matrix(0.92997584,0,0,1,3.1526464,0)">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2068"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 211 215.44 226.56 236.56 246.56 252.12 263.23999 274.35999 285.48001 291.04001 302.16 313.28"
             y="0"
             id="tspan2066">Relationship between Class and age</tspan></text>
      </g>
      <g
         id="g2070" />
      <g
         id="g2072" />
      <g
         id="g2074" />
      <g
         id="g2076" />
      <g
         id="g2078" />
      <g
         id="g2080" />
    </g>
    <g
       id="g3035"
       transform="matrix(0.171875,0,0,-0.12566071,-0.19866687,151.28027)">
      <g
         id="g3037" />
      <g
         id="g3039" />
      <g
         id="g3041" />
      <g
         id="g3043" />
      <g
         id="g3045" />
      <g
         id="g3047" />
      <g
         id="g3049" />
      <g
         id="g3051" />
      <g
         id="g3053" />
      <g
         id="g3055" />
      <g
         id="g3057" />
      <g
         id="g3059" />
      <g
         id="g3061">
        <g
           id="g3063"
           clip-path="url(#clipPath3067)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3069" />
        </g>
      </g>
      <g
         id="g3071" />
      <g
         id="g3073">
        <g
           id="g3075"
           clip-path="url(#clipPath3079)">
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path3081" />
          <path
             d="M 49.56,107.81 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3083" />
          <path
             d="M 49.56,206.86 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3085" />
          <path
             d="M 49.56,305.9 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3087" />
          <path
             d="M 49.56,404.95 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3089" />
          <path
             d="m 178.26,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3091" />
          <path
             d="m 362.12,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3093" />
          <path
             d="m 545.97,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3095" />
          <path
             d="m 729.83,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3097" />
          <path
             d="M 49.56,58.29 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3099" />
          <path
             d="M 49.56,157.33 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3101" />
          <path
             d="M 49.56,256.38 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3103" />
          <path
             d="M 49.56,355.43 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3105" />
          <path
             d="M 49.56,454.48 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3107" />
          <path
             d="m 86.34,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3109" />
          <path
             d="m 270.19,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3111" />
          <path
             d="m 454.05,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3113" />
          <path
             d="m 637.9,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3115" />
          <path
             d="m 821.76,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3117" />
          <path
             d="m 86.34,139.34 9.3,-81.05 h 9.31 l 9.31,0.09 9.31,197.26 9.31,2.98 9.31,181.36 9.31,13.18 9.31,0.95 9.31,0.12 9.31,-0.06 9.3,-0.19 9.31,-0.37 9.31,-0.59 9.31,-0.93 9.31,-1.44 9.31,-2.16 9.31,-3.14 9.31,-4.42 9.31,-6.04 9.31,-7.98 9.31,-10.15 9.3,-12.44 9.31,-14.67 9.31,-16.6 9.31,-18.06 9.31,-18.91 9.31,-19.08 9.31,-18.62 9.31,-17.63 9.31,-16.23 9.31,-14.57 9.3,-12.8 9.31,-11 9.31,-9.21 9.31,-7.52 9.31,-5.91 9.31,-4.4 9.31,-2.97 9.31,-1.61 9.31,-0.31 9.31,0.96 9.31,2.2 9.3,3.44 9.31,4.69 9.31,5.94 9.31,7.21 9.31,8.47 9.31,9.72 9.31,10.93 9.31,12.06 9.31,13.07 9.31,13.91 9.3,14.54 9.31,14.89 9.31,14.95 9.31,14.71 9.31,14.17 9.31,13.36 9.31,12.33 9.31,11.16 9.31,9.91 9.31,8.62 9.31,7.39 9.3,6.23 9.31,5.17 9.31,4.25 9.31,3.44 9.31,2.76 9.31,2.19 9.31,1.73 9.31,1.36 9.31,1.05 9.31,0.82 9.3,0.62 9.31,0.49 9.31,0.36 9.31,0.28 9.31,0.22 9.31,0.16"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3121" />
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3123" />
        </g>
      </g>
      <g
         id="g3125" />
      <g
         id="g3127" />
      <g
         id="g3129" />
      <g
         id="g3131" />
      <g
         id="g3133" />
      <g
         id="g3135" />
      <g
         id="g3137" />
      <g
         id="g3139" />
      <g
         id="g3141" />
      <g
         id="g3143" />
      <g
         id="g3145" />
      <g
         id="g3147" />
      <g
         id="g3149" />
      <g
         id="g3151" />
      <g
         id="g3153" />
      <g
         id="g3155">
        <text
           transform="matrix(1,0,0,-1,23.23,54.34)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3159"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3157">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,153.38)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3163"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3161">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,252.43)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3167"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3165">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,351.48)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3171"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3169">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,450.53)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3175"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3173">1.00</tspan></text>
      </g>
      <g
         id="g3177" />
      <g
         id="g3179">
        <path
           d="m 46.82,58.29 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3181" />
        <path
           d="m 46.82,157.33 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3183" />
        <path
           d="m 46.82,256.38 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3185" />
        <path
           d="m 46.82,355.43 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3187" />
        <path
           d="m 46.82,454.48 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3189" />
      </g>
      <g
         id="g3191" />
      <g
         id="g3193" />
      <g
         id="g3195" />
      <g
         id="g3197" />
      <g
         id="g3199" />
      <g
         id="g3201" />
      <g
         id="g3203" />
      <g
         id="g3205" />
      <g
         id="g3207" />
      <g
         id="g3209">
        <path
           d="m 86.34,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3211" />
        <path
           d="m 270.19,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3213" />
        <path
           d="m 454.05,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3215" />
        <path
           d="m 637.9,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3217" />
        <path
           d="m 821.76,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3219" />
      </g>
      <g
         id="g3221" />
      <g
         id="g3223">
        <text
           transform="matrix(1,0,0,-1,83.28,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3227"><tspan
             x="0"
             y="0"
             id="tspan3225">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,254.9,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3231"><tspan
             x="0 6.1160002 12.232 18.348 24.464001"
             y="0"
             id="tspan3229">25000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,438.76,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3235"><tspan
             x="0 6.1160002 12.232 18.348 24.464001"
             y="0"
             id="tspan3233">50000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,622.61,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3239"><tspan
             x="0 6.1160002 12.232 18.348 24.464001"
             y="0"
             id="tspan3237">75000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,803.41,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3243"><tspan
             x="0 6.1160002 12.232 18.348 24.464001 30.58"
             y="0"
             id="tspan3241">100000</tspan></text>
      </g>
      <g
         id="g3245" />
      <g
         id="g3247" />
      <g
         id="g3249" />
      <g
         id="g3251" />
      <g
         id="g3253" />
      <g
         id="g3255">
        <text
           transform="matrix(1,0,0,-1,434.04,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3259"><tspan
             x="0 11.552 15.104 24 32"
             y="0"
             id="tspan3257">Class</tspan></text>
      </g>
      <g
         id="g3261" />
      <g
         id="g3263">
        <text
           transform="matrix(0,1,1,0,16.97,213.69)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3267"><tspan
             x="0 8 16.896 25.792 29.344 33.792 42.688 46.240002 55.136002 64.031998 72.928001 76.480003"
             y="0"
             id="tspan3265">capital_gain</tspan></text>
      </g>
      <g
         id="g3269" />
      <g
         id="g3271" />
      <g
         id="g3273" />
      <g
         id="g3275" />
      <g
         id="g3277" />
      <g
         id="g3279"
         transform="matrix(0.94288997,0,0,1,2.9201651,0)">
        <text
           transform="matrix(1,0,0,-1,49.56,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3283"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 211 215.44 226.56 236.56 246.56 252.12 263.23999 274.35999 285.48001 291.04001 301.04001 312.16 323.28 327.72 333.28 344.39999 348.84 359.95999 371.07999 382.20001 386.64001"
             y="0"
             id="tspan3281">Relationship between Class and capital_gain</tspan></text>
      </g>
      <g
         id="g3285" />
      <g
         id="g3287" />
      <g
         id="g3289" />
      <g
         id="g3291" />
      <g
         id="g3293" />
      <g
         id="g3295" />
    </g>
    <g
       id="g3436"
       transform="matrix(0.171875,0,0,-0.12566071,148.30133,119.61427)">
      <g
         id="g3438" />
      <g
         id="g3440" />
      <g
         id="g3442" />
      <g
         id="g3444" />
      <g
         id="g3446" />
      <g
         id="g3448" />
      <g
         id="g3450" />
      <g
         id="g3452" />
      <g
         id="g3454" />
      <g
         id="g3456" />
      <g
         id="g3458" />
      <g
         id="g3460" />
      <g
         id="g3462">
        <g
           id="g3464"
           clip-path="url(#clipPath3468)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3470" />
        </g>
      </g>
      <g
         id="g3472" />
      <g
         id="g3474">
        <g
           id="g3476"
           clip-path="url(#clipPath3480)">
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path3482" />
          <path
             d="M 49.56,110.9 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3484" />
          <path
             d="M 49.56,216.12 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3486" />
          <path
             d="M 49.56,321.35 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3488" />
          <path
             d="M 49.56,426.57 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3490" />
          <path
             d="m 170.75,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3492" />
          <path
             d="m 339.58,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3494" />
          <path
             d="m 508.41,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3496" />
          <path
             d="m 677.23,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3498" />
          <path
             d="m 846.06,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3500" />
          <path
             d="M 49.56,58.29 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3502" />
          <path
             d="M 49.56,163.51 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3504" />
          <path
             d="M 49.56,268.73 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3506" />
          <path
             d="M 49.56,373.96 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3508" />
          <path
             d="m 86.34,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3510" />
          <path
             d="m 255.16,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3512" />
          <path
             d="m 423.99,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3514" />
          <path
             d="m 592.82,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3516" />
          <path
             d="m 761.65,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3518" />
          <path
             d="m 86.34,153.72 9.3,-49.67 9.31,-25.24 9.31,-11.47 9.31,-5.01 9.31,-2.17 9.31,-0.97 9.31,-0.44 9.31,-0.21 9.31,-0.1 9.31,-0.05 9.3,-0.03 9.31,-0.02 h 9.31 9.31 l 9.31,0.01 9.31,0.03 9.31,0.05 9.31,0.13 9.31,0.37 9.31,1.12 9.31,3.71 9.3,11.45 9.31,28.81 9.31,49.45 9.31,48.69 9.31,16.51 9.31,-32.17 9.31,-67.56 9.31,-45.3 9.31,-7.81 9.31,18.82 9.3,131.1 9.31,151.4 9.31,42.72 9.31,-10.59 9.31,-66.13 9.31,-127.17 9.31,-89.66 9.31,-26.63 9.31,10.76 9.31,69.27 9.31,123.94 9.3,75.03 9.31,29.01 9.31,9.53 9.31,3.68 9.31,2.21 9.31,0.95 9.31,-0.22 9.31,-1.38 9.31,-2.53 9.31,-3.76 9.3,-5.09 9.31,-6.56 9.31,-8.2 9.31,-10.02 9.31,-12.03 9.31,-14.23 9.31,-16.52 9.31,-18.83 9.31,-20.98 9.31,-22.79 9.31,-24.06 9.3,-24.59 9.31,-24.29 9.31,-23.14 9.31,-21.28 9.31,-18.89 9.31,-16.23 9.31,-13.53 9.31,-11 9.31,-8.73 9.31,-6.79 9.3,-5.21 9.31,-3.93 9.31,-2.93 9.31,-2.18 9.31,-1.6 9.31,-1.16"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3522" />
          <path
             d="m 49.56,38.48 h 808.96 v 435.8 H 49.56 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3524" />
        </g>
      </g>
      <g
         id="g3526" />
      <g
         id="g3528" />
      <g
         id="g3530" />
      <g
         id="g3532" />
      <g
         id="g3534" />
      <g
         id="g3536" />
      <g
         id="g3538" />
      <g
         id="g3540" />
      <g
         id="g3542" />
      <g
         id="g3544" />
      <g
         id="g3546" />
      <g
         id="g3548" />
      <g
         id="g3550" />
      <g
         id="g3552" />
      <g
         id="g3554" />
      <g
         id="g3556">
        <text
           transform="matrix(1,0,0,-1,23.23,54.34)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3560"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3558">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,159.56)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3564"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3562">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,264.79)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3568"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3566">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,370.01)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3572"><tspan
             x="0 6.1160002 9.1739998 15.29"
             y="0"
             id="tspan3570">0.75</tspan></text>
      </g>
      <g
         id="g3574" />
      <g
         id="g3576">
        <path
           d="m 46.82,58.29 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3578" />
        <path
           d="m 46.82,163.51 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3580" />
        <path
           d="m 46.82,268.73 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3582" />
        <path
           d="m 46.82,373.96 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3584" />
      </g>
      <g
         id="g3586" />
      <g
         id="g3588" />
      <g
         id="g3590" />
      <g
         id="g3592" />
      <g
         id="g3594" />
      <g
         id="g3596" />
      <g
         id="g3598" />
      <g
         id="g3600" />
      <g
         id="g3602" />
      <g
         id="g3604">
        <path
           d="m 86.34,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3606" />
        <path
           d="m 255.16,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3608" />
        <path
           d="m 423.99,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3610" />
        <path
           d="m 592.82,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3612" />
        <path
           d="m 761.65,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3614" />
      </g>
      <g
         id="g3616" />
      <g
         id="g3618">
        <text
           transform="matrix(1,0,0,-1,83.28,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3622"><tspan
             x="0"
             y="0"
             id="tspan3620">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,242.93,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3626"><tspan
             x="0 6.1160002 12.232 18.348"
             y="0"
             id="tspan3624">1000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,411.76,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3630"><tspan
             x="0 6.1160002 12.232 18.348"
             y="0"
             id="tspan3628">2000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,580.59,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3634"><tspan
             x="0 6.1160002 12.232 18.348"
             y="0"
             id="tspan3632">3000</tspan></text>
        <text
           transform="matrix(1,0,0,-1,749.42,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3638"><tspan
             x="0 6.1160002 12.232 18.348"
             y="0"
             id="tspan3636">4000</tspan></text>
      </g>
      <g
         id="g3640" />
      <g
         id="g3642" />
      <g
         id="g3644" />
      <g
         id="g3646" />
      <g
         id="g3648" />
      <g
         id="g3650">
        <text
           transform="matrix(1,0,0,-1,434.04,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3654"><tspan
             x="0 11.552 15.104 24 32"
             y="0"
             id="tspan3652">Class</tspan></text>
      </g>
      <g
         id="g3656" />
      <g
         id="g3658">
        <text
           transform="matrix(0,1,1,0,16.97,214.59)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3662"><tspan
             x="0 8 16.896 25.792 29.344 33.792 42.688 46.240002 55.136002 58.688 67.584 75.584"
             y="0"
             id="tspan3660">capital_loss</tspan></text>
      </g>
      <g
         id="g3664" />
      <g
         id="g3666" />
      <g
         id="g3668" />
      <g
         id="g3670" />
      <g
         id="g3672" />
      <g
         id="g3674"
         transform="matrix(0.8398214,0,0,1,8.1902947,0)">
        <text
           transform="matrix(1,0,0,-1,49.56,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3678"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 211 215.44 226.56 236.56 246.56 252.12 263.23999 274.35999 285.48001 291.04001 301.04001 312.16 323.28 327.72 333.28 344.39999 348.84 359.95999 364.39999 375.51999 385.51999"
             y="0"
             id="tspan3676">Relationship between Class and capital_loss</tspan></text>
      </g>
      <g
         id="g3680" />
      <g
         id="g3682" />
      <g
         id="g3684" />
      <g
         id="g3686" />
      <g
         id="g3688" />
      <g
         id="g3690" />
    </g>
    <g
       id="g3828"
       transform="matrix(0.171875,0,0,-0.12566071,147.35946,182.39638)">
      <g
         id="g3830" />
      <g
         id="g3832" />
      <g
         id="g3834" />
      <g
         id="g3836" />
      <g
         id="g3838" />
      <g
         id="g3840" />
      <g
         id="g3842" />
      <g
         id="g3844" />
      <g
         id="g3846" />
      <g
         id="g3848" />
      <g
         id="g3850" />
      <g
         id="g3852" />
      <g
         id="g3854">
        <g
           id="g3856"
           clip-path="url(#clipPath3860)">
          <path
             d="M 0,0 H 864 V 504 H 0 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3862" />
        </g>
      </g>
      <g
         id="g3864" />
      <g
         id="g3866">
        <g
           id="g3868"
           clip-path="url(#clipPath3872)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path3874" />
          <path
             d="M 43.45,103.55 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3876" />
          <path
             d="M 43.45,202.83 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3878" />
          <path
             d="M 43.45,302.1 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3880" />
          <path
             d="M 43.45,401.38 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3882" />
          <path
             d="m 129.9,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3884" />
          <path
             d="m 327.49,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3886" />
          <path
             d="m 525.08,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3888" />
          <path
             d="m 722.68,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3890" />
          <path
             d="M 43.45,53.92 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3892" />
          <path
             d="M 43.45,153.19 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3894" />
          <path
             d="M 43.45,252.47 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3896" />
          <path
             d="M 43.45,351.74 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3898" />
          <path
             d="M 43.45,451.02 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3900" />
          <path
             d="m 228.69,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3902" />
          <path
             d="m 426.29,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3904" />
          <path
             d="m 623.88,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3906" />
          <path
             d="m 821.47,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3908" />
          <path
             d="m 80.5,65.55 9.38,0.97 9.38,1.04 9.38,1.1 9.38,1.16 9.37,1.2 9.38,1.24 9.38,1.25 9.38,1.25 9.38,1.22 9.38,1.17 9.38,1.09 9.38,1 9.38,0.9 9.38,0.76 9.38,0.63 9.38,0.47 9.38,0.29 9.38,0.12 9.38,-0.04 9.38,-0.15 9.38,-0.23 9.37,-0.25 9.38,-0.23 9.38,-0.16 9.38,-0.05 9.38,0.1 9.38,0.3 9.38,0.52 9.38,0.76 9.38,1.02 9.38,1.31 9.38,1.62 9.38,2 9.38,2.41 9.38,2.9 9.38,3.46 9.38,4.02 9.38,4.55 9.37,5.02 9.38,5.4 9.38,5.62 9.38,5.67 9.38,5.49 9.38,5.03 9.38,4.36 9.38,3.73 9.38,3.22 9.38,2.84 9.38,2.63 9.38,2.61 9.38,2.78 9.38,3.18 9.38,3.8 9.38,4.59 9.38,5.44 9.38,6.32 9.37,7.25 9.38,8.22 9.38,9.25 9.38,10.3 9.38,11.37 9.38,12.44 9.38,13.42 9.38,14.17 9.38,14.68 9.38,14.91 9.38,14.86 9.38,14.5 9.38,13.88 9.38,13.01 9.38,11.97 9.38,10.94 9.38,10.01 9.37,9.2 9.38,8.49 9.38,7.89 9.38,7.39 9.38,6.98 9.38,6.65"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3912" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3914" />
        </g>
      </g>
      <g
         id="g3916" />
      <g
         id="g3918" />
      <g
         id="g3920" />
      <g
         id="g3922" />
      <g
         id="g3924" />
      <g
         id="g3926" />
      <g
         id="g3928" />
      <g
         id="g3930" />
      <g
         id="g3932" />
      <g
         id="g3934" />
      <g
         id="g3936" />
      <g
         id="g3938" />
      <g
         id="g3940" />
      <g
         id="g3942" />
      <g
         id="g3944" />
      <g
         id="g3946">
        <text
           transform="matrix(1,0,0,-1,23.23,49.97)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3950"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan3948">0.0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,149.24)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3954"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan3952">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,248.52)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3958"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan3956">0.4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,347.79)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3962"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan3960">0.6</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,447.07)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3966"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan3964">0.8</tspan></text>
      </g>
      <g
         id="g3968" />
      <g
         id="g3970">
        <path
           d="m 40.71,53.92 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3972" />
        <path
           d="m 40.71,153.19 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3974" />
        <path
           d="m 40.71,252.47 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3976" />
        <path
           d="m 40.71,351.74 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3978" />
        <path
           d="m 40.71,451.02 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3980" />
      </g>
      <g
         id="g3982" />
      <g
         id="g3984" />
      <g
         id="g3986" />
      <g
         id="g3988" />
      <g
         id="g3990" />
      <g
         id="g3992" />
      <g
         id="g3994" />
      <g
         id="g3996" />
      <g
         id="g3998" />
      <g
         id="g4000">
        <path
           d="m 228.69,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4002" />
        <path
           d="m 426.29,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4004" />
        <path
           d="m 623.88,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4006" />
        <path
           d="m 821.47,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4008" />
      </g>
      <g
         id="g4010" />
      <g
         id="g4012">
        <text
           transform="matrix(1,0,0,-1,225.63,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4016"><tspan
             x="0"
             y="0"
             id="tspan4014">4</tspan></text>
        <text
           transform="matrix(1,0,0,-1,423.23,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4020"><tspan
             x="0"
             y="0"
             id="tspan4018">8</tspan></text>
        <text
           transform="matrix(1,0,0,-1,617.76,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4024"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan4022">12</tspan></text>
        <text
           transform="matrix(1,0,0,-1,815.36,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4028"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan4026">16</tspan></text>
      </g>
      <g
         id="g4030" />
      <g
         id="g4032" />
      <g
         id="g4034" />
      <g
         id="g4036" />
      <g
         id="g4038" />
      <g
         id="g4040">
        <text
           transform="matrix(1,0,0,-1,430.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4044"><tspan
             x="0 11.552 15.104 24 32"
             y="0"
             id="tspan4042">Class</tspan></text>
      </g>
      <g
         id="g4046" />
      <g
         id="g4048">
        <text
           transform="matrix(0,1,1,0,16.97,201.76)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4052"><tspan
             x="0 8.8959999 17.792 26.688 34.688 43.584 48.032001 51.584 60.48 69.375999 78.272003 87.008003 95.903999"
             y="0"
             id="tspan4050">education_num</tspan></text>
      </g>
      <g
         id="g4054" />
      <g
         id="g4056" />
      <g
         id="g4058" />
      <g
         id="g4060" />
      <g
         id="g4062" />
      <g
         id="g4064"
         transform="matrix(0.83890284,0,0,1,7.2529591,0)">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4068"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 211 215.44 226.56 236.56 246.56 252.12 263.23999 274.35999 285.48001 291.04001 302.16 313.28 324.39999 334.39999 345.51999 351.07999 355.51999 366.64001 377.76001 388.88 399.79999 410.92001"
             y="0"
             id="tspan4066">Relationship between Class and education_num</tspan></text>
      </g>
      <g
         id="g4070" />
      <g
         id="g4072" />
      <g
         id="g4074" />
      <g
         id="g4076" />
      <g
         id="g4078" />
      <g
         id="g4080" />
    </g>
    <g
       id="g4217"
       transform="matrix(0.17166247,0,0,-0.12539454,-0.19860566,214.47915)">
      <g
         id="g4219" />
      <g
         id="g4221" />
      <g
         id="g4223" />
      <g
         id="g4225" />
      <g
         id="g4227" />
      <g
         id="g4229" />
      <g
         id="g4231" />
      <g
         id="g4233" />
      <g
         id="g4235" />
      <g
         id="g4237" />
      <g
         id="g4239" />
      <g
         id="g4241" />
      <g
         id="g4243">
        <path
           d="M 0,0 H 864 V 504 H 0 Z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4245" />
      </g>
      <g
         id="g4247" />
      <g
         id="g4249">
        <g
           id="g4251"
           clip-path="url(#clipPath4255)">
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path4257" />
          <path
             d="M 43.45,56.48 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4259" />
          <path
             d="M 43.45,150.15 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4261" />
          <path
             d="M 43.45,243.82 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4263" />
          <path
             d="M 43.45,337.49 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4265" />
          <path
             d="M 43.45,431.16 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4267" />
          <path
             d="m 167.45,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4269" />
          <path
             d="m 356.47,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4271" />
          <path
             d="m 545.5,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4273" />
          <path
             d="m 734.52,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4275" />
          <path
             d="M 43.45,103.32 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4277" />
          <path
             d="M 43.45,196.99 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4279" />
          <path
             d="M 43.45,290.66 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4281" />
          <path
             d="M 43.45,384.33 H 858.52"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4283" />
          <path
             d="m 72.94,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4285" />
          <path
             d="m 261.96,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4287" />
          <path
             d="m 450.98,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4289" />
          <path
             d="m 640.01,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4291" />
          <path
             d="m 829.03,38.48 v 435.8"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4293" />
          <path
             d="m 80.5,116.55 9.38,-7.83 9.38,-7.21 9.38,-6.53 9.38,-5.84 9.37,-5.11 9.38,-4.39 9.38,-3.68 9.38,-2.96 9.38,-2.25 9.38,-1.63 9.38,-1.09 9.38,-0.63 9.38,-0.23 9.38,0.11 9.38,0.41 9.38,0.65 9.38,0.87 9.38,1.12 9.38,1.49 9.38,2.02 9.38,2.69 9.37,3.58 9.38,4.72 9.38,6.19 9.38,8.09 9.38,10.41 9.38,12.95 9.38,15.66 9.38,18.49 9.38,21.3 9.38,23.88 9.38,26.03 9.38,27.48 9.38,28.04 9.38,27.81 9.38,26.78 9.38,24.93 9.38,22.34 9.37,19.07 9.38,15.2 9.38,10.82 9.38,6.08 9.38,1.71 9.38,-1.93 9.38,-4.83 9.38,-7 9.38,-8.4 9.38,-9.06 9.38,-8.99 9.38,-8.23 9.38,-7.18 9.38,-6.22 9.38,-5.35 9.38,-4.55 9.38,-3.86 9.38,-3.22 9.37,-2.68 9.38,-2.21 9.38,-1.82 9.38,-1.51 9.38,-1.28 9.38,-1.16 9.38,-1.13 9.38,-1.21 9.38,-1.39 9.38,-1.66 9.38,-2.04 9.38,-2.5 9.38,-3.07 9.38,-3.7 9.38,-4.29 9.38,-4.81 9.38,-5.24 9.37,-5.6 9.38,-5.88 9.38,-6.07 9.38,-6.2 9.38,-6.25 9.38,-6.24"
             style="fill:none;stroke:#3366ff;stroke-width:2.13;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4297" />
          <path
             d="m 43.45,38.48 h 815.07 v 435.8 H 43.45 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4299" />
        </g>
      </g>
      <g
         id="g4301" />
      <g
         id="g4303" />
      <g
         id="g4305" />
      <g
         id="g4307" />
      <g
         id="g4309" />
      <g
         id="g4311" />
      <g
         id="g4313" />
      <g
         id="g4315" />
      <g
         id="g4317" />
      <g
         id="g4319" />
      <g
         id="g4321" />
      <g
         id="g4323" />
      <g
         id="g4325" />
      <g
         id="g4327" />
      <g
         id="g4329" />
      <g
         id="g4331">
        <text
           transform="matrix(1,0,0,-1,23.23,99.37)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4335"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan4333">0.1</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,193.04)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4339"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan4337">0.2</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,286.71)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4343"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan4341">0.3</tspan></text>
        <text
           transform="matrix(1,0,0,-1,23.23,380.38)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4347"><tspan
             x="0 6.1160002 9.1739998"
             y="0"
             id="tspan4345">0.4</tspan></text>
      </g>
      <g
         id="g4349" />
      <g
         id="g4351">
        <path
           d="m 40.71,103.32 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4353" />
        <path
           d="m 40.71,196.99 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4355" />
        <path
           d="m 40.71,290.66 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4357" />
        <path
           d="m 40.71,384.33 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4359" />
      </g>
      <g
         id="g4361" />
      <g
         id="g4363" />
      <g
         id="g4365" />
      <g
         id="g4367" />
      <g
         id="g4369" />
      <g
         id="g4371" />
      <g
         id="g4373" />
      <g
         id="g4375" />
      <g
         id="g4377" />
      <g
         id="g4379">
        <path
           d="m 72.94,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4381" />
        <path
           d="m 261.96,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4383" />
        <path
           d="m 450.98,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4385" />
        <path
           d="m 640.01,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4387" />
        <path
           d="m 829.03,35.74 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path4389" />
      </g>
      <g
         id="g4391" />
      <g
         id="g4393">
        <text
           transform="matrix(1,0,0,-1,69.88,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4397"><tspan
             x="0"
             y="0"
             id="tspan4395">0</tspan></text>
        <text
           transform="matrix(1,0,0,-1,255.84,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4401"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan4399">25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,444.87,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4405"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan4403">50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,633.89,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4409"><tspan
             x="0 6.1160002"
             y="0"
             id="tspan4407">75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,819.86,25.65)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4413"><tspan
             x="0 6.1160002 12.232"
             y="0"
             id="tspan4411">100</tspan></text>
      </g>
      <g
         id="g4415" />
      <g
         id="g4417" />
      <g
         id="g4419" />
      <g
         id="g4421" />
      <g
         id="g4423" />
      <g
         id="g4425">
        <text
           transform="matrix(1,0,0,-1,430.98,9)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4429"><tspan
             x="0 11.552 15.104 24 32"
             y="0"
             id="tspan4427">Class</tspan></text>
      </g>
      <g
         id="g4431" />
      <g
         id="g4433">
        <text
           transform="matrix(0,1,1,0,16.97,197.32)"
           style="font-variant:normal;font-weight:normal;font-size:16px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4437"><tspan
             x="0 8.8959999 17.792 26.688 32.015999 40.015999 48.911999 57.807999 66.704002 72.031998 80.928001 92.32 101.216 110.112"
             y="0"
             id="tspan4435">hours_per_week</tspan></text>
      </g>
      <g
         id="g4439" />
      <g
         id="g4441" />
      <g
         id="g4443" />
      <g
         id="g4445" />
      <g
         id="g4447" />
      <g
         id="g4449"
         transform="matrix(0.87352882,0,0,1,5.6940193,0)">
        <text
           transform="matrix(1,0,0,-1,43.45,484.16)"
           style="font-variant:normal;font-weight:normal;font-size:20px;font-family:Helvetica;-inkscape-font-specification:Helvetica;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text4453"><tspan
             x="0 14.44 25.559999 30 41.119999 46.68 51.119999 62.240002 73.360001 83.360001 94.480003 98.919998 110.04 115.6 126.72 137.84 143.39999 157.64 168.75999 179.88 191 196.56 211 215.44 226.56 236.56 246.56 252.12 263.23999 274.35999 285.48001 291.04001 302.16 313.28 324.39999 331.06 341.06 352.17999 363.29999 374.42001 381.07999 392.20001 406.44 417.56 428.67999"
             y="0"
             id="tspan4451">Relationship between Class and hours_per_week</tspan></text>
      </g>
      <g
         id="g4455" />
      <g
         id="g4457" />
      <g
         id="g4459" />
      <g
         id="g4461" />
      <g
         id="g4463" />
      <g
         id="g4465" />
    </g>
  </g>
</svg>


The relationship between the continuous independent variables and the depeendent variables is as it was with the banking data-set. The sole exception is the education_num variable which represents the number of years in school. We can see that it has a clearly positive relationship and might be approximated with a linear function, despite the concavity.




# Implementation

## Building Decision Trees

We begin by building simple decision trees to use as a factor *instead* of the original categorical variables. These trees will have to be small in order to avoid overfitting. I've tested models using trees with depths of two three, and four and chosen on using a maximum depth of four in each model based on its improved accuracy in ROC and PR AUC. In the code chunk below, the CART decision trees are fitted in the code chunk below and the leaf node prediction is then used to create a new factor variable in the data. After that, the nodal values are use in a logistic regression specification.
```r
synthetic.tree.2 <- rpart(data = training.data$Synthetic,
                       formula = Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3,
                        control = rpart.control(  maxdepth = 4 ))
banking.tree.3 <- rpart(data = banking.smote,
                       formula = y~.-V1, control = rpart.control(  maxdepth = 4)  )
adult.tree.3 <- rpart(data = adult.smote,
                       formula = Class~age+workclass+fnlwgt+education+education_num +marital_status+occupation+relationship+race+sex+capital_gain +  capital_loss + hours_per_week+ native_country,
                        control = rpart.control(  maxdepth = 4))

source("build_logistic_w_tree.r")
synth.models <- list(synthetic.tree.2)
banking.models <- list(banking.tree.3)
adult.models <- list(adult.tree.3)

synth.train.preds <- lapply(X = synth.models,FUN = function(x){   tree.to.feature(tree.model = x,dt = training.data$Synthetic)  }) %>% data.frame
banking.train.preds <-lapply(X = banking.models,FUN = function(x){   tree.to.feature(tree.model = x,dt = training.data$Banking)  }) %>% data.frame
adult.train.preds <-lapply(X = adult.models,FUN = function(x){   tree.to.feature(tree.model = x,dt = training.data$Adult)  }) %>% data.frame

names(synth.train.preds) <- c("three.nodes")
names(banking.train.preds) <- c("four.nodes")
names(adult.train.preds) <- c("four.nodes")

training.data$Synthetic <- cbind( training.data$Synthetic,synth.train.preds )
training.data$Banking <- cbind(training.data$Banking ,banking.train.preds )
training.data$Adult <- cbind(training.data$Adult ,adult.train.preds )


synth.model.three.deep<- glm(formula = Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3 + three.nodes,family = "binomial",data = training.data$Synthetic)

banking.mode.four.deep <- glm(formula = y~.-V1 ,family = "binomial",data = training.data$Banking)

adult.mode.four.deep <- glm(formula = Class~age+workclass+fnlwgt+education+education_num +marital_status+occupation+relationship+race+sex+capital_gain +  capital_loss + hours_per_week+ native_country + four.nodes,
                           family = "binomial",
                           data = training.data$Adult)
```

Logistic regression models and their generalized additive counterparts are fitted below:

```r
# Create GAM models and the GLM models:
synth.gam <- gam(data = training.data$Synthetic,formula = Class~s(Linear1)+s(Linear2)+s(Linear3)+s(Linear4)+s(Linear5)+s(Linear6)+s(Nonlinear1) +s(Nonlinear2) +s(Nonlinear3),family = binomial)

banking.gam <- gam(formula = y~s(age)+s(previous)+s(euribor3m)+s(cons_conf_idx)+s(cons_price_idx)+s(nr_employed)+s(emp_var_rate)+s(pdays)+`job_blue-collar`+
                     job_management+`job_other 1`+`job_other 2`+job_services+job_technician+marital_married+marital_single+ education_high.school+education_professional.course+education_university.degree+education_unknown+default_unknown+housing_unknown+
                     housing_yes+loan_unknown+loan_yes+poutcome_nonexistent+poutcome_success,family = "binomial",data = training.data$Banking)

adult.gam <- gam(formula = Class~s(age)+(workclass)+s(fnlwgt)+(education)+s(education_num) +(marital_status)+(occupation)+(relationship)+(race)+(sex)+s(capital_gain) +  s(capital_loss) + s(hours_per_week)+ (native_country),
                           family = "binomial",
                           data = training.data$Adult)


# Create GLM Models

synth.model.glm<- glm(formula = Class~Linear1+Linear2+Linear3+Linear4+Linear5+Linear6+Nonlinear1 +Nonlinear2 +Nonlinear3 ,family = "binomial",data = training.data$Synthetic)

banking.mode.glm <- glm(formula = y~.-V1-four.nodes ,family = "binomial",data = training.data$Banking)

adult.mode.glm <- glm(formula = Class~age+workclass+fnlwgt+education+education_num +marital_status+occupation+relationship+race+sex+capital_gain +  capital_loss + hours_per_week+ native_country ,
                           family = "binomial",
                           data = training.data$Adult)
```
 

# Model Testing

Now that we've fit the models, we can engage a comparison of their results. To do this, we will rely on ROC AUC and PR AUC. I've summarized the in-sample and out-of-sample results for the model together in the figures and table below:

```r
synth.test.preds <- lapply(X = synth.models,FUN = function(x){  
     tree.to.feature(tree.model = x,dt = test.data$Synthetic)  }) %>% data.frame
banking.test.preds <-lapply(X = banking.models,FUN = function(x){ 
      tree.to.feature(tree.model = x,dt = test.data$Banking)  }) %>% data.frame
adult.test.preds <- lapply(X = adult.models,FUN = function(x){ 
      tree.to.feature(tree.model = x,dt = test.data$Adult)  }) %>% data.frame

names(synth.test.preds) <- c( "three.nodes")
names(banking.test.preds) <- c( "four.nodes")
names(adult.test.preds) <- c( "four.nodes")

test.data$Synthetic <- cbind( test.data$Synthetic,synth.test.preds )
test.data$Banking <- cbind(test.data$Banking ,banking.test.preds )
test.data$Adult <- cbind(test.data$Adult ,adult.test.preds )
```

 
```r
training.for.mmdata <- data.frame(predict(banking.mode.glm,newdata = training.data$Banking, type = "response" ),
                                 predict(banking.mode.four.deep, newdata = training.data$Banking,type = "response" ),
                                  predict(banking.gam,newdata = training.data$Banking, type = "response" )  ) 
 
training.mdat <- mmdata(scores = training.for.mmdata,labels = training.data$Banking$y,
                       modnames = c("Logistic Regression", "Tree w/ GLM", "GAM"))
```


```r
autoplot(evalmod(training.mdat),curvetype = c("ROC"))+theme(legend.position = "bottom") +ggtitle("ROC Curve - Training Data")
```
```r
autoplot(evalmod(training.mdat),curvetype = c("PR"))+theme(legend.position = "bottom") +ggtitle("PR Curve - Training Data")
```

```r
testing.for.mmdata <- data.frame(predict(synth.model.glm,newdata = test.data$Synthetic, type = "response" ),
                                 predict(synth.model.three.deep, newdata = test.data$Synthetic,type = "response" ),
                                  predict(synth.gam,newdata = test.data$Synthetic, type = "response" )  ) 
testing_mdat <- mmdata(scores = testing.for.mmdata,labels = test.data$Synthetic$Class,
                       modnames = c("Logistic Regression", "Tree w/ GLM", "GAM"))
```
```r
autoplot(evalmod(testing_mdat),curvetype = c("ROC"))+theme(legend.position = "bottom") +ggtitle("ROC Curve - Test Data")
```  

```r
autoplot(evalmod(testing_mdat),curvetype = c("PR"))+theme(legend.position = "bottom") +ggtitle("PR Curve - Test Data")
```
Each model performs well in both the training and testing data-sets and there is little to no deterioration of model accuracy when switching to the out-of-sample data. In all cases there is a small improvement of the performance metrics from using decision trees as a feature engineering step. We also see that in all cases the GAM model outperforms either model. In several cases, the GAM outperforms the GLM+Tree model by a greater margin than the GLM+Tree outperforms the logistic regression model (see PR AUC for Banking and Synthetic and the Adult data-set). From this, we can see that including the structure of the nonlinear data in the model specification instead of transforming it allows for improved performance in all cases. This is significant given that using GAM also eliminates the investigation and comparison of potential decision trees for the model.

TABLE HERE

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="270mm"
   height="149.14511mm"
   viewBox="0 0 270 149.14511"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="training_curves for post.svg">
  <defs
     id="defs2">
    <style
       type="text/css"
       id="style1139"><![CDATA[
    .svglite line, .svglite polyline, .svglite polygon, .svglite path, .svglite rect, .svglite circle {
      fill: none;
      stroke: #000000;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-miterlimit: 10.00;
    }
  ]]></style>
    <clipPath
       id="cpMC4wMHw1MDQuMDB8MC4wMHw1MDQuMDA=">
      <rect
         x="0"
         y="0"
         width="504"
         height="504"
         id="rect1145" />
    </clipPath>
    <clipPath
       id="cpMjMuOTZ8NDgwLjA0fDAuMDB8NTA0LjAw">
      <rect
         x="23.959999"
         y="0"
         width="456.09"
         height="504"
         id="rect1152" />
    </clipPath>
    <clipPath
       id="cpNjQuNDR8NDc0LjU3fDIzLjE4fDQzMy4zMQ==">
      <rect
         x="64.440002"
         y="23.18"
         width="410.13"
         height="410.13"
         id="rect1163" />
    </clipPath>
    <style
       type="text/css"
       id="style1671"><![CDATA[
    .svglite line, .svglite polyline, .svglite polygon, .svglite path, .svglite rect, .svglite circle {
      fill: none;
      stroke: #000000;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-miterlimit: 10.00;
    }
  ]]></style>
    <clipPath
       id="cpMC4wMHw1MDQuMDB8MC4wMHw1MDQuMDA=-8">
      <rect
         x="0"
         y="0"
         width="504"
         height="504"
         id="rect1677" />
    </clipPath>
    <clipPath
       id="cpMjMuOTZ8NDgwLjA0fDAuMDB8NTA0LjAw-8">
      <rect
         x="23.959999"
         y="0"
         width="456.09"
         height="504"
         id="rect1684" />
    </clipPath>
    <clipPath
       id="cpNjQuNDR8NDc0LjU3fDIzLjE4fDQzMy4zMQ==-1">
      <rect
         x="64.440002"
         y="23.18"
         width="410.13"
         height="410.13"
         id="rect1695" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2896">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path2894" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2908">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path2906" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3471">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path3469" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3483">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path3481" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.979899"
     inkscape:cx="519.59932"
     inkscape:cy="248.26555"
     inkscape:document-units="mm"
     inkscape:current-layer="g3465"
     inkscape:document-rotation="0"
     showgrid="false"
     inkscape:window-width="2880"
     inkscape:window-height="1526"
     inkscape:window-x="2869"
     inkscape:window-y="-11"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1">
    <g
       id="g2852"
       inkscape:label="training_ROC_AUC_plot"
       transform="matrix(0.29592284,0,0,-0.29592284,-7.0725558,149.14512)">
      <g
         id="g2854" />
      <g
         id="g2856" />
      <g
         id="g2858" />
      <g
         id="g2860" />
      <g
         id="g2862" />
      <g
         id="g2864" />
      <g
         id="g2866" />
      <g
         id="g2868" />
      <g
         id="g2870" />
      <g
         id="g2872" />
      <g
         id="g2874" />
      <g
         id="g2876" />
      <g
         id="g2878" />
      <g
         id="g2880" />
      <g
         id="g2882" />
      <g
         id="g2884" />
      <g
         id="g2886" />
      <g
         id="g2888" />
      <g
         id="g2890">
        <g
           id="g2892"
           clip-path="url(#clipPath2896)">
          <path
             d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2898" />
        </g>
      </g>
      <g
         id="g2900" />
      <g
         id="g2902">
        <g
           id="g2904"
           clip-path="url(#clipPath2908)">
          <path
             d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path2910" />
          <path
             d="M 64.88,136.29 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2912" />
          <path
             d="M 64.88,229.42 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2914" />
          <path
             d="M 64.88,322.54 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2916" />
          <path
             d="M 64.88,415.66 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2918" />
          <path
             d="M 130.07,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2920" />
          <path
             d="M 223.19,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2922" />
          <path
             d="M 316.31,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2924" />
          <path
             d="M 409.44,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2926" />
          <path
             d="M 64.88,89.73 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2928" />
          <path
             d="M 64.88,182.86 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2930" />
          <path
             d="M 64.88,275.98 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2932" />
          <path
             d="M 64.88,369.1 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2934" />
          <path
             d="M 64.88,462.22 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2936" />
          <path
             d="M 83.51,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2938" />
          <path
             d="M 176.63,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2940" />
          <path
             d="M 269.75,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2942" />
          <path
             d="M 362.87,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2944" />
          <path
             d="M 456,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2946" />
          <path
             d="m 83.51,89.73 v 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 l 0.37,9.05 0.37,8.05 0.37,8.95 0.38,7.06 0.37,12.43 0.37,6.26 0.37,5.47 0.38,3.48 0.37,3.78 0.37,2.39 0.37,2.58 0.38,3.48 0.37,2.39 0.37,1.49 0.37,2.09 0.38,2.39 0.37,1.89 0.37,2.38 0.37,2.39 0.38,1.29 0.37,1.99 0.37,1.99 0.37,2.58 0.38,2.09 0.37,2.29 0.37,1.99 0.37,1.79 0.38,1.99 0.37,2.08 0.37,2.29 0.37,1.29 0.38,1.69 0.37,1.89 0.37,1.69 0.37,1.2 0.38,2.09 0.37,1.98 0.37,2.49 0.37,1.79 0.38,1.49 0.37,2.49 0.37,1.39 0.37,1.69 0.38,0.5 0.37,1.59 0.37,1.79 0.37,2.48 0.38,1.69 0.37,1.1 0.37,1.59 0.37,1.19 0.38,2.29 0.37,1.49 0.37,1.79 0.37,1.29 0.38,1.4 0.37,1.39 0.37,2.09 0.37,1.19 0.38,1.29 0.37,1.59 0.37,0.6 0.37,1.49 0.38,1.69 0.37,1.59 0.37,1.3 0.37,1.69 0.38,1.69 0.37,0.99 0.37,1.09 0.37,0.9 0.38,1.09 0.37,0.8 0.37,1.09 0.37,1.59 0.38,0.9 0.37,1.39 0.37,0.9 0.37,0.8 0.38,1.68 0.37,1.19 0.37,0.5 0.37,1.29 0.38,0.8 0.37,1.79 0.37,1.59 0.37,1.49 0.38,0.6 0.37,0.4 0.37,0.49 0.37,1 0.38,1.69 0.37,0.5 0.37,0.49 0.37,1.3 0.38,1.19 0.37,0.79 0.37,0.7 0.37,0.4 0.38,0.89 0.37,0.7 0.37,0.6 0.37,0.49 0.38,0.1 0.37,0.4 0.37,0.3 0.37,1.39 0.38,0.7 0.37,0.59 0.37,0.8 0.37,0.5 0.38,0.39 0.37,1 0.37,0.3 0.37,0.49 0.38,0.8 0.37,0.5 0.37,0.49 0.37,0.8 0.38,0.99 0.37,0.2 0.37,0.6 0.37,0.5 0.38,0.99 0.37,0.6 0.37,0.39 0.37,0.3 0.38,0.4 0.37,0.46 0.37,0.04 0.37,0.3 0.38,0.49 0.37,0.8 0.37,0.4 0.37,0.59 0.38,0.2 0.37,0.6 0.37,0.6 0.37,0.49 0.38,0.8 0.37,0.59 0.37,0.1 0.37,0.2 0.38,0.4 0.37,0.5 0.37,0.55 0.37,0.64 0.38,0.7 0.37,0.59 0.37,0.72 0.37,0.68 0.38,0.59 0.37,0.8 0.37,0.4 0.37,0.1 0.38,0.69 0.37,0.5 0.37,0.3 0.37,0.49 0.38,0.5 0.37,0.1 0.37,0.2 0.37,0.79 0.38,0.1 0.37,0.1 0.37,0.3 0.37,0.31 0.38,0.29 0.37,0.3 0.37,0.39 0.37,0.3 0.38,0.3 0.37,0.5 0.37,0.3 0.37,0.2 h 0.37 l 0.38,0.29 0.37,0.4 0.37,0.2 0.37,0.2 0.38,0.4 0.37,0.3 0.37,0.07 0.37,0.92 0.38,0.7 0.37,0.19 0.37,0.1 0.37,0.4 0.38,0.2 0.37,0.4 0.37,0.3 0.37,0.49 0.38,0.5 0.37,0.7 0.37,0.49 0.37,0.1 0.38,0.5 0.37,0.2 0.37,0.6 0.37,0.29 0.38,0.4 0.37,0.3 0.37,0.2 0.37,0.3 0.38,0.4 h 0.37 l 0.37,0.19 0.37,0.2 0.38,0.4 0.37,0.3 h 0.37 l 0.37,0.4 0.38,0.2 0.37,0.29 0.37,0.7 0.37,0.5 0.38,0.5 0.37,0.69 0.37,0.3 0.37,0.3 0.38,0.3 h 0.37 l 0.37,0.29 0.37,0.3 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.4 0.37,0.1 0.37,0.21 0.37,0.38 h 0.38 0.37 0.37 l 0.37,0.2 0.38,0.3 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.19 0.38,0.2 0.37,0.3 0.37,0.3 0.37,0.4 0.38,0.3 0.37,0.1 0.37,0.49 0.37,0.34 0.38,0.66 0.37,0.39 0.37,0.2 0.37,0.3 0.38,0.2 h 0.37 l 0.37,0.6 h 0.37 l 0.38,0.3 0.37,0.19 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.2 0.37,0.3 0.37,0.2 h 0.38 l 0.37,0.3 0.37,0.29 h 0.37 l 0.38,0.2 0.37,0.3 0.37,0.3 0.37,0.2 0.38,0.1 h 0.37 l 0.37,0.2 h 0.37 l 0.38,0.2 0.37,0.13 0.37,0.26 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.3 0.38,0.2 h 0.37 l 0.37,0.19 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.2 0.37,0.3 0.37,0.1 0.37,0.1 0.38,0.39 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.3 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.29 0.37,0.2 0.37,0.1 0.37,0.2 0.38,0.3 0.37,0.3 h 0.37 l 0.37,0.3 h 0.38 l 0.37,0.29 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.3 h 0.37 l 0.38,0.2 h 0.37 l 0.37,0.2 h 0.37 0.38 l 0.37,0.09 0.37,0.2 0.37,0.2 h 0.38 l 0.37,0.1 h 0.37 0.37 0.38 l 0.37,0.3 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.3 H 215 l 0.37,0.19 0.37,0.1 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.3 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.3 0.37,0.1 h 0.38 l 0.37,0.19 0.37,0.2 0.37,0.1 0.38,0.3 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.3 0.37,0.1 h 0.37 0.38 0.37 l 0.37,0.2 0.37,0.09 0.38,0.4 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.39 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.09 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.3 h 0.38 0.37 l 0.37,0.1 0.37,0.2 h 0.38 0.37 l 0.37,0.19 h 0.37 l 0.38,0.3 0.37,0.1 h 0.37 0.37 0.38 l 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.3 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.2 h 0.37 0.37 l 0.37,0.3 0.38,0.19 0.37,0.3 h 0.37 l 0.37,0.06 0.38,0.34 0.37,0.1 0.37,0.1 0.37,0.08 0.38,0.02 0.37,0.2 0.37,0.1 h 0.37 0.38 l 0.37,0.1 h 0.37 l 0.37,0.29 0.38,0.2 0.37,0.3 h 0.37 0.37 l 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.1 0.38,0.1 h 0.37 0.37 l 0.37,0.39 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.2 h 0.38 0.37 l 0.37,0.2 0.37,0.2 0.38,0.1 0.37,0.2 h 0.37 0.37 l 0.38,0.2 0.37,0.09 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.1 0.37,0.3 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.49 h 0.37 l 0.37,0.2 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.3 h 0.37 l 0.37,0.19 0.37,0.02 0.38,0.08 0.37,0.2 0.37,0.3 h 0.37 0.38 l 0.37,0.3 h 0.37 0.37 l 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.1 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.2 0.38,0.19 0.37,0.2 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.3 h 0.37 l 0.37,0.3 h 0.38 l 0.37,0.3 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.09 0.37,0.3 h 0.37 0.38 l 0.37,0.3 h 0.37 l 0.37,0.2 0.38,0.3 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 0.37 l 0.37,0.1 h 0.38 l 0.37,0.19 0.37,0.2 h 0.37 l 0.38,0.3 0.37,0.4 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.19 0.37,0.1 h 0.37 l 0.37,0.2 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.3 0.37,0.4 h 0.37 l 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.09 0.37,0.5 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.2 0.38,0.09 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.19 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.3 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.09 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.3 h 0.37 l 0.37,0.2 0.37,0.2 0.38,0.04 0.37,0.26 h 0.37 l 0.37,0.2 h 0.38 l 0.37,0.2 0.37,0.19 0.37,0.1 h 0.38 l 0.37,0.3 0.37,0.3 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.29 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.2 0.37,0.29 0.37,0.11 0.37,0.1 0.38,0.09 0.37,0.3 0.37,0.2 0.37,0.2 0.38,0.3 h 0.37 0.37 l 0.37,0.2 0.38,0.59 0.37,0.3 0.37,0.2 0.37,0.1 h 0.38 0.37 l 0.37,0.4 0.37,0.3 0.38,0.2 0.37,0.1 0.37,0.19 h 0.37 l 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.4 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.12 0.38,0.18 0.37,0.09 0.37,0.2 h 0.37 l 0.38,0.1 0.37,0.4 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.21 0.37,0.19 0.38,0.1 0.37,0.19 0.37,0.2 0.37,0.3 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.2 h 0.38 0.37 l 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.39 h 0.37 l 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.2 0.38,0.2 0.37,0.16 0.37,0.14 0.37,0.1 h 0.38 l 0.37,0.19 0.37,0.1 0.37,0.1 0.38,0.27 0.37,0.03 0.37,0.1 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.3 0.37,0.1 0.37,0.09 h 0.37 l 0.38,0.2 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.4 0.37,0.19 h 0.38 l 0.37,0.2 0.37,0.4 0.37,0.2 0.38,0.3 0.37,0.2 h 0.37 l 0.37,0.1 0.38,0.3 0.37,0.29 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.1 h 0.37 0.38 l 0.37,0.1 0.37,0.09 0.37,0.11 0.38,0.3 0.37,0.2 0.37,0.1 0.37,0.19 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.1 h 0.38 0.37 l 0.37,0.2 h 0.37 l 0.38,0.19 0.37,0.01 0.37,0.1 0.37,0.1 h 0.38 0.37 l 0.37,0.29 0.37,0.1 0.38,0.4 0.37,0.1 0.37,0.2 0.37,0.3 h 0.38 l 0.37,0.1 0.37,0.59 0.37,0.2 0.38,0.1 0.37,0.3 h 0.37 l 0.37,0.2 0.38,0.3 0.37,0.1 h 0.37 l 0.37,0.02 0.38,0.08 0.37,0.1 0.37,0.19 0.37,0.1 0.38,0.3 0.37,0.1 0.37,0.1 0.37,0.52 0.38,0.18 h 0.37 0.37 0.37 l 0.38,0.1 h 0.37 l 0.37,0.29 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.3 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.4 0.37,0.09 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.1 0.37,0.29 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.3 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.3 0.37,0.08 0.37,0.41 0.38,0.3 h 0.37 0.37 0.37 l 0.38,0.3 0.37,0.2 0.37,0.2 0.37,0.2 0.38,0.39 0.37,0.2 0.37,0.2 0.37,0.51 0.38,0.09 0.37,0.4 h 0.37 l 0.37,0.09 0.38,0.2 h 0.37 l 0.37,0.3 0.37,0.4 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.19 h 0.37 l 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.4 h 0.38 l 0.37,0.26 0.37,0.14 0.37,0.1 h 0.37 l 0.38,0.19 0.37,0.2 0.37,0.3 0.37,0.48 0.38,0.02 0.37,0.59 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.2 0.37,0.3 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.2 h 0.37 l 0.38,0.2 0.37,0.49 h 0.37 l 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.3 0.38,0.3 0.37,0.29 0.37,0.2 0.37,0.2 0.38,0.2 h 0.37 0.37 l 0.37,0.1 0.38,0.34 0.37,0.36 0.37,0.1 0.37,0.09 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.2 h 0.37 0.37 0.37 l 0.38,0.3 h 0.37 0.37 l 0.37,0.2 0.38,0.4 0.37,0.29 h 0.37 l 0.37,0.3 0.38,0.1 h 0.37 0.37 l 0.37,0.2 0.38,0.3 0.37,0.2 0.37,0.3 0.37,0.1 0.38,0.29 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.5 h 0.38 0.37 l 0.37,0.29 0.37,0.3 0.38,0.1 h 0.37 l 0.37,0.4 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.59 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.2 0.38,0.1 h 0.37 l 0.37,0.29 0.37,0.1 0.38,0.3"
             style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2948" />
          <path
             d="m 83.51,89.73 v 0.1 0.1 0.1 0.1 0.1 l 0.37,10.04 0.37,7.16 0.37,13.43 0.38,6.06 0.37,8.55 0.37,5.67 0.37,3.28 0.38,7.96 0.37,3.48 0.37,4.08 0.37,3.18 0.38,2.38 0.37,2.49 0.37,1.97 0.37,2.6 0.38,1.79 0.37,3.09 0.37,1.59 0.37,2.09 0.38,2.08 0.37,2.39 0.37,1.59 0.37,1.29 0.38,1.5 0.37,1.98 0.37,1.99 0.37,1.69 0.38,1.89 0.37,1.39 0.37,1.6 0.37,1.79 0.38,1.59 0.37,1.98 0.37,2.79 0.37,1.09 0.38,2.03 0.37,1.55 0.37,2.89 0.37,2.68 0.38,3.68 0.37,1.19 0.37,1.1 0.37,0.79 0.38,1.79 0.37,1.3 0.37,1.59 0.37,1.69 0.38,1.59 0.37,1.19 0.37,1.39 0.37,1.69 0.38,0.8 0.37,0.99 0.37,2.29 0.37,1.29 0.38,1.69 0.37,1.3 0.37,1.59 0.37,0.89 0.38,1.39 0.37,1.79 0.37,0.8 0.37,1.48 0.38,2.2 0.37,1.29 0.37,1.49 0.37,0.9 0.38,2.28 0.37,1.63 0.37,1.16 0.37,1.89 0.38,1.79 0.37,1.99 0.37,1.29 0.37,1.89 0.38,1.79 0.37,1.49 0.37,1.49 0.37,1.2 0.38,1.09 0.37,1.79 0.37,0.3 0.37,1.59 0.38,1.39 0.37,1.39 0.37,1.59 0.37,1.3 0.38,0.79 0.37,0.7 0.37,1.49 0.37,1.29 0.38,1.1 0.37,1.68 0.37,1.1 0.37,1.29 0.38,0.8 0.37,0.79 0.37,1.2 0.37,0.3 0.38,1.19 0.37,0.89 0.37,0.3 0.37,0.5 0.38,0.92 0.37,0.67 0.37,0.3 0.37,0.69 0.38,0.8 0.37,0.2 0.37,0.3 0.37,0.59 0.38,0.9 0.37,0.1 0.37,0.3 0.37,0.39 0.38,0.4 0.37,1 0.37,0.19 0.37,0.8 0.38,0.1 0.37,0.5 0.37,0.3 h 0.37 l 0.38,0.19 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.3 0.37,0.2 0.37,0.4 0.37,0.1 0.38,0.1 0.37,0.49 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.1 0.37,0.3 0.37,0.49 0.38,0.3 0.37,0.5 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.19 0.37,0.3 0.37,0.3 0.38,0.2 0.37,0.6 0.37,0.39 0.37,0.4 0.38,0.1 0.37,0.4 0.37,0.5 0.37,0.29 0.38,0.2 0.37,0.4 0.37,0.5 0.37,0.4 0.38,0.49 0.37,0.4 0.37,0.4 0.37,0.4 0.38,0.1 0.37,0.19 0.37,0.3 0.37,0.2 0.38,0.2 0.37,0.3 0.37,0.1 0.37,0.58 0.38,0.51 0.37,0.1 0.37,0.4 0.37,0.2 0.37,0.5 0.38,0.49 0.37,0.3 0.37,1 0.37,0.29 0.38,0.5 0.37,0.1 0.37,0.2 0.37,0.27 0.38,0.33 0.37,0.19 0.37,0.4 0.37,0.2 0.38,0.3 0.37,0.4 0.37,0.2 0.37,0.29 0.38,0.1 0.37,0.4 0.37,0.28 0.37,0.32 0.38,0.5 0.37,0.29 0.37,0.1 0.37,0.2 0.38,0.3 0.37,0.1 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.19 0.37,0.3 0.37,0.2 h 0.38 l 0.37,0.3 0.37,0.1 0.37,0.4 0.38,0.4 0.37,0.29 0.37,0.6 0.37,0.2 0.38,0.89 0.37,0.2 0.37,0.4 0.37,0.2 0.38,0.3 0.37,0.3 0.37,0.1 0.37,0.39 0.38,0.2 0.37,0.4 0.37,0.3 0.37,0.3 0.38,0.59 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.1 0.37,0.4 h 0.37 0.37 l 0.38,0.1 0.37,0.29 0.37,0.3 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.5 0.37,0.1 0.38,0.1 0.37,0.3 0.37,0.29 0.37,0.1 0.38,0.48 0.37,0.12 h 0.37 0.37 l 0.38,0.6 0.37,0.3 0.37,0.19 0.37,0.1 0.38,0.3 h 0.37 l 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.2 0.37,0.39 0.37,0.1 h 0.38 l 0.37,0.4 0.37,0.2 0.37,0.2 h 0.38 l 0.37,0.2 0.37,0.5 0.37,0.19 0.38,0.4 0.37,0.4 h 0.37 0.37 0.38 l 0.37,0.1 h 0.37 l 0.37,0.3 0.38,0.46 0.37,0.13 0.37,0.3 0.37,0.3 0.38,0.4 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.19 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.3 0.38,0.2 h 0.37 l 0.37,0.3 0.37,0.3 0.38,0.09 0.37,0.3 0.37,0.1 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.5 0.37,0.2 0.38,0.49 0.37,0.2 0.37,0.4 0.37,0.1 0.38,0.2 0.37,0.2 h 0.37 0.37 l 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.39 0.38,0.3 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.4 0.37,0.3 0.37,0.1 0.37,0.09 0.38,0.3 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.3 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.19 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.1 0.37,0.4 h 0.37 l 0.37,0.19 h 0.38 0.37 l 0.37,0.2 0.37,0.3 0.38,0.4 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.18 0.37,0.22 0.37,0.19 0.37,0.4 0.38,0.1 0.37,0.2 0.37,0.2 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.1 0.37,0.39 0.38,0.2 0.37,0.2 0.37,0.4 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.18 0.37,0.02 0.38,0.19 0.37,0.3 h 0.37 l 0.37,0.29 0.38,0.11 h 0.37 l 0.37,0.2 0.37,0.3 0.38,0.3 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.29 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.29 0.37,0.2 0.37,0.4 0.38,0.4 0.37,0.2 0.37,0.1 0.37,0.49 0.38,0.1 0.37,0.2 0.37,0.04 0.37,0.56 h 0.38 l 0.37,0.1 0.37,0.3 0.37,0.1 0.37,0.19 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.7 0.37,0.2 0.37,0.39 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.3 0.38,0.1 h 0.37 0.37 0.37 l 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.39 h 0.37 0.37 l 0.38,0.3 0.37,0.3 0.37,0.21 0.37,0.19 0.38,0.4 0.37,0.2 h 0.37 l 0.37,0.19 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.3 0.38,0.5 0.37,0.2 0.37,0.3 0.37,0.29 0.38,0.3 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.4 0.37,0.2 0.37,0.1 0.37,0.39 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.4 0.37,0.29 0.38,0.4 0.37,0.1 0.37,0.3 0.37,0.3 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.1 0.38,0.19 0.37,0.1 0.37,0.4 h 0.37 l 0.38,0.3 0.37,0.2 0.37,0.2 0.37,0.2 0.38,0.1 0.37,0.19 0.37,0.2 0.37,0.2 h 0.38 l 0.37,0.4 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.3 h 0.37 l 0.37,0.29 0.38,0.2 0.37,0.1 0.37,0.2 h 0.37 l 0.38,0.3 0.37,0.2 0.37,0.3 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.19 h 0.37 l 0.38,0.1 0.37,0.2 0.37,0.2 0.37,0.36 0.38,0.24 0.37,0.1 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.09 h 0.37 l 0.37,0.1 h 0.38 l 0.37,0.38 0.37,0.32 0.37,0.4 0.38,0.2 0.37,0.1 0.37,0.19 h 0.37 l 0.38,0.2 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.2 0.37,0.2 0.38,0.09 0.37,0.4 0.37,0.2 H 288 l 0.38,0.4 h 0.37 l 0.37,0.2 0.37,0.2 0.38,0.3 0.37,0.09 0.37,0.1 h 0.37 l 0.38,0.4 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.2 h 0.37 0.38 0.37 l 0.37,0.09 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.1 h 0.37 l 0.38,0.2 h 0.37 l 0.37,0.2 h 0.37 l 0.38,0.2 0.37,0.29 h 0.37 0.37 l 0.38,0.3 0.37,0.3 0.37,0.1 h 0.37 0.38 l 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.3 0.37,0.2 h 0.37 0.37 l 0.38,0.07 0.37,0.13 h 0.37 0.37 l 0.38,0.19 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.3 0.37,0.1 0.37,0.1 h 0.38 0.37 l 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.29 0.37,0.2 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.2 0.37,0.3 0.38,0.1 0.37,0.4 0.37,0.2 h 0.37 0.38 l 0.37,0.49 0.37,0.2 h 0.37 l 0.38,0.1 0.37,0.3 0.37,0.2 0.37,0.07 0.38,0.03 h 0.37 0.37 0.37 l 0.38,0.1 0.37,0.1 0.37,0.29 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.2 h 0.37 0.37 l 0.37,0.1 0.37,0.2 0.38,0.5 0.37,0.09 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.1 h 0.37 0.37 l 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.1 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.19 0.38,0.2 h 0.37 0.37 0.37 l 0.38,0.2 0.37,0.4 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.3 0.37,0.1 0.38,0.29 0.37,0.1 0.37,0.4 0.37,0.4 0.38,0.1 0.37,0.4 0.37,0.1 h 0.37 0.38 l 0.37,0.09 h 0.37 l 0.37,0.1 0.38,0.14 0.37,0.24 0.37,0.32 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.09 h 0.37 l 0.37,0.03 0.38,0.07 0.37,0.2 0.37,0.2 h 0.37 l 0.38,0.2 0.37,0.1 h 0.37 0.37 l 0.38,0.2 0.37,0.1 h 0.37 0.37 l 0.38,0.2 h 0.37 0.37 l 0.37,0.1 h 0.38 l 0.37,0.21 0.37,0.18 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.3 0.37,0.2 h 0.37 l 0.38,0.1 0.37,0.2 0.37,0.09 h 0.37 l 0.38,0.3 0.37,0.3 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.4 0.37,0.09 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.5 0.37,0.1 0.37,0.4 h 0.38 l 0.37,0.19 h 0.37 0.37 l 0.38,0.2 0.37,0.1 0.37,0.2 h 0.37 0.38 l 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.4 0.37,0.29 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.1 h 0.37 0.38 l 0.37,0.4 0.37,0.2 0.37,0.2 0.38,0.2 h 0.37 l 0.37,0.29 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.3 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.19 h 0.38 l 0.37,0.3 0.37,0.2 0.37,0.4 0.38,0.3 0.37,0.2 0.37,0.19 0.37,0.1 h 0.38 0.37 0.37 0.37 l 0.38,0.1 0.37,0.2 h 0.37 l 0.37,0.5 0.38,0.2 0.37,0.2 h 0.37 0.37 l 0.38,0.2 h 0.37 0.37 l 0.37,0.1 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.29 0.38,0.4 0.37,0.3 h 0.37 l 0.37,0.1 0.38,0.3 0.37,0.3 0.37,0.1 0.37,0.19 0.38,0.4 0.37,0.2 0.37,0.2 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.39 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.2 h 0.38 0.37 0.37 l 0.37,0.09 0.38,0.2 h 0.37 l 0.37,0.3 0.37,0.1 h 0.38 l 0.37,0.4 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.3 0.38,0.01 0.37,0.38 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.1 h 0.38 0.37 l 0.37,0.3 0.37,0.2 h 0.38 0.37 l 0.37,0.09 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.1 h 0.38 0.37 0.37 0.37 l 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.3 h 0.38 l 0.37,0.2 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.19 h 0.37 l 0.37,0.31 0.38,0.39 0.37,0.2 0.37,0.1 0.37,0.1 h 0.38 0.37 0.37 l 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.09 h 0.37 0.37 l 0.38,0.3 h 0.37 l 0.37,0.2 h 0.37 0.38 0.37 l 0.37,0.2 0.37,0.3 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.21 0.38,0.18 0.37,0.1 h 0.37 0.37 0.38 0.37 l 0.37,0.1 h 0.37 l 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.3 0.37,0.3 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.4 0.37,0.09 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.3 h 0.37 l 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.19 0.38,0.1 h 0.37 0.37 l 0.37,0.3 0.38,0.12 0.37,0.28 0.37,0.1 h 0.37 l 0.38,0.1 0.37,0.2 0.37,0.2 0.37,0.19 h 0.38 0.37 0.37 l 0.37,0.2 0.38,0.2"
             style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2950" />
          <path
             d="m 83.51,89.73 0.37,12.43 0.37,8.45 0.37,6.57 0.38,11.73 0.37,6.96 0.37,5.77 0.37,5.67 0.38,5.66 0.37,4.78 0.37,2.48 0.37,3.09 0.38,2.78 0.37,2.39 0.37,2.27 0.37,2.3 0.38,1.99 0.37,1.59 0.37,2.59 0.37,1.69 0.38,3.68 0.37,2.18 0.37,2.29 0.37,1.59 0.38,1.69 0.37,2.69 0.37,2.38 0.37,2.99 0.38,1.29 0.37,2.58 0.37,1.3 0.37,1.69 0.38,0.69 0.37,2.79 0.37,1.68 0.37,1.8 0.38,2.48 0.37,2.29 0.37,1.29 0.37,2.29 0.38,2.2 0.37,1.48 0.37,1.79 0.37,2.09 0.38,1.39 0.37,0.8 0.37,0.99 0.37,1.89 0.38,1.69 0.37,0.79 0.37,1.1 0.37,2.58 0.38,2.09 0.37,1.39 0.37,1.3 0.37,1.49 0.38,1.39 0.37,1.19 0.37,0.9 0.37,0.89 0.38,1.79 0.37,1.69 0.37,1.59 0.37,1.89 0.38,2.39 0.37,1.89 0.37,0.89 0.37,1.7 0.38,1.49 0.37,0.89 0.37,1.15 0.37,1.64 0.38,1.98 0.37,1.3 0.37,1.99 0.37,1.22 0.38,1.16 0.37,2.09 0.37,0.9 0.37,1.59 0.38,1.79 0.37,1.69 0.37,1.76 0.37,1.32 0.38,1.16 0.37,1.42 0.37,1 0.37,1.59 0.38,1.59 0.37,1.19 0.37,1.5 0.37,0.59 0.38,1.39 0.37,1.5 0.37,0.79 0.37,0.6 0.38,0.59 0.37,1.5 0.37,0.69 0.37,1.3 0.38,0.19 0.37,1.4 0.37,0.59 0.37,0.7 0.38,0.4 0.37,0.69 0.37,0.4 0.37,0.4 0.38,0.89 0.37,0.3 0.37,0.3 0.37,0.7 0.38,0.59 0.37,0.2 0.37,0.6 0.37,0.5 0.38,0.39 0.37,0.4 0.37,0.3 0.37,0.3 0.38,0.49 0.37,0.5 0.37,0.5 0.37,0.5 0.38,0.59 0.37,0.4 h 0.37 l 0.37,0.1 0.38,0.5 h 0.37 l 0.37,0.39 h 0.37 l 0.38,0.4 0.37,0.1 0.37,0.5 0.37,0.5 0.38,0.2 0.37,0.19 0.37,0.5 0.37,0.1 0.38,0.2 0.37,0.4 0.37,0.2 0.37,0.19 0.38,0.7 0.37,0.5 0.37,0.59 0.37,0.4 0.38,0.5 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.59 0.37,0.1 0.37,0.3 0.37,0.2 0.38,0.7 0.37,0.1 0.37,0.2 0.37,0.39 0.38,0.3 0.37,0.3 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.29 0.37,0.5 0.37,0.5 0.38,0.2 h 0.37 l 0.37,0.4 0.37,0.09 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.5 0.37,0.2 0.38,0.35 0.37,0.44 0.37,0.4 0.37,0.5 0.38,0.5 0.37,0.29 0.37,0.3 0.37,0.6 0.38,0.6 0.37,0.19 0.37,0.2 0.37,0.2 0.38,0.4 0.37,0.2 0.37,0.2 0.37,0.4 0.38,0.39 0.37,0.5 0.37,0.1 0.37,0.4 0.38,0.3 0.37,0.1 0.37,0.39 0.37,0.1 0.38,0.1 0.37,0.4 0.37,0.2 0.37,0.5 0.38,0.29 0.37,0.3 0.37,0.4 0.37,0.1 0.38,0.06 0.37,0.14 0.37,0.6 0.37,0.19 0.38,0.2 0.37,0.3 0.37,0.5 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.2 0.37,0.19 0.38,0.4 0.37,0.2 0.37,0.3 0.37,0.3 0.38,0.1 0.37,0.4 0.37,0.29 0.37,0.5 0.38,0.3 0.37,0.1 0.37,0.5 0.37,0.1 0.38,0.19 h 0.37 l 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.3 0.38,0.1 0.37,0.1 0.37,0.49 0.37,0.3 h 0.38 l 0.37,0.5 0.37,0.2 h 0.37 0.38 l 0.37,0.1 0.37,0.2 0.37,0.19 h 0.38 0.37 l 0.37,0.5 0.37,0.5 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.09 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.3 0.37,0.2 0.37,0.5 0.38,0.19 0.37,0.2 0.37,0.2 0.37,0.2 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.39 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.3 0.37,0.19 0.37,0.19 0.38,0.21 0.37,0.3 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.01 0.37,0.19 0.37,0.2 0.38,0.2 0.37,0.09 0.37,0.3 0.37,0.4 h 0.38 l 0.37,0.11 0.37,0.09 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.19 0.38,0.1 0.37,0.2 0.37,0.6 0.37,0.3 h 0.38 l 0.37,0.49 0.37,0.2 h 0.37 0.38 l 0.37,0.6 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.49 0.37,0.37 0.38,0.13 0.37,0.1 0.37,0.3 h 0.37 0.38 l 0.37,0.2 h 0.37 l 0.37,0.3 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.09 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.2 0.38,0.4 0.37,0.3 0.37,0.1 0.37,0.3 0.38,0.49 0.37,0.2 0.37,0.4 0.37,0.1 0.38,0.3 0.37,0.1 0.37,0.1 0.37,0.19 0.38,0.4 h 0.37 l 0.37,0.4 0.37,0.1 0.38,0.3 0.37,0.2 0.37,0.1 0.37,0.29 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.6 0.38,0.3 0.37,0.3 0.37,0.2 h 0.37 l 0.38,0.1 h 0.37 0.37 l 0.37,0.09 0.38,0.2 0.37,0.2 0.37,0.1 0.37,0.1 0.38,0.2 h 0.37 0.37 l 0.37,0.2 h 0.38 0.37 l 0.37,0.6 h 0.37 l 0.38,0.19 0.37,0.2 0.37,0.4 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.3 0.38,0.29 0.37,0.3 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.1 h 0.38 0.37 l 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.1 0.37,0.29 0.38,0.3 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.6 0.37,0.1 0.37,0.19 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.3 0.37,0.2 0.37,0.3 0.37,0.2 0.38,0.1 h 0.37 l 0.37,0.29 0.37,0.3 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.33 0.38,0.27 0.37,0.1 0.37,0.29 0.37,0.2 0.38,0.4 0.37,0.2 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 h 0.37 l 0.37,0.3 0.38,0.1 h 0.37 l 0.37,0.39 h 0.37 l 0.38,0.3 0.37,0.3 0.37,0.1 0.37,0.2 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.4 h 0.38 l 0.37,0.19 h 0.37 0.37 0.38 l 0.37,0.3 h 0.37 0.37 l 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 0.37 l 0.38,0.09 0.37,0.1 h 0.37 l 0.37,0.04 0.38,0.06 h 0.37 0.37 l 0.37,0.1 0.38,0.2 0.37,0.2 0.37,0.3 h 0.37 l 0.38,0.3 0.37,0.1 0.37,0.39 v 0.1 l 0.37,0.3 0.38,0.5 0.37,0.3 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.29 0.38,0.1 0.37,0.1 0.37,0.5 0.37,0.1 0.38,0.28 0.37,0.42 0.37,0.29 0.37,0.08 0.38,0.02 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.4 0.37,0.4 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.26 0.37,0.33 0.37,0.19 0.38,0.21 0.37,0.2 0.37,0.2 0.37,0.3 0.38,0.1 0.37,0.49 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.3 0.37,0.1 0.37,0.1 0.38,0.3 0.37,0.1 0.37,0.19 0.37,0.4 0.38,0.4 0.37,0.4 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.09 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.3 0.37,0.2 0.37,0.3 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.29 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.2 0.37,0.29 h 0.38 0.37 l 0.37,0.4 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.1 0.38,0.2 0.37,0.3 0.37,0.09 0.37,0.2 h 0.38 0.37 l 0.37,0.2 0.37,0.1 0.38,0.3 0.37,0.3 0.37,0.3 0.37,0.2 0.38,0.19 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.4 0.38,0.2 0.37,0.29 0.37,0.3 0.37,0.1 0.38,0.1 0.37,0.5 h 0.37 l 0.37,0.4 0.38,0.1 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.49 0.37,0.3 0.37,0.2 h 0.37 l 0.38,0.4 0.37,0.1 h 0.37 l 0.37,0.2 0.38,0.09 0.37,0.1 0.37,0.2 0.37,0.3 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.3 0.37,0.3 0.37,0.2 0.38,0.09 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.2 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.1 h 0.38 l 0.37,0.2 0.37,0.1 0.37,0.29 0.38,0.1 0.37,0.4 0.37,0.2 0.37,0.1 0.38,0.4 0.37,0.3 0.37,0.2 0.37,0.09 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.5 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.19 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.3 h 0.37 l 0.38,0.2 0.37,0.1 0.37,0.2 0.37,0.4 h 0.38 l 0.37,0.1 0.37,0.09 h 0.37 l 0.38,0.2 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.3 0.37,0.4 h 0.37 l 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.09 0.37,0.1 0.37,0.1 h 0.37 0.38 0.37 l 0.37,0.1 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.4 0.38,0.1 0.37,0.2 h 0.37 0.37 l 0.38,0.69 h 0.37 l 0.37,0.2 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.2 h 0.37 l 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.3 0.38,0.19 h 0.37 l 0.37,0.1 h 0.37 l 0.38,0.2 0.37,0.1 h 0.37 0.37 0.38 0.37 l 0.37,0.14 0.37,0.26 0.38,0.1 h 0.37 l 0.37,0.2 h 0.37 0.38 l 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 0.37,0.1 h 0.37 l 0.38,0.19 h 0.37 0.37 l 0.37,0.1 0.38,0.3 0.37,0.2 h 0.37 0.37 l 0.38,0.1 0.37,0.32 0.37,0.18 0.37,0.1 0.38,0.3 0.37,0.09 0.37,0.1 0.37,0.1 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.3 0.37,0.1 h 0.38 0.37 l 0.37,0.1 h 0.37 0.38 l 0.37,0.2 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.09 0.37,0.4 0.37,0.2 0.38,0.2 0.37,0.3 0.37,0.1 h 0.37 0.38 l 0.37,0.3 0.37,0.1 0.37,0.19 0.38,0.1 0.37,0.1 h 0.37 0.37 l 0.38,0.3 0.37,0.1 h 0.37 0.37 l 0.38,0.4 h 0.37 0.37 l 0.37,0.1 0.38,0.3 0.37,0.29 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.2 h 0.37 l 0.37,0.1 h 0.38 0.37 l 0.37,0.1 0.37,0.3 0.38,0.2 0.37,0.1 0.37,0.3 0.37,0.19 0.38,0.1 0.37,0.2 0.37,0.1 h 0.37 0.38 0.37 0.37 l 0.37,0.17 0.38,0.03 0.37,0.3 0.37,0.2 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.1 0.37,0.49 0.38,0.3 0.37,0.2 0.37,0.3 h 0.37 0.38 0.37 l 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.09 h 0.38 l 0.37,0.1 0.37,0.2 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.1 h 0.38 0.37 0.37 l 0.37,0.2 0.38,0.1 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.2 0.37,0.2 0.37,0.09 0.37,0.1 0.38,0.1 0.37,0.2 0.37,0.1 h 0.37 l 0.38,0.2 h 0.37 0.37 0.37 l 0.38,0.2 0.37,0.3 0.37,0.1 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.1 0.38,0.29 0.37,0.2 0.37,0.2 0.37,0.4 0.38,0.1 h 0.37 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.09 H 418 l 0.37,0.2 h 0.38 l 0.37,0.2 h 0.37 l 0.37,0.2 0.38,0.17 0.37,0.13 h 0.37 l 0.37,0.2 h 0.38 0.37 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 0.37 l 0.38,0.29 h 0.37 l 0.37,0.1 0.37,0.2 0.38,0.1 h 0.37 0.37 l 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.1 0.37,0.1 h 0.38 l 0.37,0.1 0.37,0.1 0.37,0.3 h 0.38 l 0.37,0.1 0.37,0.09 0.37,0.2 0.38,0.3 0.37,0.02 0.37,0.08 h 0.37 l 0.38,0.1 0.37,0.2 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.3 0.37,0.1 h 0.37 0.37 l 0.38,0.1 h 0.37 l 0.37,0.1 0.37,0.29 h 0.38 0.37 l 0.37,0.2 0.37,0.1 h 0.38 0.37 0.37 l 0.37,0.1 0.38,0.1 0.37,0.1 h 0.37 l 0.37,0.1 h 0.38 0.37 0.37 l 0.37,0.1 0.38,0.2 0.37,0.1 h 0.37 l 0.37,0.1 0.38,0.2 0.37,0.3 0.37,0.09 h 0.37 0.38 l 0.37,0.2 0.37,0.1 0.37,0.3 h 0.38 0.37 0.37 l 0.37,0.1 0.38,0.1 h 0.37 l 0.37,0.3 0.37,0.2 0.38,0.2 0.37,0.1 0.37,0.09 0.37,0.2 h 0.38 0.37 l 0.37,0.1 h 0.37 l 0.38,0.1"
             style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2952" />
          <path
             d="M 64.88,71.11 474.62,480.85"
             style="fill:none;stroke:#bebebe;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:1.07, 3.2;stroke-dashoffset:0;stroke-opacity:1"
             id="path2954" />
          <path
             d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path2956" />
        </g>
      </g>
      <g
         id="g2958" />
      <g
         id="g2960" />
      <g
         id="g2962" />
      <g
         id="g2964" />
      <g
         id="g2966" />
      <g
         id="g2968" />
      <g
         id="g2970" />
      <g
         id="g2972" />
      <g
         id="g2974" />
      <g
         id="g2976" />
      <g
         id="g2978" />
      <g
         id="g2980" />
      <g
         id="g2982" />
      <g
         id="g2984" />
      <g
         id="g2986" />
      <g
         id="g2988">
        <text
           transform="matrix(1,0,0,-1,42.44,86.5)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2992"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan2990">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,179.62)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text2996"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan2994">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,272.75)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3000"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan2998">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,365.87)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3004"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3002">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,458.99)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3008"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3006">1.00</tspan></text>
      </g>
      <g
         id="g3010" />
      <g
         id="g3012">
        <path
           d="m 62.14,89.73 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3014" />
        <path
           d="m 62.14,182.86 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3016" />
        <path
           d="m 62.14,275.98 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3018" />
        <path
           d="m 62.14,369.1 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3020" />
        <path
           d="m 62.14,462.22 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3022" />
      </g>
      <g
         id="g3024" />
      <g
         id="g3026" />
      <g
         id="g3028" />
      <g
         id="g3030" />
      <g
         id="g3032" />
      <g
         id="g3034" />
      <g
         id="g3036" />
      <g
         id="g3038" />
      <g
         id="g3040" />
      <g
         id="g3042">
        <path
           d="m 83.51,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3044" />
        <path
           d="m 176.63,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3046" />
        <path
           d="m 269.75,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3048" />
        <path
           d="m 362.87,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3050" />
        <path
           d="m 456,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3052" />
      </g>
      <g
         id="g3054" />
      <g
         id="g3056">
        <text
           transform="matrix(1,0,0,-1,74.75,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3060"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3058">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,167.87,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3064"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3062">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,260.99,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3068"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3066">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,354.12,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3072"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3070">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,447.24,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3076"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3074">1.00</tspan></text>
      </g>
      <g
         id="g3078" />
      <g
         id="g3080" />
      <g
         id="g3082" />
      <g
         id="g3084" />
      <g
         id="g3086" />
      <g
         id="g3088">
        <text
           transform="matrix(1,0,0,-1,235.67,47.1)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3092"><tspan
             x="0 6.1160002 9.1739998 15.598 18.656 25.993 32.109001 38.224998 43.724998 46.167 49.224998 51.667 57.167 59.609001 62.667"
             y="0"
             sodipodi:role="line"
             id="tspan3090">1 − Specificity</tspan></text>
      </g>
      <g
         id="g3094" />
      <g
         id="g3096">
        <text
           transform="matrix(0,1,1,0,37.28,251.22)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3100"><tspan
             x="0 7.3369999 13.453 19.569 25.069 27.511 30.569 33.011002 38.511002 40.952999 44.011002"
             y="0"
             sodipodi:role="line"
             id="tspan3098">Sensitivity</tspan></text>
      </g>
      <g
         id="g3102" />
      <g
         id="g3104" />
      <g
         id="g3106" />
      <g
         id="g3108" />
      <g
         id="g3110" />
      <g
         id="g3112" />
      <g
         id="g3114" />
      <g
         id="g3120" />
      <g
         id="g3122" />
      <g
         id="g3124" />
      <g
         id="g3126">
        <path
           d="m 157.89,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3128" />
      </g>
      <g
         id="g3130" />
      <g
         id="g3132">
        <path
           d="m 159.62,19.6 h 13.83"
           style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3134" />
      </g>
      <g
         id="g3136" />
      <g
         id="g3138">
        <path
           d="m 264.66,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3140" />
      </g>
      <g
         id="g3142" />
      <g
         id="g3144">
        <path
           d="m 266.39,19.6 h 13.82"
           style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3146" />
      </g>
      <g
         id="g3148" />
      <g
         id="g3150">
        <path
           d="m 343.83,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3152" />
      </g>
      <g
         id="g3154" />
      <g
         id="g3156">
        <path
           d="m 345.56,19.6 h 13.82"
           style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3158" />
      </g>
      <g
         id="g3160" />
      <g
         id="g3162" />
      <g
         id="g3164" />
      <g
         id="g3166">
        <text
           transform="matrix(1,0,0,-1,180.65,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3170"><tspan
             x="0 5.0040002 10.008 15.012 17.01 21.51 24.011999 26.01 30.51 33.012001 39.509998 44.514 49.518002 52.514999 57.519001 62.019001 66.518997 68.516998 73.521004"
             y="0"
             sodipodi:role="line"
             id="tspan3168">Logistic Regression</tspan></text>
      </g>
      <g
         id="g3172" />
      <g
         id="g3174" />
      <g
         id="g3176" />
      <g
         id="g3178" />
      <g
         id="g3180">
        <text
           transform="matrix(1,0,0,-1,287.42,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3184"><tspan
             x="0 4.4190001 7.4159999 12.42 17.424 19.926001 26.424 28.926001 31.427999 38.43 43.433998"
             y="0"
             sodipodi:role="line"
             id="tspan3182">Tree w/ GLM</tspan></text>
      </g>
      <g
         id="g3186" />
      <g
         id="g3188" />
      <g
         id="g3190" />
      <g
         id="g3192" />
      <g
         id="g3194">
        <text
           transform="matrix(1,0,0,-1,366.59,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3198"><tspan
             x="0 7.0019999 13.005"
             y="0"
             sodipodi:role="line"
             id="tspan3196">GAM</tspan></text>
      </g>
      <g
         id="g3200" />
      <g
         id="g3202" />
      <g
         id="g3204" />
      <g
         id="g3206" />
      <g
         id="g3208" />
      <g
         id="g3210" />
      <g
         id="g3212">
        <text
           transform="matrix(1,0,0,-1,64.88,489.19)"
           style="font-variant:normal;font-weight:normal;font-size:13px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3216"><tspan
             x="0 9.1260004 19.24 28.625999 32.240002 41.625999 48.854 53.573002 59.748001 66.975998 70.589996 78.181999 81.795998 88.179001 92.377998 99.606003 102.492 109.72 112.606 119.834 127.062 130.67599 140.062 147.28999 150.90401"
             y="0"
             sodipodi:role="line"
             id="tspan3214">ROC Curve − Training Data</tspan></text>
      </g>
      <g
         id="g3218" />
      <g
         id="g3220" />
      <g
         id="g3222" />
      <g
         id="g3224" />
      <g
         id="g3226" />
      <g
         id="g3228" />
    </g>
    <g
       id="g3427"
       inkscape:label="training_PR_AUC_plot"
       transform="matrix(0.29592284,0,0,-0.29592284,127.92744,149.14512)">
      <g
         id="g3429" />
      <g
         id="g3431" />
      <g
         id="g3433" />
      <g
         id="g3435" />
      <g
         id="g3437" />
      <g
         id="g3439" />
      <g
         id="g3441" />
      <g
         id="g3443" />
      <g
         id="g3445" />
      <g
         id="g3447" />
      <g
         id="g3449" />
      <g
         id="g3451" />
      <g
         id="g3453" />
      <g
         id="g3455" />
      <g
         id="g3457" />
      <g
         id="g3459" />
      <g
         id="g3461" />
      <g
         id="g3463" />
      <g
         id="g3465">
        <g
           id="g3467"
           clip-path="url(#clipPath3471)">
          <path
             d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3473" />
        </g>
      </g>
      <g
         id="g3475" />
      <g
         id="g3477">
        <g
           id="g3479"
           clip-path="url(#clipPath3483)">
          <path
             d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path3485" />
          <path
             d="M 64.88,136.29 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3487" />
          <path
             d="M 64.88,229.42 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3489" />
          <path
             d="M 64.88,322.54 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3491" />
          <path
             d="M 64.88,415.66 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3493" />
          <path
             d="M 130.07,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3495" />
          <path
             d="M 223.19,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3497" />
          <path
             d="M 316.31,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3499" />
          <path
             d="M 409.44,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3501" />
          <path
             d="M 64.88,89.73 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3503" />
          <path
             d="M 64.88,182.86 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3505" />
          <path
             d="M 64.88,275.98 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3507" />
          <path
             d="M 64.88,369.1 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3509" />
          <path
             d="M 64.88,462.22 H 474.62"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3511" />
          <path
             d="M 83.51,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3513" />
          <path
             d="M 176.63,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3515" />
          <path
             d="M 269.75,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3517" />
          <path
             d="M 362.87,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3519" />
          <path
             d="M 456,71.11 V 480.85"
             style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3521" />
          <path
             d="m 83.51,462.22 h 0.37 0.37 l 0.37,-56.27 0.38,-5.86 0.37,-39.21 0.37,12.88 0.37,9.98 0.38,7.95 0.37,-0.9 0.37,-13.12 0.37,6.07 0.38,5.25 0.37,-0.41 0.37,-0.35 0.37,4.01 0.38,3.59 0.37,-0.74 0.37,-0.65 0.37,-0.58 0.38,2.81 0.37,2.6 0.37,-15.53 0.37,-2.67 0.38,2.67 0.37,-2.46 0.37,0.09 0.37,-2.15 0.38,0.16 0.37,-7.92 0.37,-1.5 0.37,-1.38 0.38,0.45 0.37,-1.27 0.37,0.45 0.37,-1.17 0.38,1.99 0.37,0.37 0.37,0.36 0.37,0.34 0.38,1.75 0.37,-2.48 0.37,1.67 0.37,1.61 0.38,-1.06 0.37,1.53 0.37,1.48 0.37,0.19 0.38,-2.23 0.37,-2.1 0.37,-0.87 0.37,1.37 0.38,1.33 0.37,1.29 0.37,-0.88 0.37,1.24 0.38,1.21 0.37,1.17 0.37,0.13 0.37,1.12 0.38,1.09 0.37,0.09 0.37,1.04 0.37,-0.89 0.38,-1.79 0.37,-0.8 0.37,1 0.37,-2.54 0.38,-2.42 0.37,0.16 0.37,-1.47 0.37,0.97 0.38,0.95 0.37,-1.42 0.37,0.15 0.37,0.16 0.38,-2.09 0.37,-2 0.37,0.91 0.37,0.89 0.38,-1.23 0.37,0.87 0.37,0.85 0.37,-1.19 0.38,0.17 0.37,0.16 0.37,0.16 0.37,0.8 0.38,-0.49 0.37,-1.1 0.37,-1.08 0.37,0.78 0.38,-0.44 0.37,0.76 0.37,0.74 0.37,0.74 0.38,0.14 0.37,0.13 0.37,-0.45 0.37,-0.43 0.38,0.7 0.37,0.13 0.37,0.12 0.37,-0.42 0.38,0.67 0.37,-0.42 0.37,0.65 0.37,0.65 0.38,0.11 0.37,0.63 0.37,0.62 0.37,0.61 0.38,0.61 0.37,0.59 0.37,0.59 0.37,-0.43 0.38,0.57 0.37,0.57 0.37,-2.4 0.37,0.08 0.38,-0.4 0.37,-0.86 0.37,0.56 0.37,0.08 0.38,0.55 0.37,0.54 0.37,-0.39 0.37,0.08 0.38,-1.28 0.37,0.53 0.37,-0.36 0.37,0.08 0.38,0.08 0.37,-1.21 0.37,-0.76 0.37,0.09 0.38,-0.33 0.37,0.5 0.37,0.09 0.37,-0.73 0.38,-0.31 0.37,-0.31 0.37,0.09 0.37,-0.69 0.38,0.48 0.37,-0.68 0.37,-0.67 0.37,0.48 0.38,-2.15 0.37,0.47 0.37,0.11 0.37,-0.26 0.38,-0.26 0.37,-0.26 0.37,-0.25 0.37,0.11 0.38,-0.03 0.37,-0.47 0.37,-0.24 0.37,-0.57 0.38,-0.24 0.37,0.11 0.37,-0.22 0.37,-0.23 0.38,-1.85 0.37,-0.21 0.37,-1.48 0.37,-0.19 0.38,-1.43 0.37,0.13 0.37,0.13 0.37,-0.48 0.38,-0.47 0.37,-0.17 0.37,-0.76 0.37,-0.16 0.37,-1.32 0.38,-0.43 0.37,-1.84 0.37,-0.13 0.37,-0.13 0.38,-0.13 0.37,-0.67 0.37,-2.81 0.37,-0.89 0.38,-0.35 0.37,-0.87 0.37,-0.34 0.37,-0.85 0.38,-0.33 0.37,-0.82 0.37,0.17 0.37,-0.07 0.38,-0.07 0.37,-0.31 0.37,-0.55 0.37,-0.54 0.38,-0.3 0.37,0.17 0.37,-1.22 0.37,-0.28 0.38,-0.28 0.37,0.4 0.37,-0.28 0.37,-2.05 0.38,-1.12 0.37,-0.7 0.37,-1.4 0.37,-0.57 0.38,-1.06 0.37,-0.84 0.37,-0.41 0.37,-0.42 0.38,-1.6 0.37,-1.17 0.37,-0.97 0.37,-0.18 0.38,0.02 0.37,0.01 0.37,-1.12 0.37,0.37 0.38,-0.52 0.37,-2.18 0.37,-0.16 0.37,-0.51 0.38,0.21 0.37,-1.21 0.37,-0.66 0.37,-0.48 0.38,-0.28 0.37,-0.33 0.37,-0.47 0.37,0.04 0.38,-0.79 0.37,-0.05 0.37,-0.52 0.37,-0.61 0.38,-0.43 0.37,-0.43 0.37,-0.11 0.37,-2.3 0.38,-0.71 0.37,0.36 0.37,-0.7 0.37,-0.24 0.38,-1.28 0.37,0.07 0.37,-0.38 0.37,-1.1 0.38,0.36 0.37,0.22 0.37,-0.08 0.37,0.36 0.38,-2.2 0.37,-0.49 0.37,-0.2 0.37,0.35 0.38,-0.34 0.37,-0.6 0.37,-0.33 0.37,-0.06 0.38,-0.46 0.37,-0.32 0.37,-0.84 0.37,0.21 0.38,-0.57 0.37,-0.31 0.37,0.09 0.37,0.08 0.38,-0.05 0.37,-0.3 0.37,-0.55 0.37,-0.03 0.38,-1.41 0.37,-0.21 0.37,-0.23 0.37,-0.15 0.38,-0.28 0.37,-0.27 0.37,-0.62 0.37,-0.85 0.38,-0.03 0.37,0.32 0.37,-0.95 0.37,-0.13 0.38,-0.37 0.37,-0.36 0.37,-0.24 0.37,-0.13 0.38,0.2 0.37,-1.01 0.37,-0.12 0.37,0.31 0.38,-0.02 0.37,-1.41 0.37,-0.87 h 0.37 l 0.38,-0.01 0.37,-0.01 0.37,0.2 0.37,-0.63 0.38,-0.31 0.37,-0.93 0.37,-0.51 0.37,0.2 0.38,-0.2 0.37,-0.4 0.37,-0.69 0.37,0.01 0.38,-1.27 0.37,-0.28 0.37,0.11 0.37,0.2 0.38,-0.76 0.37,-0.08 0.37,-0.65 0.37,0.1 0.38,-0.17 0.37,-0.91 0.37,0.11 0.37,-0.45 0.38,-0.52 0.37,-0.35 0.37,0.02 0.37,0.29 0.38,-0.87 0.37,-0.16 0.37,0.11 0.37,-0.07 0.38,-0.67 0.37,-0.41 0.37,0.11 0.37,-0.07 0.38,-0.01 0.37,0.06 0.37,0.11 0.37,-0.07 0.38,-0.09 0.37,-0.7 0.37,-0.14 0.37,-0.14 0.38,-0.31 0.37,-0.05 0.37,-0.39 0.37,0.19 0.38,-1.34 0.37,0.18 0.37,-0.52 0.37,0.27 0.38,0.19 0.37,-0.79 0.37,0.06 0.37,-0.05 0.38,0.03 0.37,-0.2 0.37,-0.04 0.37,-0.59 0.38,-0.42 0.37,-0.64 0.37,0.26 0.37,0.18 0.38,-0.42 0.37,-1.36 0.37,-0.91 0.37,-0.03 0.38,-0.25 0.37,-0.17 0.37,-0.11 0.37,-0.38 0.38,-0.59 0.37,-0.03 0.37,-0.24 0.37,0.05 0.38,-0.51 0.37,-0.03 0.37,0.05 0.37,0.11 0.38,0.11 0.37,-0.03 0.37,-0.16 0.37,-0.8 0.38,0.01 0.37,0.18 0.37,-0.09 0.37,-1.07 0.38,-0.15 0.37,0.17 0.37,-0.28 0.37,0.18 0.38,0.24 0.37,-0.47 0.37,-0.66 0.37,-0.36 0.38,-0.34 0.37,-0.48 0.37,0.17 0.37,-0.01 0.38,-0.14 0.37,0.05 0.37,-0.38 0.37,-0.2 0.38,-0.38 0.37,0.18 0.37,-0.02 0.37,-0.67 0.38,-0.19 0.37,-0.12 0.37,-0.07 0.37,0.17 0.38,-0.13 0.37,-0.07 0.37,-0.3 0.37,-0.82 0.38,-0.07 0.37,0.06 0.37,-0.58 0.37,-0.11 0.37,-0.46 h 0.38 l 0.37,-0.12 0.37,-0.5 0.37,0.11 0.38,-0.17 0.37,-0.05 0.37,0.16 0.37,-0.22 0.38,-0.38 0.37,-0.3 0.37,-0.03 0.37,-0.37 0.38,0.11 0.37,-0.38 0.37,-0.68 0.37,0.06 0.38,0.16 0.37,-0.26 0.37,-0.47 0.37,-0.04 0.38,-0.97 0.37,-0.35 0.37,0.09 0.37,-0.13 0.38,0.01 0.37,0.01 0.37,0.06 0.37,0.21 0.38,-0.55 0.37,-0.53 0.37,0.06 0.37,-0.39 0.38,0.01 0.37,0.01 0.37,-0.42 0.37,-0.19 0.38,-0.13 0.37,-0.37 0.37,0.01 0.37,0.16 0.38,-0.13 0.37,-0.56 0.37,0.02 0.37,0.1 0.38,0.06 0.37,-0.4 0.37,-0.13 0.37,-0.22 0.38,-0.53 0.37,-0.03 0.37,-0.22 0.37,-0.66 0.38,-0.43 0.37,-0.03 0.37,-0.16 0.37,0.12 0.38,-0.65 0.37,-0.12 0.37,-0.85 0.37,-0.11 0.38,0.03 0.37,-0.54 0.37,-0.11 0.37,-0.02 0.38,-0.4 0.37,0.11 0.37,-0.15 0.37,-0.43 0.38,-0.15 0.37,0.03 0.37,-0.31 0.37,-0.26 0.38,-0.1 0.37,-0.14 0.37,-0.82 0.37,-0.22 0.38,-0.19 0.37,-0.19 0.37,-0.29 0.37,0.06 0.38,0.06 0.37,-0.04 0.37,-0.13 v -0.04 l 0.37,-0.09 0.38,-0.24 0.37,-1.02 0.37,0.11 0.37,-0.42 0.38,-0.02 0.37,-0.23 0.37,-0.12 0.37,-0.34 0.38,0.06 0.37,-0.11 0.37,-0.34 0.37,0.14 0.38,-0.34 0.37,-0.05 0.37,-0.13 0.37,0.13 0.38,0.12 0.37,-0.2 0.37,0.17 0.37,-0.29 0.38,-0.47 0.37,-0.33 0.37,0.03 0.37,-1.55 0.38,-0.48 h 0.37 l 0.37,-0.21 0.37,-0.06 0.38,-0.07 0.37,-0.03 0.37,-0.04 0.37,-0.13 0.38,-0.57 0.37,-0.7 0.37,-0.33 0.37,-0.02 0.38,-0.03 0.37,-0.19 0.37,-0.1 0.37,-0.08 0.38,-0.13 0.37,-0.22 0.37,-0.32 0.37,-0.28 0.38,-0.19 0.37,-0.65 0.37,-0.49 0.37,-0.09 0.38,-0.48 0.37,-0.33 0.37,-0.29 0.37,-0.3 0.38,-0.59 0.37,-1 0.37,-0.94 0.37,-0.42 0.38,0.04 0.37,-0.04 0.37,-0.17 0.37,-0.15 0.38,-0.06 0.37,-0.58 0.37,-0.54 0.37,-0.32 0.38,-0.29 0.37,-0.2 0.37,-0.28 0.37,-0.58 0.38,-0.07 0.37,-0.16 0.37,-0.68 0.37,-0.59 0.38,-0.29 0.37,0.08 0.37,-0.55 0.37,-0.26 0.38,-0.65 0.37,0.05 0.37,-0.28 0.37,-0.05 0.38,-0.23 0.37,-0.42 0.37,-0.35 0.37,-0.57 0.38,-0.44 0.37,-0.11 0.37,0.02 0.37,-0.25 0.38,-0.32 0.37,-0.62 0.37,-0.34 0.37,-0.76 0.38,-0.42 0.37,-0.71 0.37,-0.67 0.37,-0.41 0.38,-0.36 0.37,0.06 0.37,-0.46 0.37,-0.14 0.38,-0.58 0.37,-0.5 0.37,-0.23 0.37,-0.21 0.38,-0.31 0.37,-0.24 0.37,-0.05 0.37,-0.35 0.38,-0.17 0.37,-0.94 0.37,-0.75 0.37,-0.3 0.38,-0.31 0.37,-0.12 0.37,-0.34 0.37,-0.16 0.38,-0.21 h 0.37 l 0.37,-0.2 0.37,-0.52 0.38,0.04 0.37,-0.1 0.37,-0.11 0.37,-0.42 0.38,-0.02 0.37,-0.17 0.37,-0.28 0.37,-0.29 0.38,-0.65 0.37,-0.03 0.37,-0.18 0.37,-0.63 0.38,0.06 0.37,-0.56 0.37,-0.51 0.37,-0.68 0.38,0.08 0.37,-0.44 0.37,-1.03 0.37,-0.5 0.38,-0.84 0.37,0.04 0.37,-0.66 0.37,-0.28 0.38,-0.36 0.37,-0.34 0.37,-1.04 0.37,-0.5 0.38,-0.28 0.37,-0.61 0.37,-0.45 0.37,-0.63 0.38,-0.03 0.37,-0.12 0.37,-0.12 0.37,0.01 0.37,-0.73 0.38,-0.32 0.37,-0.6 0.37,-0.17 0.37,-0.42 0.38,-0.3 0.37,0.03 0.37,-0.18 0.37,-0.02 0.38,-0.28 0.37,-0.63 0.37,-0.09 0.37,-0.35 0.38,-0.15 0.37,-0.37 0.37,-0.22 0.37,-0.55 0.38,-0.33 0.37,-0.26 0.37,-0.98 0.37,-0.18 0.38,-0.29 0.37,-0.69 0.37,-0.56 0.37,0.02 0.38,-0.19 0.37,0.01 0.37,-0.34 0.37,-0.04 0.38,-0.04 0.37,-0.16 0.37,-0.27 0.37,-0.37 0.38,-0.52 0.37,-0.52 0.37,-0.35 0.37,-0.78 0.38,-0.55 0.37,-0.24 0.37,-1.29 0.37,-0.58 0.38,-0.44 0.37,-0.87 0.37,-0.26 0.37,-0.37 0.38,-0.37 0.37,-0.12 0.37,-0.16 0.37,-0.45 0.38,-0.16 0.37,-0.03 0.37,-0.02 0.37,-0.16 0.38,-0.52 0.37,-0.52 0.37,-0.03 0.37,-0.52 0.38,-0.25 0.37,-0.79 0.37,-0.2 0.37,-0.64 0.38,-0.29 0.37,-0.63 0.37,-0.05 0.37,-0.34 0.38,-1.05 0.37,-0.37 0.37,-0.33 0.37,-0.76 0.38,-0.64 0.37,-0.33 0.37,-0.38 0.37,-0.73 0.38,-0.35 0.37,-0.21 0.37,-0.56 0.37,-0.29 0.38,-0.45 0.37,-0.35 0.37,-0.21 0.37,-0.65 0.38,-0.37 0.37,-0.35 0.37,-0.16 0.37,-0.34 0.38,-0.41 0.37,-0.22 0.37,-0.29 0.37,-0.23 0.38,-0.6 0.37,-0.79 0.37,-0.45 0.37,-0.78 0.38,-0.73 0.37,-0.11 0.37,-0.49 0.37,-0.62 0.38,-0.54 0.37,-0.25 0.37,-0.48 0.37,-0.4 0.38,-0.25 0.37,-0.44 0.37,-0.18 0.37,-0.91 0.38,-0.06 0.37,-0.44 0.37,-0.68 0.37,-0.63 0.38,-0.09 0.37,-0.14 0.37,-0.4 0.37,-0.32 0.38,-0.28 0.37,-0.23 0.37,-0.35 0.37,-0.74 0.38,-0.45 0.37,-0.38 0.37,-0.27 0.37,-0.88 0.38,-0.36 0.37,-0.09 0.37,-0.35 0.37,-0.17 0.38,-0.27 0.37,-0.34 0.37,-0.56 0.37,-0.28 0.38,-0.12 0.37,-0.37 0.37,-0.16 0.37,-0.4 0.38,-0.11 0.37,-0.07 0.37,-0.45 0.37,-0.18 0.38,-0.38 0.37,-0.37 0.37,-0.23 0.37,-0.11 0.38,-0.33 0.37,0.02 0.37,-0.45 0.37,-0.12 0.38,-0.24 0.37,-0.08 0.37,-0.42 0.37,-0.05 0.38,-0.56 0.37,-0.18 0.37,-0.31 0.37,-0.18 0.38,-0.26 0.37,-0.15 0.37,-0.1 0.37,-0.38 h 0.38 l 0.37,-0.38 0.37,-0.09 0.37,-0.15 0.38,-0.46 0.37,-0.23 0.37,-0.03 0.37,-0.14 0.38,-0.24 0.37,-0.14 0.37,-0.44 0.37,-0.05 0.38,0.01 0.37,-0.32 0.37,-0.11 0.37,0.01 0.38,-0.25 0.37,-0.17 0.37,-0.09 0.37,-0.06 0.38,-0.06 0.37,-0.21 0.37,-0.33 0.37,-0.18 0.38,0.02 0.37,-0.14 0.37,-0.2 0.37,-0.31 0.38,-0.23 0.37,-0.15 0.37,-0.19 0.37,-0.24 0.38,-0.09 0.37,-0.2 0.37,-0.04 0.37,-0.13 0.38,-0.08 0.37,-0.07 0.37,-0.15 0.37,-0.11 0.38,-0.18 0.37,-0.08 0.37,-0.17 0.37,-0.06 0.38,-0.03 0.37,-0.18 0.37,-0.03 h 0.37 l 0.38,-0.01 0.37,-0.23 0.37,-0.03 0.37,-0.11 0.38,-0.11 0.37,-0.11 0.37,-0.02 0.37,-0.15 0.38,-0.08 0.37,-0.2 0.37,-0.03 0.37,-0.17 0.38,-0.03 0.37,-0.13 0.37,-0.06 0.37,-0.14 0.38,-0.19 0.37,-0.08 0.37,-0.03 0.37,-0.21 0.38,-0.06 0.37,-0.08 0.37,-0.17 0.37,-0.1 0.38,-0.13 0.37,-0.14 0.37,-0.07 0.37,-0.05 0.38,-0.18 0.37,-0.22 0.37,-0.15 h 0.37 l 0.38,-0.11 0.37,-0.01 0.37,-0.04 0.37,-0.03 0.37,-0.13 0.38,-0.02 0.37,-0.16 0.37,-0.25 0.37,-0.05 0.38,-0.12 0.37,-0.05 0.37,-0.05 0.37,-0.19 0.38,-0.23 0.37,-0.23 0.37,-0.07 0.37,-0.02 0.38,-0.05 0.37,-0.11 0.37,0.03 0.37,-0.03 0.38,-0.11 0.37,-0.05 0.37,-0.19 0.37,-0.08 0.38,-0.07 0.37,-0.02 h 0.37 l 0.37,-0.27 0.38,-0.1 0.37,-0.14 0.37,-0.01 0.37,-0.14 0.38,-0.08 0.37,-0.06 0.37,-0.1 0.37,0.02 0.38,-0.1 0.37,-0.1 0.37,-0.1 0.37,-0.06 0.38,-0.09 h 0.37 l 0.37,-0.03 0.37,-0.03 0.38,-0.15 0.37,-0.03 0.37,-0.05 0.37,0.01 0.38,-0.07 0.37,0.04 0.37,-0.03 0.37,-0.1 0.38,-0.07 0.37,0.01 0.37,-0.07 0.37,-0.06 0.38,-0.11 0.37,-0.01 0.37,-0.06 0.37,-0.05 0.38,-0.08 0.37,-0.07 0.37,-0.01 0.37,-0.01 0.38,-0.04 0.37,0.03 0.37,-0.13 0.37,-0.04 0.38,-0.07 0.37,-0.07 0.37,0.01 0.37,-0.17 0.38,-0.08 0.37,-0.04 h 0.37 l 0.37,-0.03 0.38,-0.02 0.37,-0.15 h 0.37 l 0.37,-0.01 0.38,-0.17 0.37,-0.17 0.37,-0.1 0.37,0.01 0.38,-0.08 0.37,-0.01 0.37,-0.1 0.37,-0.06 0.38,-0.02 0.37,-0.03 0.37,-0.1 0.37,-0.06 0.38,-0.08 h 0.37 l 0.37,-0.07 0.37,-0.04 0.38,-0.05 0.37,-0.08 0.37,0.01 0.37,-0.07 0.38,-0.08 0.37,-0.02 0.37,-0.05 0.37,-0.07 0.38,-0.02 v 0"
             style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3523" />
          <path
             d="m 83.51,462.22 h 0.37 l 0.37,-78.48 0.37,-19.3 0.38,19.3 0.37,12.93 0.37,9.28 0.37,6.97 0.38,-12.83 0.37,5.86 0.37,-2.37 0.37,-8.13 0.38,-1.01 0.37,4.32 0.37,-1.12 0.37,-0.97 0.38,-0.84 0.37,3.24 0.37,2.95 0.37,-4.64 0.38,2.71 0.37,-0.82 0.37,-12.69 0.37,-5.51 0.38,2.67 0.37,2.5 0.37,-0.1 0.37,-4.68 0.38,2.28 0.37,-2.12 0.37,-1.96 0.37,2.09 0.38,0.06 0.37,0.06 0.37,-3.5 0.37,-6.52 0.38,-1.26 0.37,-1.19 0.37,-2.56 0.37,1.83 0.38,1.76 0.37,0.31 0.37,1.65 0.37,1.6 0.38,0.21 0.37,-3.61 0.37,-2.17 0.37,1.5 0.38,0.28 0.37,1.42 0.37,1.37 0.37,-0.92 0.38,0.22 0.37,1.29 0.37,0.18 0.37,0.18 0.38,0.17 0.37,-0.85 0.37,1.17 0.37,1.14 0.38,1.11 0.37,1.09 0.37,-0.86 0.37,1.04 0.38,1.03 0.37,0.99 0.37,0.06 0.37,0.96 0.38,0.94 0.37,-1.74 0.37,0.92 0.37,0.89 0.38,0.03 0.37,0.02 0.37,-0.8 0.37,0.84 0.38,0.03 0.37,0.02 0.37,0.81 0.37,-0.77 0.38,0.02 0.37,-1.49 0.37,0.04 0.37,0.03 0.38,0.04 0.37,-4.19 0.37,-0.61 0.37,0.77 0.38,0.07 0.37,0.48 0.37,-0.98 0.37,0.72 0.38,-1.21 0.37,0.09 0.37,0.7 0.37,0.08 0.38,-3.58 0.37,-0.49 0.37,0.7 0.37,0.1 0.38,0.1 0.37,0.67 0.37,-1.61 0.37,-0.45 0.38,0.66 0.37,0.1 0.37,-2.06 0.37,0.65 0.38,0.64 0.37,0.1 0.37,0.62 0.37,-0.42 0.38,-0.42 0.37,-0.4 0.37,-0.4 0.37,0.1 0.38,0.11 0.37,-0.39 0.37,0.59 0.37,-0.38 0.38,0.09 0.37,0.57 0.37,0.1 0.37,-0.38 0.38,0.56 0.37,-0.37 0.37,-0.36 0.37,-0.36 0.38,0.1 0.37,-0.35 0.37,0.53 0.37,-1.21 0.38,0.1 0.37,-0.33 0.37,0.52 0.37,0.1 0.38,-1.15 0.37,0.1 0.37,-1.1 0.37,-1.11 0.38,-0.68 0.37,-0.28 0.37,-0.65 0.37,0.12 0.38,-1.77 0.37,-1.72 0.37,0.13 0.37,-1.31 0.38,-0.21 0.37,0.49 0.37,0.14 0.37,-0.22 0.38,-0.21 0.37,-0.21 0.37,0.48 0.37,-0.55 0.38,-0.2 0.37,0.14 0.37,0.13 0.37,0.47 0.38,0.46 0.37,-0.21 0.37,0.46 0.37,0.45 0.38,-0.86 0.37,-0.52 0.37,0.45 0.37,-0.2 0.38,0.13 0.37,-1.44 0.37,-0.49 0.37,0.44 0.38,-1.69 0.37,0.43 0.37,0.13 0.37,-0.76 0.37,-1.33 0.38,0.14 0.37,-0.15 0.37,0.42 0.37,-2.41 0.38,0.43 0.37,0.14 0.37,-0.69 0.37,0.14 0.38,0.14 0.37,-0.13 0.37,0.14 0.37,-1.99 0.38,-1.17 0.37,-0.04 0.37,-2.22 0.37,0.41 0.38,0.16 0.37,0.4 0.37,-0.1 0.37,-0.35 0.38,-0.34 0.37,-1.31 0.37,0.16 0.37,-2.23 0.38,-0.07 0.37,-1.47 0.37,0.17 0.37,-0.07 0.38,-1.42 0.37,-0.06 0.37,-1.16 0.37,-0.49 0.38,-0.04 0.37,-0.05 0.37,-1.76 0.37,-1.51 0.38,-1.68 0.37,0.18 0.37,-0.02 0.37,-0.42 0.38,-0.02 0.37,-0.62 0.37,-0.41 0.37,-0.6 0.38,-0.98 0.37,-0.2 0.37,-0.77 0.37,-1.88 0.38,-1.28 0.37,0.19 0.37,0.19 0.37,-0.53 0.38,-0.35 0.37,0.09 0.37,-1.3 0.37,0.37 0.38,-0.16 0.37,0.02 0.37,0.02 0.37,-0.68 0.38,-1.16 0.37,-0.99 0.37,-0.47 0.37,-0.46 0.38,-0.13 0.37,-0.14 0.37,-1.41 0.37,-0.13 0.38,-0.27 0.37,-0.6 0.37,0.05 0.37,-0.43 0.38,0.04 0.37,-1.64 0.37,-0.55 0.37,0.35 0.38,0.04 0.37,-0.69 0.37,-0.69 0.37,-0.53 0.38,-0.09 0.37,-0.38 0.37,-1.23 0.37,-0.03 0.38,-0.41 0.37,-2.16 0.37,-0.88 0.37,-0.61 0.38,0.07 0.37,-0.72 0.37,-0.46 0.37,-0.58 0.38,-0.18 0.37,-1.08 0.37,0.2 0.37,-0.3 0.38,-0.38 0.37,-0.69 0.37,-0.81 0.37,-0.41 0.38,0.09 0.37,-0.16 0.37,-0.88 0.37,-0.52 0.38,-0.03 0.37,-1.55 0.37,-0.03 0.37,0.21 0.38,-0.37 0.37,-1.28 0.37,0.32 0.37,-0.14 0.38,-0.24 0.37,-0.91 0.37,-0.89 0.37,0.11 0.38,0.08 0.37,-0.8 0.37,-0.52 0.37,-0.11 0.38,-0.75 0.37,-0.63 0.37,-0.42 0.37,0.21 0.38,-0.93 0.37,0.11 0.37,-0.81 0.37,-0.5 0.38,-0.39 0.37,0.1 0.37,-0.09 0.37,-0.19 0.38,-0.39 0.37,0.01 0.37,-0.28 0.37,-0.38 0.38,0.11 0.37,-0.28 0.37,-0.18 0.37,-0.15 0.38,-0.2 0.37,0.19 0.37,0.11 0.37,-0.36 0.38,-0.09 0.37,-0.99 0.37,-0.44 0.37,-0.34 0.38,-0.52 0.37,-0.25 0.37,0.11 0.37,-0.16 0.38,-0.25 0.37,-0.59 0.37,0.03 0.37,-0.59 0.38,0.11 0.37,-0.66 0.37,0.11 0.37,0.19 0.38,-0.06 0.37,-0.14 0.37,0.19 0.37,-0.06 0.38,-0.15 0.37,0.19 0.37,-0.14 0.37,-0.31 0.38,0.19 0.37,0.19 0.37,-0.31 0.37,-0.3 0.38,0.19 0.37,-0.06 0.37,0.1 0.37,0.19 0.38,-0.14 0.37,0.18 0.37,-0.06 0.37,0.1 0.38,-0.37 0.37,-0.76 0.37,-0.37 0.37,-0.2 0.38,-0.59 0.37,-0.36 0.37,-0.57 0.37,-0.5 0.38,-1.16 0.37,0.11 0.37,-0.33 0.37,-0.19 0.38,0.11 0.37,-0.69 0.37,-0.18 0.37,0.03 0.38,-0.12 0.37,-0.94 0.37,0.25 0.37,-0.52 0.38,-0.26 0.37,-0.15 0.37,0.1 0.37,0.18 0.38,-0.92 0.37,-0.16 0.37,-0.37 0.37,-0.63 0.38,-0.16 0.37,0.18 0.37,-0.6 0.37,-0.37 0.38,-0.09 0.37,-0.02 0.37,-0.09 0.37,-0.21 0.38,-0.79 0.37,-0.14 0.37,-0.27 0.37,-0.21 0.38,0.17 0.37,-0.39 0.37,-0.95 0.37,-0.56 0.38,-0.19 0.37,0.05 0.37,-0.92 0.37,0.17 0.38,-0.19 0.37,0.23 0.37,-0.36 0.37,0.05 0.38,-0.01 0.37,-0.3 0.37,-0.13 0.37,-0.35 0.38,-0.59 h 0.37 l 0.37,-0.06 0.37,-0.12 0.37,-0.13 0.38,-0.73 0.37,-0.06 0.37,0.06 0.37,-0.28 0.38,-0.5 0.37,0.16 0.37,-0.11 0.37,-0.93 0.38,-0.2 0.37,-0.01 0.37,-0.16 0.37,-0.26 0.38,-0.05 0.37,-0.37 0.37,-0.21 0.37,-0.41 0.38,0.16 0.37,0.11 0.37,-0.26 0.37,0.01 0.38,-0.82 0.37,-0.25 0.37,-0.09 0.37,-0.05 0.38,-0.19 0.37,0.01 0.37,-0.04 0.37,-0.1 0.38,-0.09 0.37,0.01 0.37,-0.29 0.37,-0.14 0.38,-0.28 0.37,-0.19 0.37,-0.33 0.37,0.07 0.38,-0.09 0.37,-0.28 0.37,-0.51 0.37,-0.18 0.38,-0.37 0.37,0.07 0.37,-0.13 0.37,0.15 0.38,0.2 0.37,-0.08 0.37,-0.08 0.37,0.16 0.38,-0.18 0.37,-0.03 0.37,-0.49 0.37,0.06 0.38,-0.56 0.37,-0.19 0.37,-0.07 0.37,0.1 0.38,0.02 0.37,-0.3 0.37,-0.01 0.37,0.04 0.38,-0.07 0.37,-0.03 0.37,-0.21 0.37,-0.07 0.38,-0.29 0.37,0.06 0.37,0.06 0.37,0.02 0.38,-0.11 0.37,-0.07 0.37,-0.07 0.37,-0.42 0.38,-0.02 0.37,-0.03 0.37,-0.28 0.37,0.11 0.38,-0.11 0.37,-0.07 0.37,-0.32 0.37,-0.06 0.38,0.1 0.37,0.06 0.37,-0.07 v -0.04 -0.04 -0.04 -0.04 -0.04 l 0.37,-0.07 0.38,-0.06 0.37,-0.15 0.37,0.02 0.37,-0.02 0.38,0.06 0.37,-0.35 0.37,-0.14 0.37,0.06 0.38,-0.5 0.37,-0.58 0.37,0.18 0.37,-0.14 0.38,-0.53 0.37,0.06 0.37,-0.05 0.37,0.06 0.38,-0.1 0.37,-0.98 0.37,-0.09 0.37,-0.06 0.38,-0.28 0.37,0.08 0.37,-0.11 0.37,-0.31 0.38,-0.07 h 0.37 l 0.37,-0.16 0.37,-0.05 0.38,-0.01 0.37,-0.21 0.37,0.07 0.37,-0.38 0.38,-0.05 0.37,0.06 0.37,-0.04 0.37,0.02 0.38,-0.3 0.37,-0.19 0.37,-0.15 0.37,-0.48 0.38,-0.54 0.37,-0.5 0.37,0.02 0.37,-0.09 0.38,0.12 0.37,-0.29 0.37,-0.15 0.37,-0.1 0.38,0.01 0.37,-0.13 0.37,-0.35 0.37,0.03 0.38,-0.14 0.37,-0.41 0.37,0.13 0.37,0.06 0.38,-0.07 0.37,-0.1 0.37,-0.01 0.37,-0.38 0.38,-0.19 0.37,0.06 0.37,-0.16 0.37,-0.11 0.38,-0.43 0.37,-0.32 0.37,-0.29 0.37,-0.29 0.38,-0.23 0.37,0.07 0.37,-0.48 0.37,-0.44 0.38,-0.19 0.37,0.04 0.37,-0.28 0.37,-0.18 0.38,-0.25 0.37,-0.82 0.37,-0.51 0.37,-0.33 0.38,-0.23 0.37,-0.29 0.37,-0.17 0.37,-0.53 0.38,-0.51 0.37,-0.4 0.37,0.1 0.37,-0.66 0.38,-0.82 0.37,-0.71 0.37,-0.43 0.37,-0.18 0.38,-0.04 0.37,-1.6 0.37,-0.19 0.37,-0.95 0.38,-0.35 0.37,0.07 0.37,-0.22 0.37,-0.73 0.38,-0.27 0.37,-0.16 0.37,-1.04 0.37,-0.78 0.38,-1.84 0.37,-1.34 0.37,-0.8 0.37,-0.63 0.38,-1.8 0.37,-0.03 0.37,-0.84 0.37,-0.72 0.38,-1.53 0.37,-0.05 0.37,-0.86 0.37,-0.03 0.38,-0.59 0.37,-0.44 0.37,-1.13 0.37,-1.2 0.38,-0.38 0.37,-0.27 0.37,-0.4 0.37,-0.4 0.38,-0.21 0.37,-0.75 0.37,-0.39 0.37,-0.38 0.38,-0.49 0.37,-0.45 0.37,-0.4 0.37,-0.61 0.38,-0.06 0.37,-0.3 0.37,-0.53 0.37,-0.37 0.38,-0.17 0.37,-0.97 0.37,-0.69 0.37,-0.57 0.38,-0.81 0.37,-0.2 0.37,-0.12 0.37,-0.67 0.38,-0.28 0.37,-0.51 0.37,-0.4 0.37,0.02 0.38,-0.47 0.37,-0.16 0.37,-0.01 0.37,-0.03 0.38,-0.42 0.37,-0.32 0.37,-0.74 0.37,-0.57 0.37,-0.37 0.38,-0.35 0.37,-0.65 0.37,-0.16 0.37,-0.47 0.38,-0.78 0.37,-0.36 0.37,-0.21 0.37,-0.27 0.38,-0.22 0.37,-0.75 0.37,-0.48 0.37,-0.61 0.38,-0.32 0.37,-0.87 0.37,-0.19 0.37,-0.76 0.38,-0.63 0.37,-0.22 0.37,-0.21 0.37,-0.23 0.38,-0.4 0.37,-0.06 0.37,-0.03 0.37,-0.06 0.38,-0.41 0.37,-0.44 0.37,-0.21 0.37,-0.63 0.38,-0.18 0.37,-0.38 0.37,-0.06 0.37,-0.45 0.38,-0.08 0.37,-0.24 0.37,-0.3 0.37,-0.61 0.38,-1.02 0.37,-0.34 0.37,-0.65 0.37,-0.5 0.38,-0.38 0.37,-0.59 0.37,-0.26 0.37,-0.22 0.38,-0.85 0.37,-0.07 0.37,-0.25 0.37,-0.5 0.38,-0.68 0.37,-0.18 0.37,-0.36 0.37,-0.25 0.38,-0.58 0.37,-0.49 0.37,-0.48 0.37,-0.16 0.38,-0.23 0.37,-0.24 h 0.37 l 0.37,-1.42 0.38,-0.18 0.37,-0.3 0.37,-0.12 0.37,-0.21 0.38,-0.54 0.37,-0.85 0.37,-0.34 0.37,-0.47 0.38,-0.57 0.37,-0.15 0.37,-0.35 0.37,-0.44 0.38,-0.4 0.37,-0.05 0.37,-0.26 0.37,-0.03 0.38,-0.25 0.37,-0.26 0.37,-0.12 0.37,-1.22 0.38,-0.05 0.37,-0.32 0.37,-0.3 0.37,-0.18 0.38,-0.29 0.37,-0.33 0.37,-0.51 0.37,-0.51 0.38,-0.34 0.37,-0.32 0.37,-0.14 0.37,-0.22 0.38,-0.37 h 0.37 l 0.37,-0.88 0.37,-0.11 0.38,-0.25 0.37,-0.25 0.37,-0.29 0.37,-0.14 0.38,-0.27 0.37,-0.15 0.37,-0.5 0.37,-0.2 0.38,-0.23 0.37,-0.09 0.37,-0.17 0.37,-0.39 0.38,-0.3 0.37,-0.28 0.37,-0.38 0.37,-0.19 0.38,-0.45 0.37,-0.12 0.37,-0.28 0.37,-0.76 0.38,-0.18 0.37,-0.37 0.37,-0.16 0.37,-0.06 0.38,-0.09 0.37,-0.18 0.37,-0.07 0.37,-0.18 0.38,-0.23 0.37,-0.14 0.37,-0.22 0.37,-0.47 0.38,-0.26 0.37,-0.03 0.37,-0.02 0.37,-0.11 0.38,-0.22 0.37,-0.22 0.37,-0.73 0.37,-0.33 0.38,-0.07 0.37,-0.25 0.37,-0.08 0.37,-0.21 0.38,-0.09 0.37,-0.21 0.37,-0.45 0.37,-0.07 0.38,-0.09 0.37,0.01 0.37,-0.15 0.37,-0.14 0.38,-0.28 0.37,0.03 0.37,-0.2 0.37,-0.08 0.38,-0.16 0.37,-0.23 0.37,-0.1 0.37,-0.07 0.38,-0.08 0.37,-0.1 0.37,-0.12 0.37,-0.3 0.38,-0.08 0.37,-0.21 0.37,-0.12 0.37,-0.19 h 0.38 l 0.37,-0.27 0.37,-0.24 0.37,-0.06 0.38,-0.17 0.37,-0.26 0.37,-0.07 0.37,-0.31 0.38,-0.11 0.37,-0.04 0.37,-0.37 0.37,-0.29 0.38,-0.09 0.37,-0.05 0.37,-0.11 0.37,-0.4 0.38,-0.21 0.37,-0.04 0.37,-0.04 0.37,-0.18 0.38,-0.16 0.37,-0.29 0.37,-0.23 0.37,-0.22 0.38,-0.1 0.37,-0.06 0.37,-0.2 0.37,-0.14 0.38,-0.07 0.37,-0.1 0.37,-0.24 0.37,-0.23 0.38,-0.1 0.37,-0.25 0.37,-0.51 0.37,-0.23 0.38,-0.28 0.37,-0.3 0.37,-0.17 0.37,-0.21 0.38,-0.06 0.37,-0.38 0.37,-0.16 0.37,-0.46 0.38,-0.18 0.37,-0.12 0.37,-0.49 0.37,-0.21 0.38,-0.12 0.37,-0.13 0.37,-0.16 0.37,-0.09 0.38,0.01 0.37,-0.22 0.37,-0.05 0.37,-0.13 0.38,-0.02 0.37,-0.55 0.37,-0.12 0.37,-0.09 0.38,-0.32 0.37,-0.06 0.37,-0.01 0.37,-0.26 0.38,-0.29 0.37,-0.14 0.37,-0.1 0.37,-0.2 0.38,-0.26 0.37,-0.07 0.37,-0.19 0.37,-0.02 0.38,-0.09 0.37,-0.06 0.37,-0.05 0.37,0.04 0.37,-0.07 0.38,-0.37 0.37,-0.08 0.37,-0.11 0.37,-0.28 0.38,-0.15 0.37,-0.24 0.37,-0.12 0.37,-0.22 0.38,-0.44 0.37,-0.1 0.37,-0.11 0.37,-0.05 0.38,-0.07 0.37,-0.15 0.37,-0.1 0.37,-0.03 0.38,-0.01 0.37,-0.16 0.37,0.01 0.37,-0.16 0.38,-0.12 0.37,0.02 0.37,-0.03 0.37,-0.12 0.38,-0.19 0.37,-0.22 0.37,-0.11 0.37,-0.01 0.38,-0.11 0.37,-0.22 0.37,-0.08 0.37,-0.09 0.38,-0.12 0.37,-0.07 0.37,-0.09 0.37,-0.08 0.38,-0.03 0.37,-0.14 0.37,-0.08 0.37,-0.04 0.38,-0.02 0.37,0.01 0.37,-0.04 0.37,-0.34 0.38,-0.06 0.37,0.01 0.37,-0.16 0.37,-0.29 h 0.38 l 0.37,-0.03 0.37,-0.11 0.37,0.02 0.38,-0.08 0.37,0.02 0.37,-0.08 0.37,-0.12 0.38,-0.01 h 0.37 l 0.37,-0.06 0.37,-0.05 0.38,-0.2 0.37,-0.23 0.37,-0.12 0.37,-0.06 0.38,-0.07 0.37,-0.13 0.37,-0.06 0.37,0.01 0.38,-0.13 0.37,-0.04 0.37,-0.11 0.37,-0.2 0.38,-0.36 0.37,-0.1 0.37,-0.06 0.37,-0.08 0.38,-0.08 0.37,0.02 0.37,-0.06 0.37,-0.25 0.38,-0.25 0.37,-0.04 0.37,-0.16 0.37,-0.07 0.38,-0.02 0.37,-0.07 0.37,-0.31 0.37,-0.08 0.38,0.01 0.37,-0.06 0.37,-0.05 0.37,-0.16 0.38,-0.14 0.37,-0.03 0.37,-0.1 0.37,-0.16 0.38,-0.04 0.37,-0.01 0.37,-0.1 0.37,-0.16 0.38,-0.01 v -0.01 0 0 0 0 0 0 0 -0.01 0 0 0"
             style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3525" />
          <path
             d="m 83.51,89.73 0.37,294.01 0.37,34.62 0.37,-34.62 0.38,16.35 0.37,10.71 0.37,7.56 0.37,5.62 0.38,4.35 0.37,3.46 0.37,2.81 0.37,-12.68 0.38,3.02 0.37,2.6 0.37,2.26 0.37,1.99 0.38,1.75 0.37,-3.39 0.37,-16.04 0.37,2.22 0.38,2.03 0.37,1.85 0.37,-8.74 0.37,1.92 0.38,-1.36 0.37,-4.16 0.37,-1 0.37,-3.56 0.38,-0.74 0.37,-0.69 0.37,-7.44 0.37,-0.36 0.38,1.75 0.37,-2.4 0.37,-2.22 0.37,1.65 0.38,1.58 0.37,1.5 0.37,-2.08 0.37,-3.63 0.38,-3.36 0.37,-0.08 0.37,1.44 0.37,1.39 0.38,-0.14 0.37,-0.13 0.37,1.28 0.37,1.23 0.38,1.2 0.37,-1.54 0.37,-0.16 0.37,-2.68 0.38,-0.1 0.37,-0.09 0.37,-3.6 0.37,-2.27 0.38,1.12 0.37,-4.29 0.37,-1.98 0.37,1.11 0.38,0.08 0.37,1.07 0.37,0.06 0.37,-1.84 0.38,0.09 0.37,0.09 0.37,0.08 0.37,-0.81 0.38,-0.77 0.37,-0.76 0.37,0.11 0.37,-1.54 0.38,-2.27 0.37,0.15 0.37,0.15 0.37,0.91 0.38,0.89 0.37,-2.12 0.37,0.14 0.37,0.86 0.38,0.13 0.37,0.84 0.37,0.82 0.37,0.81 0.38,0.79 0.37,0.78 0.37,0.77 0.37,0.75 0.38,0.73 0.37,-0.63 0.37,-0.61 0.37,0.72 0.38,0.7 0.37,-1.89 0.37,0.7 0.37,-0.58 0.38,-0.56 0.37,0.07 0.37,-0.55 0.37,0.67 0.38,-1.12 0.37,0.07 0.37,0.07 0.37,0.06 0.38,0.64 0.37,0.62 0.37,-2.72 0.37,0.62 0.38,0.62 0.37,-1.02 0.37,-2.04 0.37,-1.97 0.38,0.61 0.37,0.61 0.37,0.09 0.37,0.09 0.38,0.59 0.37,-0.9 0.37,0.58 0.37,0.09 0.38,0.56 0.37,0.56 0.37,0.55 0.37,-0.4 0.38,-2.25 0.37,-0.83 0.37,0.55 0.37,-0.81 0.38,0.1 0.37,0.09 0.37,0.09 0.37,0.09 0.38,0.09 0.37,-0.34 0.37,0.09 0.37,0.09 0.38,-1.17 0.37,-1.95 0.37,0.51 0.37,-0.3 0.38,0.11 0.37,0.49 0.37,0.1 0.37,-0.68 0.38,-1.06 0.37,-0.27 0.37,0.48 0.37,-0.65 0.38,0.48 0.37,-1 0.37,0.1 0.37,0.48 0.38,-2.06 0.37,0.11 0.37,0.47 0.37,-0.59 0.38,0.11 0.37,-0.93 0.37,0.46 0.37,0.11 0.38,0.11 0.37,-0.89 0.37,-0.55 0.37,0.44 0.38,-0.21 0.37,-0.22 0.37,-0.21 0.37,-0.84 0.38,0.12 0.37,0.43 0.37,-1.14 0.37,-0.19 0.38,0.42 0.37,0.12 0.37,0.42 0.37,0.42 0.37,-0.2 0.38,-0.19 0.37,-1.08 0.37,-0.48 0.37,-0.18 0.38,-1.33 0.37,-0.45 0.37,-1.01 0.37,0.41 0.38,-0.71 0.37,-1.53 0.37,-0.68 0.37,-1.74 0.38,-0.12 0.37,-1.43 0.37,-0.63 0.37,-0.1 0.38,0.15 0.37,-0.36 0.37,-0.61 0.37,-0.84 0.38,0.4 0.37,-1.08 0.37,-1.53 0.37,-0.32 0.38,-0.32 0.37,0.16 0.37,0.16 0.37,-0.77 0.38,-0.54 0.37,-1.66 0.37,-0.28 0.37,-0.72 0.38,-1.16 0.37,-0.7 0.37,-0.04 0.37,-0.47 0.38,-0.04 0.37,-0.46 0.37,-0.67 0.37,-1.48 0.38,-0.43 0.37,-0.23 0.37,-0.43 0.37,-0.62 0.38,-0.41 0.37,-0.02 0.37,-0.01 0.37,-2.14 0.38,-0.77 0.37,-1.49 0.37,-0.37 h 0.37 l 0.38,-0.36 0.37,-1 0.37,-0.78 0.37,-1.74 0.38,0.02 0.37,-1.02 0.37,-1 0.37,-0.64 0.38,0.03 0.37,-0.14 0.37,0.2 0.37,-0.13 0.38,-0.3 0.37,-0.3 0.37,-0.93 0.37,-1.4 0.38,-0.27 0.37,-0.43 0.37,-0.11 0.37,0.35 0.38,-0.57 0.37,0.04 0.37,0.35 0.37,-0.41 0.38,-0.11 0.37,-0.11 0.37,-0.25 0.37,-0.26 0.38,-0.84 0.37,-0.24 0.37,-0.24 0.37,-0.39 0.38,0.05 0.37,-0.23 0.37,-0.38 0.37,-0.37 0.38,-0.36 0.37,-0.51 0.37,-0.49 0.37,-1.17 0.38,-0.48 0.37,-0.08 0.37,-1 0.37,-0.46 0.38,-0.2 0.37,-0.71 0.37,-0.32 0.37,-0.31 0.38,-0.14 0.37,0.15 0.37,-0.18 0.37,0.19 0.38,-0.68 0.37,-0.18 0.37,-0.54 0.37,-0.05 0.38,-0.18 0.37,-0.65 0.37,-0.05 0.37,-0.04 0.38,-0.29 0.37,-0.75 0.37,-0.63 0.37,-0.04 0.38,-0.04 0.37,0.08 0.37,0.31 0.37,0.19 0.38,-0.04 0.37,-0.27 0.37,-0.49 0.37,-0.24 0.38,-1.62 0.37,-0.47 0.37,-0.14 0.37,0.08 0.38,0.3 0.37,-0.35 0.37,-0.35 0.37,0.08 0.38,-1.07 0.37,-0.47 0.37,-0.64 0.37,-0.75 0.38,-0.52 0.37,-0.32 0.37,0.19 0.37,-0.42 0.38,-1.26 0.37,-0.85 h 0.37 l 0.37,-0.3 0.38,0.09 0.37,0.29 0.37,-0.68 0.37,0.02 0.38,-0.12 0.37,0.18 0.37,-0.66 0.37,-0.63 0.38,-0.12 0.37,-0.56 0.37,-0.45 0.37,-0.36 h 0.38 l 0.37,-0.27 0.37,-0.17 0.37,0.11 0.38,-0.28 0.37,-0.17 0.37,0.1 0.37,-0.17 0.38,-0.35 0.37,0.01 0.37,0.01 0.37,0.09 0.38,0.07 0.37,-0.48 0.37,-0.08 0.37,-0.25 0.38,-0.5 0.37,-0.91 0.37,-0.15 0.37,-0.07 0.38,-0.4 0.37,-0.15 0.37,0.02 0.37,0.01 0.38,-0.16 0.37,-0.13 0.37,-0.15 0.37,-0.14 0.38,0.15 0.37,-0.52 0.37,-0.06 0.37,-0.06 0.38,-0.69 0.37,-0.37 0.37,-0.52 0.37,-0.21 0.38,-0.59 0.37,0.02 0.37,0.26 0.37,-0.28 0.38,-0.13 0.37,0.1 0.37,0.17 0.37,-0.57 0.38,-0.57 0.37,0.17 0.37,0.1 0.37,-0.71 0.38,-0.98 0.37,-0.47 0.37,-1.04 0.37,-0.53 0.38,-0.17 0.37,-0.59 0.37,0.03 0.37,-0.27 0.38,0.14 0.37,0.1 0.37,-0.44 0.37,-0.1 0.38,0.03 0.37,-0.29 0.37,-0.51 0.37,-0.88 0.38,-0.49 0.37,-0.41 0.37,-0.54 0.37,-0.28 0.38,-0.34 0.37,0.04 0.37,-0.15 0.37,0.11 0.38,0.17 0.37,-0.15 0.37,0.04 0.37,-0.08 0.38,0.1 0.37,-0.15 0.37,0.17 0.37,-0.33 0.38,-0.21 0.37,-0.2 0.37,-0.32 0.37,-0.5 0.37,-0.14 0.38,0.04 0.37,-0.43 0.37,-0.45 0.37,-0.23 0.38,-0.3 0.37,-0.13 0.37,-0.25 0.37,0.05 0.38,-0.24 0.37,-0.47 0.37,-0.18 0.37,-0.69 0.38,-0.06 0.37,-0.07 0.37,-0.56 0.37,-0.56 0.38,-0.11 0.37,-0.61 0.37,-0.49 0.37,-0.27 0.38,0.05 h 0.37 l 0.37,0.11 0.37,-0.22 0.38,-0.16 0.37,-0.26 0.37,-0.05 h 0.37 l 0.38,-0.63 0.37,-0.2 0.37,-0.09 0.37,0.09 0.38,0.05 0.37,-0.15 0.37,0.05 h 0.37 l 0.38,-0.25 0.37,-0.3 0.37,0.1 0.37,-0.04 0.38,-0.1 0.37,0.05 0.37,0.21 0.37,-0.2 0.38,-0.1 0.37,-0.34 0.37,0.05 0.37,0.2 0.38,-0.42 0.37,-0.3 0.37,-0.19 0.37,-0.38 0.38,-0.28 0.37,0.05 0.37,-0.04 0.37,-0.37 0.38,-0.14 0.37,0.15 0.37,-0.13 0.37,-0.32 0.38,-0.51 0.37,-0.26 0.37,-0.69 0.37,-0.03 0.38,-0.06 0.37,0.04 0.37,-0.35 0.37,0.01 0.38,0.15 0.37,-0.4 0.37,-0.3 0.37,0.02 0.38,0.13 0.37,0.02 0.37,-0.07 0.37,-0.39 0.38,0.01 0.37,-0.11 0.37,-0.16 0.37,0.1 0.38,-0.29 0.37,-0.07 0.37,-0.12 0.37,0.06 0.38,-0.37 0.37,-0.12 0.37,-0.02 0.37,-0.41 0.38,0.06 0.37,-0.32 0.37,0.1 0.37,-0.2 0.38,0.06 0.37,0.06 0.37,-0.07 0.37,-0.39 0.38,-0.19 0.37,-0.31 0.37,-0.03 0.37,0.14 0.38,-0.04 0.37,-0.53 0.37,0.06 0.37,-0.1 0.38,-0.05 0.37,0.04 0.37,0.06 0.37,-0.26 0.38,0.02 0.37,-0.1 0.37,-0.1 0.37,-0.22 0.38,-0.17 0.37,-0.14 0.37,0.13 0.37,-0.05 0.38,-0.08 0.37,-0.08 0.37,-0.17 0.37,-0.06 0.38,-0.08 0.37,-0.26 0.37,-0.46 0.37,-0.01 0.38,-0.11 0.37,-0.06 0.37,-0.01 0.37,-0.54 0.38,-0.12 0.37,-0.39 0.37,0.06 0.37,-0.16 0.38,0.01 0.37,-0.14 0.37,-0.05 0.37,0.05 0.38,-0.3 0.37,0.06 0.37,-0.01 0.37,-0.34 0.38,-0.08 0.37,-0.26 0.37,0.09 0.37,-0.22 0.38,0.06 0.37,-0.36 0.37,-0.5 0.37,-0.01 0.38,-0.17 0.37,-0.09 0.37,-0.21 0.37,-0.12 0.38,-0.07 0.37,0.02 0.37,-0.31 0.37,-0.24 0.38,-0.45 0.37,-0.2 0.37,-0.5 0.37,-0.5 0.38,-0.07 0.37,0.03 0.37,-0.17 0.37,-0.07 0.38,-0.2 0.37,-0.45 0.37,-0.19 0.37,-0.23 0.38,0.13 0.37,-0.32 0.37,-0.72 0.37,-0.22 0.38,0.06 0.37,-0.34 0.37,-0.21 0.37,-0.37 0.38,-0.42 0.37,-0.36 0.37,-0.54 0.37,-0.26 0.38,-0.74 0.37,-0.75 0.37,-0.05 0.37,-0.14 0.38,-0.86 0.37,-0.5 0.37,-0.52 0.37,-0.39 0.38,-0.24 0.37,-0.78 0.37,-0.78 0.37,-0.18 0.38,-0.41 0.37,-0.26 0.37,-0.91 0.37,-0.64 0.38,-0.99 0.37,-0.26 0.37,-0.46 0.37,-0.49 0.38,-0.28 0.37,-0.23 0.37,-0.57 0.37,-0.33 0.38,-1.35 0.37,-0.72 0.37,-1.19 0.37,-0.93 0.38,-0.92 0.37,-0.51 0.37,-0.45 0.37,-0.58 0.38,-0.65 0.37,-0.49 0.37,-1.26 0.37,-0.42 0.38,-0.41 0.37,-0.87 0.37,-0.15 0.37,-0.06 0.38,-0.38 0.37,-0.45 0.37,-0.23 0.37,-0.36 0.38,-1.05 0.37,-0.93 0.37,-0.67 0.37,-0.6 0.38,-0.34 h 0.37 l 0.37,-1.23 0.37,-0.4 0.38,-0.3 0.37,-0.75 0.37,-0.52 0.37,-0.72 0.38,-0.18 0.37,-0.32 0.37,-0.49 0.37,-0.87 0.37,-0.69 0.38,-1.39 0.37,-0.11 0.37,-0.57 0.37,-0.24 0.38,-0.36 0.37,-0.3 0.37,-0.27 0.37,-0.15 0.38,-0.33 0.37,-0.33 0.37,-0.36 0.37,-0.06 0.38,-0.15 0.37,-0.48 0.37,-0.94 0.37,-0.04 0.38,-0.66 0.37,-0.29 0.37,-0.39 0.37,-0.14 0.38,-0.52 0.37,-0.41 0.37,-0.58 0.37,-0.21 0.38,-0.95 0.37,-0.33 0.37,-0.22 0.37,-0.19 0.38,-0.47 0.37,-0.26 0.37,-0.67 0.37,-0.44 0.38,-0.22 0.37,-0.52 0.37,-0.44 0.37,-0.36 0.38,-0.89 0.37,-0.41 0.37,-0.24 0.37,-0.41 0.38,-0.28 0.37,-0.44 0.37,-0.35 0.37,-0.14 0.38,-0.27 0.37,-0.39 0.37,-0.25 0.37,-0.77 0.38,-0.67 0.37,-0.58 0.37,-0.02 0.37,-0.85 0.38,-0.23 0.37,-0.53 0.37,-0.02 0.37,-1.12 0.38,-1.03 0.37,-0.01 0.37,-0.18 0.37,-0.56 0.38,-0.9 0.37,-1.4 0.37,-0.36 0.37,-0.26 0.38,-0.26 0.37,-0.28 0.37,-0.16 0.37,-0.34 0.38,-1.03 0.37,-0.6 0.37,-0.35 0.37,-0.63 0.38,-0.25 0.37,-0.44 0.37,-0.46 0.37,-0.3 0.38,-0.31 0.37,-0.54 0.37,-0.46 0.37,-0.45 0.38,-0.39 0.37,-0.06 0.37,-0.72 0.37,-0.36 0.38,-0.44 0.37,-0.35 0.37,-0.03 0.37,-0.07 0.38,-0.28 0.37,-0.23 0.37,-0.49 0.37,-0.1 0.38,-0.42 0.37,-0.18 0.37,-0.07 0.37,-0.25 0.38,-0.29 0.37,-0.83 0.37,-0.44 0.37,-0.52 0.38,-0.4 0.37,0.02 0.37,-0.33 0.37,-0.14 0.38,-0.01 0.37,-0.21 0.37,-0.18 0.37,-0.24 0.38,-0.36 0.37,-0.06 0.37,-0.2 0.37,-0.29 0.38,-0.34 0.37,-0.35 0.37,-0.24 0.37,-0.05 0.38,-0.15 0.37,-0.31 0.37,-0.77 0.37,-0.37 0.38,-0.87 0.37,0.02 0.37,-0.21 0.37,-0.2 0.38,-0.22 0.37,-0.2 0.37,-0.23 0.37,-0.17 0.38,-0.33 0.37,-0.73 0.37,-0.28 0.37,-0.29 0.38,-0.26 0.37,-0.09 0.37,-0.29 0.37,-0.06 0.38,-0.18 0.37,-0.53 0.37,-0.08 0.37,-0.21 0.38,-0.14 0.37,-0.25 0.37,-0.1 0.37,-0.35 0.38,-0.03 0.37,-0.2 0.37,-0.06 0.37,-0.08 0.38,-0.33 0.37,-0.39 0.37,-0.32 0.37,-0.16 0.38,-0.11 0.37,-0.1 0.37,-0.43 0.37,-0.21 0.38,-0.4 0.37,-0.37 0.37,-0.28 0.37,-0.28 0.38,-0.57 0.37,-0.78 0.37,-0.1 0.37,-0.19 0.38,-0.12 0.37,-0.03 0.37,-0.11 0.37,0.03 0.38,-0.03 0.37,-0.37 0.37,-0.19 0.37,-0.16 0.38,-0.07 0.37,-0.07 0.37,-0.09 0.37,-0.29 0.38,-0.14 0.37,-0.03 0.37,-0.16 0.37,-0.21 0.38,-0.01 0.37,-0.11 0.37,-0.17 0.37,-0.11 0.38,-0.04 0.37,-0.08 0.37,-0.16 0.37,-0.22 0.38,-0.04 0.37,-0.15 0.37,-0.01 0.37,-0.03 0.38,-0.04 0.37,-0.2 0.37,-0.33 0.37,-0.09 0.38,-0.01 0.37,-0.29 0.37,-0.13 0.37,-0.09 0.38,-0.16 0.37,-0.21 0.37,-0.2 0.37,-0.23 0.38,0.02 0.37,-0.24 0.37,-0.13 0.37,-0.07 0.38,-0.27 0.37,-0.12 0.37,-0.04 0.37,-0.08 0.38,-0.09 0.37,-0.24 0.37,-0.08 0.37,-0.09 0.38,-0.01 0.37,-0.05 0.37,-0.18 0.37,0.01 0.38,-0.1 0.37,-0.12 0.37,-0.19 0.37,-0.02 h 0.38 l 0.37,-0.12 0.37,-0.26 0.37,-0.12 0.38,-0.05 0.37,-0.36 0.37,-0.06 0.37,-0.03 0.38,-0.3 0.37,-0.15 0.37,-0.24 0.37,-0.14 0.38,-0.13 0.37,-0.03 0.37,-0.12 0.37,-0.01 0.37,-0.08 0.38,-0.15 0.37,-0.2 0.37,0.01 0.37,-0.22 0.38,-0.12 0.37,-0.15 0.37,-0.02 0.37,-0.17 0.38,-0.06 0.37,-0.27 0.37,-0.14 0.37,-0.09 0.38,-0.1 0.37,-0.07 0.37,-0.21 0.37,-0.44 0.38,-0.11 0.37,-0.08 0.37,-0.16 0.37,0.03 0.38,-0.18 0.37,-0.23 0.37,-0.1 0.37,-0.03 0.38,-0.28 0.37,-0.33 0.37,-0.09 0.37,-0.3 0.38,-0.22 0.37,-0.21 0.37,-0.18 0.37,-0.02 0.38,-0.11 0.37,-0.1 0.37,-0.08 0.37,-0.23 0.38,-0.34 0.37,-0.13 0.37,-0.06 0.37,-0.05 0.38,-0.04 0.37,-0.15 0.37,-0.06 0.37,-0.23 0.38,-0.17 h 0.37 l 0.37,-0.16 0.37,-0.09 0.38,-0.08 0.37,-0.27 0.37,0.02 0.37,-0.1 0.38,-0.07 0.37,-0.26 0.37,-0.08 0.37,-0.07 0.38,-0.09 0.37,0.02 0.37,-0.01 0.37,-0.07 0.38,-0.19 0.37,-0.19 0.37,-0.15 0.37,-0.34 0.38,-0.06 0.37,-0.09 0.37,-0.08 0.37,-0.09 0.38,-0.28 0.37,0.01 0.37,-0.2 0.37,-0.02 0.38,-0.07 0.37,0.01 0.37,-0.19 0.37,-0.19 0.38,-0.18 0.37,-0.05 0.37,-0.11 0.37,-0.26 0.38,-0.14 0.37,-0.07 0.37,-0.12 0.37,-0.16 0.38,-0.07 0.37,-0.09 0.37,-0.02 0.37,-0.18 0.38,-0.29 0.37,-0.18 0.37,-0.09 0.37,-0.25 0.38,-0.24 0.37,-0.12 0.37,-0.01 0.37,-0.13 0.38,-0.15 0.37,-0.08 0.37,-0.05 0.37,-0.06 0.38,-0.15 v -0.01 0 0 0 0 0 0 -0.01 0 0 0 0 0 0 0 -0.01 0 0 0 0 0 0 0 -0.01 0 0 0"
             style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3527" />
          <path
             d="M 64.88,132.08 H 474.62"
             style="fill:none;stroke:#bebebe;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:1.07, 3.2;stroke-dashoffset:0;stroke-opacity:1"
             id="path3529" />
          <path
             d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path3531" />
        </g>
      </g>
      <g
         id="g3533" />
      <g
         id="g3535" />
      <g
         id="g3537" />
      <g
         id="g3539" />
      <g
         id="g3541" />
      <g
         id="g3543" />
      <g
         id="g3545" />
      <g
         id="g3547" />
      <g
         id="g3549" />
      <g
         id="g3551" />
      <g
         id="g3553" />
      <g
         id="g3555" />
      <g
         id="g3557" />
      <g
         id="g3559" />
      <g
         id="g3561" />
      <g
         id="g3563">
        <text
           transform="matrix(1,0,0,-1,42.44,86.5)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3567"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3565">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,179.62)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3571"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3569">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,272.75)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3575"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3573">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,365.87)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3579"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3577">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,42.44,458.99)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3583"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3581">1.00</tspan></text>
      </g>
      <g
         id="g3585" />
      <g
         id="g3587">
        <path
           d="m 62.14,89.73 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3589" />
        <path
           d="m 62.14,182.86 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3591" />
        <path
           d="m 62.14,275.98 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3593" />
        <path
           d="m 62.14,369.1 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3595" />
        <path
           d="m 62.14,462.22 h 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3597" />
      </g>
      <g
         id="g3599" />
      <g
         id="g3601" />
      <g
         id="g3603" />
      <g
         id="g3605" />
      <g
         id="g3607" />
      <g
         id="g3609" />
      <g
         id="g3611" />
      <g
         id="g3613" />
      <g
         id="g3615" />
      <g
         id="g3617">
        <path
           d="m 83.51,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3619" />
        <path
           d="m 176.63,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3621" />
        <path
           d="m 269.75,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3623" />
        <path
           d="m 362.87,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3625" />
        <path
           d="m 456,68.37 v 2.74"
           style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3627" />
      </g>
      <g
         id="g3629" />
      <g
         id="g3631">
        <text
           transform="matrix(1,0,0,-1,74.75,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3635"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3633">0.00</tspan></text>
        <text
           transform="matrix(1,0,0,-1,167.87,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3639"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3637">0.25</tspan></text>
        <text
           transform="matrix(1,0,0,-1,260.99,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3643"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3641">0.50</tspan></text>
        <text
           transform="matrix(1,0,0,-1,354.12,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3647"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3645">0.75</tspan></text>
        <text
           transform="matrix(1,0,0,-1,447.24,59.71)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3651"><tspan
             x="0 5.0040002 7.506 12.51"
             y="0"
             sodipodi:role="line"
             id="tspan3649">1.00</tspan></text>
      </g>
      <g
         id="g3653" />
      <g
         id="g3655" />
      <g
         id="g3657" />
      <g
         id="g3659" />
      <g
         id="g3661" />
      <g
         id="g3663">
        <text
           transform="matrix(1,0,0,-1,254.47,47.1)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3667"><tspan
             x="0 7.9419999 14.058 19.558001 25.674 28.115999"
             y="0"
             sodipodi:role="line"
             id="tspan3665">Recall</tspan></text>
      </g>
      <g
         id="g3669" />
      <g
         id="g3671">
        <text
           transform="matrix(0,1,1,0,37.28,253.36)"
           style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3675"><tspan
             x="0 7.3369999 11 17.115999 22.615999 25.058001 30.558001 33 39.116001"
             y="0"
             sodipodi:role="line"
             id="tspan3673">Precision</tspan></text>
      </g>
      <g
         id="g3677" />
      <g
         id="g3679" />
      <g
         id="g3681" />
      <g
         id="g3683" />
      <g
         id="g3685" />
      <g
         id="g3687" />
      <g
         id="g3689" />
      <g
         id="g3695" />
      <g
         id="g3697" />
      <g
         id="g3699" />
      <g
         id="g3701">
        <path
           d="m 157.89,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3703" />
      </g>
      <g
         id="g3705" />
      <g
         id="g3707">
        <path
           d="m 159.62,19.6 h 13.83"
           style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3709" />
      </g>
      <g
         id="g3711" />
      <g
         id="g3713">
        <path
           d="m 264.66,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3715" />
      </g>
      <g
         id="g3717" />
      <g
         id="g3719">
        <path
           d="m 266.39,19.6 h 13.82"
           style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3721" />
      </g>
      <g
         id="g3723" />
      <g
         id="g3725">
        <path
           d="m 343.83,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3727" />
      </g>
      <g
         id="g3729" />
      <g
         id="g3731">
        <path
           d="m 345.56,19.6 h 13.82"
           style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
           id="path3733" />
      </g>
      <g
         id="g3735" />
      <g
         id="g3737" />
      <g
         id="g3739" />
      <g
         id="g3741">
        <text
           transform="matrix(1,0,0,-1,180.65,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3745"><tspan
             x="0 5.0040002 10.008 15.012 17.01 21.51 24.011999 26.01 30.51 33.012001 39.509998 44.514 49.518002 52.514999 57.519001 62.019001 66.518997 68.516998 73.521004"
             y="0"
             sodipodi:role="line"
             id="tspan3743">Logistic Regression</tspan></text>
      </g>
      <g
         id="g3747" />
      <g
         id="g3749" />
      <g
         id="g3751" />
      <g
         id="g3753" />
      <g
         id="g3755">
        <text
           transform="matrix(1,0,0,-1,287.42,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3759"><tspan
             x="0 4.4190001 7.4159999 12.42 17.424 19.926001 26.424 28.926001 31.427999 38.43 43.433998"
             y="0"
             sodipodi:role="line"
             id="tspan3757">Tree w/ GLM</tspan></text>
      </g>
      <g
         id="g3761" />
      <g
         id="g3763" />
      <g
         id="g3765" />
      <g
         id="g3767" />
      <g
         id="g3769">
        <text
           transform="matrix(1,0,0,-1,366.59,16.37)"
           style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3773"><tspan
             x="0 7.0019999 13.005"
             y="0"
             sodipodi:role="line"
             id="tspan3771">GAM</tspan></text>
      </g>
      <g
         id="g3775" />
      <g
         id="g3777" />
      <g
         id="g3779" />
      <g
         id="g3781" />
      <g
         id="g3783" />
      <g
         id="g3785" />
      <g
         id="g3787">
        <text
           transform="matrix(1,0,0,-1,64.88,489.19)"
           style="font-variant:normal;font-weight:normal;font-size:13px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="text3791"><tspan
             x="0 8.6709995 18.056999 21.671 31.056999 38.285 43.004002 49.179001 56.407001 60.021 67.612999 71.226997 77.610001 81.808998 89.037003 91.922997 99.151001 102.037 109.265 116.493 120.107 129.493 136.72099 140.33501"
             y="0"
             sodipodi:role="line"
             id="tspan3789">PR Curve − Training Data</tspan></text>
      </g>
      <g
         id="g3793" />
      <g
         id="g3795" />
      <g
         id="g3797" />
      <g
         id="g3799" />
      <g
         id="g3801" />
      <g
         id="g3803" />
    </g>
  </g>
</svg>
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="270mm"
   height="149.14511mm"
   viewBox="0 0 270 149.14511"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="testing_curves for post.svg">
  <defs
     id="defs2">
    <style
       type="text/css"
       id="style1139"><![CDATA[
    .svglite line, .svglite polyline, .svglite polygon, .svglite path, .svglite rect, .svglite circle {
      fill: none;
      stroke: #000000;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-miterlimit: 10.00;
    }
  ]]></style>
    <clipPath
       id="cpMC4wMHw1MDQuMDB8MC4wMHw1MDQuMDA=">
      <rect
         x="0"
         y="0"
         width="504"
         height="504"
         id="rect1145" />
    </clipPath>
    <clipPath
       id="cpMjMuOTZ8NDgwLjA0fDAuMDB8NTA0LjAw">
      <rect
         x="23.959999"
         y="0"
         width="456.09"
         height="504"
         id="rect1152" />
    </clipPath>
    <clipPath
       id="cpNjQuNDR8NDc0LjU3fDIzLjE4fDQzMy4zMQ==">
      <rect
         x="64.440002"
         y="23.18"
         width="410.13"
         height="410.13"
         id="rect1163" />
    </clipPath>
    <style
       type="text/css"
       id="style1671"><![CDATA[
    .svglite line, .svglite polyline, .svglite polygon, .svglite path, .svglite rect, .svglite circle {
      fill: none;
      stroke: #000000;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-miterlimit: 10.00;
    }
  ]]></style>
    <clipPath
       id="cpMC4wMHw1MDQuMDB8MC4wMHw1MDQuMDA=-8">
      <rect
         x="0"
         y="0"
         width="504"
         height="504"
         id="rect1677" />
    </clipPath>
    <clipPath
       id="cpMjMuOTZ8NDgwLjA0fDAuMDB8NTA0LjAw-8">
      <rect
         x="23.959999"
         y="0"
         width="456.09"
         height="504"
         id="rect1684" />
    </clipPath>
    <clipPath
       id="cpNjQuNDR8NDc0LjU3fDIzLjE4fDQzMy4zMQ==-1">
      <rect
         x="64.440002"
         y="23.18"
         width="410.13"
         height="410.13"
         id="rect1695" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2896">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path2894" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath2908">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path2906" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3471">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path3469" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath3483">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path3481" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4637">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path4635" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4649">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path4647" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath5212">
      <path
         d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
         id="path5210" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath5224">
      <path
         d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
         id="path5222" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4"
     inkscape:cx="216.31158"
     inkscape:cy="292.05725"
     inkscape:document-units="mm"
     inkscape:current-layer="g5168"
     inkscape:document-rotation="0"
     showgrid="false"
     inkscape:window-width="2880"
     inkscape:window-height="1526"
     inkscape:window-x="2869"
     inkscape:window-y="-11"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1">
    <g
       id="g2852"
       inkscape:label="training_ROC_AUC_plot"
       transform="matrix(0.29592284,0,0,-0.29592284,-7.0725558,149.14512)">
      <g
         id="g2854" />
      <g
         id="g2856" />
      <g
         id="g2858" />
      <g
         id="g2860" />
      <g
         id="g2862" />
      <g
         id="g2864" />
      <g
         id="g2866" />
      <g
         id="g2868" />
      <g
         id="g2870" />
      <g
         id="g2872" />
      <g
         id="g2874" />
      <g
         id="g2876" />
      <g
         id="g2878" />
      <g
         id="g2880" />
      <g
         id="g2882" />
      <g
         id="g2884" />
      <g
         id="g2886" />
      <g
         id="g2888" />
      <g
         id="g2900" />
      <g
         id="g2958" />
      <g
         id="g2960" />
      <g
         id="g2962" />
      <g
         id="g2964" />
      <g
         id="g2966" />
      <g
         id="g2968" />
      <g
         id="g2970" />
      <g
         id="g2972" />
      <g
         id="g2974" />
      <g
         id="g2976" />
      <g
         id="g2978" />
      <g
         id="g2980" />
      <g
         id="g2982" />
      <g
         id="g2984" />
      <g
         id="g2986" />
      <g
         id="g3010" />
      <g
         id="g3024" />
      <g
         id="g3026" />
      <g
         id="g3028" />
      <g
         id="g3030" />
      <g
         id="g3032" />
      <g
         id="g3034" />
      <g
         id="g3036" />
      <g
         id="g3038" />
      <g
         id="g3040" />
      <g
         id="g3054" />
      <g
         id="g3078" />
      <g
         id="g3080" />
      <g
         id="g3082" />
      <g
         id="g3084" />
      <g
         id="g3086" />
      <g
         id="g3094" />
      <g
         id="g3102" />
      <g
         id="g3104" />
      <g
         id="g3106" />
      <g
         id="g3108" />
      <g
         id="g3110" />
      <g
         id="g3112" />
      <g
         id="g3114" />
      <g
         id="g3120" />
      <g
         id="g3122" />
      <g
         id="g3124" />
      <g
         id="g3126">
        <path
           d="m 157.89,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3128" />
      </g>
      <g
         id="g3130" />
      <g
         id="g3136" />
      <g
         id="g3138">
        <path
           d="m 264.66,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3140" />
      </g>
      <g
         id="g3142" />
      <g
         id="g3148" />
      <g
         id="g3150">
        <path
           d="m 343.83,10.96 h 17.28 v 17.28 h -17.28 z"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           id="path3152" />
      </g>
      <g
         id="g3154" />
      <g
         id="g3160" />
      <g
         id="g3162" />
      <g
         id="g3164" />
      <g
         id="g3172" />
      <g
         id="g3174" />
      <g
         id="g3176" />
      <g
         id="g3178" />
      <g
         id="g3186" />
      <g
         id="g3188" />
      <g
         id="g3190" />
      <g
         id="g3192" />
      <g
         id="g3200" />
      <g
         id="g3202" />
      <g
         id="g3204" />
      <g
         id="g3206" />
      <g
         id="g3208" />
      <g
         id="g3210" />
      <g
         id="g3218" />
      <g
         id="g3220" />
      <g
         id="g3222" />
      <g
         id="g3224" />
      <g
         id="g3226" />
      <g
         id="g3228" />
      <g
         id="g4593"
         inkscape:label="testing_ROC_AUC_plot"
         transform="translate(-2.568237e-7,2.91968e-5)">
        <g
           id="g4595" />
        <g
           id="g4597" />
        <g
           id="g4599" />
        <g
           id="g4601" />
        <g
           id="g4603" />
        <g
           id="g4605" />
        <g
           id="g4607" />
        <g
           id="g4609" />
        <g
           id="g4611" />
        <g
           id="g4613" />
        <g
           id="g4615" />
        <g
           id="g4617" />
        <g
           id="g4619" />
        <g
           id="g4621" />
        <g
           id="g4623" />
        <g
           id="g4625" />
        <g
           id="g4627" />
        <g
           id="g4629" />
        <g
           id="g4631">
          <g
             id="g4633"
             clip-path="url(#clipPath4637)">
            <path
               d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
               style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4639" />
          </g>
        </g>
        <g
           id="g4641" />
        <g
           id="g4643">
          <g
             id="g4645"
             clip-path="url(#clipPath4649)">
            <path
               d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
               style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
               id="path4651" />
            <path
               d="M 64.88,136.29 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4653" />
            <path
               d="M 64.88,229.42 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4655" />
            <path
               d="M 64.88,322.54 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4657" />
            <path
               d="M 64.88,415.66 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4659" />
            <path
               d="M 130.07,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4661" />
            <path
               d="M 223.19,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4663" />
            <path
               d="M 316.31,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4665" />
            <path
               d="M 409.44,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4667" />
            <path
               d="M 64.88,89.73 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4669" />
            <path
               d="M 64.88,182.86 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4671" />
            <path
               d="M 64.88,275.98 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4673" />
            <path
               d="M 64.88,369.1 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4675" />
            <path
               d="M 64.88,462.22 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4677" />
            <path
               d="M 83.51,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4679" />
            <path
               d="M 176.63,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4681" />
            <path
               d="M 269.75,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4683" />
            <path
               d="M 362.87,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4685" />
            <path
               d="M 456,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4687" />
            <path
               d="m 83.51,89.73 v 0.34 0.34 0.34 0.34 h 0.37 l 0.37,2.04 0.37,7.81 0.38,5.09 0.37,3.4 0.37,3.39 0.37,1.7 0.38,2.72 0.37,12.56 0.37,1.7 h 0.37 l 0.38,10.18 0.37,1.7 0.37,6.45 0.37,1.02 0.38,0.68 h 0.37 l 0.37,1.7 0.37,2.04 0.38,0.34 h 0.37 l 0.37,0.68 0.37,0.67 0.38,5.78 0.37,1.69 0.37,0.68 0.37,1.36 0.38,1.7 0.37,0.34 0.37,1.02 h 0.37 0.38 0.37 l 0.37,2.71 0.37,0.68 h 0.38 l 0.37,0.68 0.37,3.06 h 0.37 0.38 0.37 0.37 0.37 l 0.38,1.7 0.37,0.34 0.37,0.34 0.37,8.82 h 0.38 0.37 l 0.37,1.7 0.37,0.34 h 0.38 0.37 0.37 l 0.37,2.04 0.38,1.36 0.37,3.39 0.37,1.36 h 0.37 l 0.38,0.68 0.37,1.36 h 0.37 l 0.37,0.34 0.38,6.11 h 0.37 0.37 l 0.37,1.36 h 0.38 0.37 l 0.37,0.34 0.37,1.36 0.38,0.33 h 0.37 l 0.37,2.04 0.37,0.68 0.38,1.36 0.37,0.34 0.37,0.68 0.37,2.71 0.38,1.02 h 0.37 0.37 0.37 l 0.38,0.34 0.37,0.34 0.37,0.34 h 0.37 l 0.38,1.7 h 0.37 0.37 l 0.37,0.34 0.38,3.39 h 0.37 l 0.37,1.02 h 0.37 l 0.38,1.36 0.37,0.68 0.37,0.34 0.37,2.04 h 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.68 0.37,2.03 0.37,2.04 h 0.38 l 0.37,0.68 h 0.37 l 0.37,1.02 h 0.38 l 0.37,1.02 h 0.37 0.37 l 0.38,2.03 0.37,1.7 0.37,0.68 h 0.37 0.38 0.37 l 0.37,2.72 0.37,0.34 h 0.38 l 0.37,2.71 0.37,1.36 h 0.37 l 0.38,1.02 0.37,1.02 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.68 h 0.37 l 0.38,0.68 h 0.37 l 0.37,0.34 0.37,0.34 h 0.38 l 0.37,0.67 0.37,0.68 h 0.37 0.38 0.37 l 0.37,0.34 0.37,1.02 0.38,0.34 0.37,0.34 h 0.37 l 0.37,0.68 h 0.38 l 0.37,0.34 0.37,0.34 h 0.37 l 0.38,2.38 0.37,1.35 0.37,0.34 h 0.37 l 0.38,0.68 h 0.37 l 0.37,1.7 h 0.37 0.38 0.37 l 0.37,1.7 0.37,0.68 0.38,1.02 0.37,0.68 0.37,0.67 0.37,0.34 0.38,2.04 0.37,0.68 h 0.37 l 0.37,1.7 h 0.37 l 0.38,0.34 0.37,0.68 0.37,0.68 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 l 0.37,0.33 0.38,0.68 h 0.37 0.37 l 0.37,2.38 h 0.38 l 0.37,0.68 0.37,0.68 h 0.37 0.38 l 0.37,0.34 0.37,1.02 h 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 0.37,1.01 0.37,1.02 0.38,0.34 0.37,0.68 h 0.37 l 0.37,0.34 0.38,0.68 0.37,0.34 h 0.37 0.37 l 0.38,0.68 h 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.68 0.38,3.05 h 0.37 0.37 l 0.37,0.68 0.38,3.4 0.37,2.04 h 0.37 l 0.37,0.34 0.38,0.34 0.37,1.01 0.37,2.38 0.37,1.02 h 0.38 l 0.37,0.68 0.37,0.68 h 0.37 0.38 0.37 0.37 l 0.37,1.7 0.38,1.01 0.37,0.68 0.37,0.68 0.37,0.34 0.38,0.34 0.37,0.68 0.37,2.04 h 0.37 0.38 l 0.37,1.02 0.37,0.34 0.37,0.68 h 0.38 l 0.37,0.33 0.37,1.36 h 0.37 l 0.38,0.68 h 0.37 l 0.37,1.7 0.37,0.34 0.38,0.34 h 0.37 0.37 l 0.37,2.72 0.38,0.33 0.37,0.34 0.37,0.34 0.37,0.68 h 0.38 l 0.37,0.68 h 0.37 l 0.37,0.34 h 0.38 l 0.37,1.7 h 0.37 l 0.37,3.06 0.38,0.67 0.37,0.68 h 0.37 l 0.37,0.34 0.38,1.02 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,0.68 0.37,0.68 h 0.38 0.37 l 0.37,0.68 0.37,0.34 0.38,0.34 h 0.37 l 0.37,0.68 0.37,0.68 0.38,0.34 h 0.37 0.37 l 0.37,0.34 0.38,1.35 H 196 l 0.37,0.68 0.37,0.68 h 0.38 l 0.37,0.34 0.37,2.04 0.37,1.02 0.38,0.34 h 0.37 l 0.37,0.68 h 0.37 l 0.38,0.34 0.37,0.34 0.37,3.39 0.37,1.36 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.68 0.37,1.36 h 0.37 0.37 l 0.38,0.67 0.37,0.34 h 0.37 l 0.37,1.7 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.68 0.37,1.7 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 l 0.37,0.68 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.67 h 0.38 0.37 0.37 l 0.37,1.02 h 0.38 l 0.37,0.34 0.37,0.34 0.37,0.68 0.38,0.34 h 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 0.37 l 0.38,1.36 0.37,0.34 h 0.37 0.37 l 0.38,0.34 0.37,0.34 h 0.37 l 0.37,0.34 0.38,0.34 h 0.37 l 0.37,0.34 0.37,0.34 h 0.38 l 0.37,0.33 0.37,1.02 0.37,0.34 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 l 0.38,2.04 h 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.68 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,1.7 0.37,0.34 h 0.37 l 0.37,0.67 h 0.38 0.37 l 0.37,0.34 h 0.37 0.37 l 0.38,1.36 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,0.34 0.37,0.34 0.38,0.68 0.37,0.68 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.67 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 0.37,0.68 0.37,0.34 0.38,0.68 h 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.68 h 0.37 0.38 l 0.37,0.34 0.37,1.02 0.37,0.68 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,0.33 0.38,0.68 h 0.37 l 0.37,0.68 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,1.02 0.37,0.34 0.38,0.68 0.37,0.34 h 0.37 0.37 0.38 l 0.37,1.02 0.37,0.68 0.37,0.34 h 0.38 l 0.37,0.68 h 0.37 0.37 0.38 l 0.37,0.33 0.37,1.36 h 0.37 l 0.38,0.68 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,1.36 0.37,1.02 0.38,1.36 h 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.33 h 0.37 l 0.37,0.34 0.38,1.7 0.37,0.34 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 l 0.37,1.02 h 0.37 0.38 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.68 0.37,0.34 0.37,0.68 h 0.38 l 0.37,0.67 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.68 h 0.38 l 0.37,0.68 h 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 l 0.37,0.68 0.37,0.68 h 0.37 0.38 l 0.37,0.68 0.37,0.68 0.37,0.34 h 0.38 0.37 l 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 0.37 l 0.37,1.01 0.38,0.68 h 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,1.36 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.68 0.37,0.34 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.68 0.37,0.68 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.37,0.34 0.38,0.33 h 0.37 l 0.37,0.34 0.37,1.02 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 0.37,0.34 h 0.38 0.37 l 0.37,0.67 0.37,1.02 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 0.37,1.02 0.37,0.34 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.68 0.38,0.68 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.33 h 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 0.37,0.34 h 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 0.37,0.34 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,1.36 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 0.38,0.34 h 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,1.01 h 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.68 0.37,0.34 0.37,0.34 0.37,0.34 h 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 0.38,0.34 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.33 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 H 456"
               style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4689" />
            <path
               d="m 83.51,89.73 v 0.34 0.34 0.34 0.34 h 0.37 l 0.37,2.04 0.37,9.17 0.38,1.69 0.37,4.76 0.37,4.41 0.37,8.83 0.38,1.02 0.37,1.7 0.37,2.71 h 0.37 l 0.38,7.47 0.37,8.83 0.37,0.68 0.37,0.34 0.38,1.7 0.37,1.36 0.37,4.41 0.37,5.09 h 0.38 0.37 l 0.37,0.34 0.37,2.04 h 0.38 l 0.37,2.04 h 0.37 l 0.37,1.7 h 0.38 l 0.37,2.71 h 0.37 0.37 0.38 l 0.37,3.4 0.37,1.36 0.37,0.68 h 0.38 l 0.37,0.67 0.37,0.68 0.37,5.1 0.38,0.68 0.37,0.67 h 0.37 0.37 l 0.38,0.68 0.37,0.34 0.37,0.34 h 0.37 l 0.38,0.68 0.37,1.7 h 0.37 l 0.37,2.04 h 0.38 l 0.37,0.34 h 0.37 l 0.37,2.03 0.38,1.7 0.37,2.04 0.37,0.68 0.37,1.7 0.38,6.45 h 0.37 0.37 l 0.37,5.77 h 0.38 l 0.37,4.41 0.37,1.7 0.37,1.02 0.38,1.02 h 0.37 l 0.37,0.68 0.37,0.68 0.38,0.68 h 0.37 l 0.37,4.07 h 0.37 l 0.38,3.4 0.37,0.68 h 0.37 l 0.37,0.68 h 0.38 l 0.37,0.67 h 0.37 0.37 0.38 l 0.37,2.04 h 0.37 l 0.37,1.02 h 0.38 l 0.37,0.68 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 0.37,0.34 h 0.38 l 0.37,1.02 0.37,0.68 0.37,0.33 0.38,0.68 0.37,0.34 0.37,1.02 0.37,1.02 h 0.38 l 0.37,0.34 0.37,0.68 0.37,0.68 0.38,1.36 0.37,1.02 h 0.37 0.37 0.38 l 0.37,0.67 h 0.37 l 0.37,1.36 h 0.38 l 0.37,0.68 0.37,2.04 0.37,0.34 h 0.38 l 0.37,0.68 h 0.37 l 0.37,1.36 h 0.38 l 0.37,0.68 h 0.37 l 0.37,0.67 h 0.38 0.37 l 0.37,0.68 0.37,0.68 0.38,0.68 0.37,1.7 0.37,0.34 h 0.37 l 0.38,0.34 0.37,0.34 0.37,2.04 0.37,1.69 h 0.38 l 0.37,1.7 0.37,1.02 h 0.37 0.38 0.37 0.37 l 0.37,1.02 0.38,0.68 0.37,0.34 h 0.37 l 0.37,0.34 0.38,1.69 h 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 l 0.37,1.7 0.38,0.34 h 0.37 0.37 l 0.37,0.34 0.38,0.68 h 0.37 l 0.37,0.34 0.37,0.34 0.38,0.34 0.37,0.34 0.37,0.68 h 0.37 l 0.38,1.35 h 0.37 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,1.02 h 0.37 l 0.37,3.06 0.38,0.68 0.37,0.34 0.37,0.34 h 0.37 l 0.38,1.35 h 0.37 l 0.37,1.7 0.37,0.68 0.38,0.34 h 0.37 l 0.37,1.36 0.37,1.36 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,1.02 0.37,0.67 0.38,0.34 0.37,3.4 h 0.37 0.37 l 0.38,0.68 0.37,0.68 h 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 l 0.37,3.05 0.37,0.34 0.37,0.34 0.38,0.34 0.37,1.36 h 0.37 l 0.37,0.68 0.38,0.68 0.37,0.34 h 0.37 l 0.37,2.03 0.38,1.02 0.37,0.68 h 0.37 0.37 l 0.38,2.04 0.37,0.34 0.37,0.34 0.37,2.04 h 0.38 0.37 l 0.37,0.67 0.37,0.68 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.68 0.37,1.02 h 0.37 l 0.38,0.68 0.37,0.68 H 177 l 0.37,1.02 0.38,0.34 h 0.37 0.37 l 0.37,1.02 h 0.38 0.37 l 0.37,1.35 0.37,2.04 0.38,1.02 0.37,0.34 0.37,1.7 h 0.37 l 0.38,1.02 0.37,1.69 0.37,2.04 0.37,1.02 h 0.38 l 0.37,1.36 0.37,0.68 0.37,1.02 0.38,1.01 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,2.04 0.38,0.68 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.68 0.37,0.34 0.37,0.68 h 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.68 0.38,0.33 0.37,2.38 h 0.37 0.37 l 0.38,0.34 H 196 l 0.37,1.36 0.37,0.34 h 0.38 0.37 l 0.37,3.06 h 0.37 0.38 0.37 l 0.37,0.67 h 0.37 0.38 l 0.37,1.7 0.37,0.34 h 0.37 l 0.38,0.68 0.37,0.34 0.37,0.34 h 0.37 0.38 l 0.37,0.68 0.37,0.68 0.37,0.68 0.38,0.68 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.67 h 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.68 h 0.38 0.37 0.37 l 0.37,1.36 h 0.38 l 0.37,0.34 0.37,1.02 h 0.37 l 0.38,0.68 h 0.37 0.37 l 0.37,1.02 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 l 0.37,0.67 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.34 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 0.37 l 0.37,0.34 0.38,0.34 0.37,0.34 h 0.37 l 0.37,0.34 0.38,0.34 h 0.37 l 0.37,1.36 0.37,1.7 h 0.38 l 0.37,0.68 0.37,1.69 0.37,0.34 h 0.38 l 0.37,0.68 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 0.37,0.34 h 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,1.02 h 0.37 l 0.38,0.68 0.37,0.34 0.37,1.36 h 0.37 0.38 l 0.37,0.67 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,1.02 h 0.37 l 0.37,0.34 h 0.37 l 0.38,1.02 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.34 0.37,0.34 0.37,0.34 0.37,0.68 h 0.38 0.37 0.37 l 0.37,0.34 0.38,0.34 h 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.33 h 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.34 0.37,0.68 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.68 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,1.36 0.37,0.34 0.37,0.34 0.38,2.37 0.37,0.68 h 0.37 0.37 0.38 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.68 0.37,0.34 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,3.39 0.37,1.7 0.37,1.02 h 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,1.02 h 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,1.02 0.37,1.35 0.38,0.34 0.37,0.34 h 0.37 0.37 l 0.38,1.02 h 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.68 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.68 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.68 0.37,0.68 0.37,1.01 0.38,1.7 h 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.68 h 0.37 0.38 0.37 l 0.37,1.36 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.68 h 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 l 0.37,0.34 0.38,0.33 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,1.36 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,1.02 h 0.37 0.38 l 0.37,1.02 0.37,0.34 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.37 l 0.38,1.36 h 0.37 l 0.37,0.33 0.37,1.02 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.68 0.38,0.34 h 0.37 0.37 0.37 l 0.38,1.02 h 0.37 0.37 0.37 l 0.38,0.68 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.33 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.68 h 0.37 l 0.38,0.68 h 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,1.02 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 0.37,0.34 0.38,0.68 0.37,0.68 h 0.37 0.37 0.38 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.33 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 l 0.37,1.02 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.68 h 0.38 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.68 h 0.37 l 0.37,0.68 h 0.38 l 0.37,0.34 0.37,0.68 0.37,0.68 h 0.38 0.37 0.37 l 0.37,1.01 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.68 h 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 0.37,0.33 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38"
               style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4691" />
            <path
               d="m 83.51,89.73 v 0.34 0.34 0.34 0.34 0.34 h 0.37 l 0.37,2.04 0.37,10.18 0.38,1.7 0.37,2.72 0.37,4.07 0.37,6.45 0.38,2.38 0.37,12.57 0.37,3.39 h 0.37 l 0.38,5.77 0.37,4.08 0.37,1.02 0.37,4.07 0.38,3.06 0.37,1.02 0.37,2.37 0.37,0.68 0.38,2.38 h 0.37 l 0.37,1.36 0.37,2.03 0.38,4.42 h 0.37 l 0.37,2.04 0.37,0.34 0.38,0.67 0.37,1.02 0.37,2.04 h 0.37 l 0.38,1.02 0.37,1.7 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 0.37,1.35 h 0.38 l 0.37,0.34 h 0.37 l 0.37,5.44 0.38,1.69 h 0.37 l 0.37,1.02 0.37,0.34 h 0.38 l 0.37,0.34 0.37,2.72 0.37,0.68 h 0.38 l 0.37,3.73 h 0.37 0.37 l 0.38,1.36 h 0.37 0.37 l 0.37,3.4 0.38,0.34 h 0.37 0.37 l 0.37,1.01 0.38,0.68 0.37,0.34 0.37,4.42 0.37,0.34 h 0.38 l 0.37,1.02 0.37,0.67 0.37,0.34 h 0.38 0.37 l 0.37,1.36 0.37,2.38 0.38,3.73 0.37,0.34 0.37,1.7 0.37,2.38 h 0.38 l 0.37,1.36 0.37,3.05 h 0.37 l 0.38,0.34 0.37,2.04 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 0.37,0.34 0.38,0.34 h 0.37 0.37 0.37 l 0.38,1.36 h 0.37 l 0.37,3.05 h 0.37 l 0.38,0.68 0.37,0.68 0.37,1.7 h 0.37 0.38 0.37 0.37 l 0.37,0.68 h 0.38 l 0.37,3.05 0.37,0.34 0.37,0.34 h 0.38 l 0.37,5.1 h 0.37 l 0.37,0.34 0.38,0.67 h 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 l 0.37,0.34 0.37,0.68 h 0.38 0.37 l 0.37,0.68 h 0.37 l 0.38,0.34 0.37,1.36 0.37,0.68 0.37,2.04 0.38,1.35 0.37,1.02 h 0.37 0.37 l 0.38,2.72 0.37,1.02 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.68 0.37,1.69 0.37,2.04 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,3.74 h 0.38 l 0.37,1.35 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 0.37 0.37 l 0.38,2.04 0.37,0.68 0.37,1.02 0.37,0.34 h 0.37 l 0.38,0.68 0.37,1.35 0.37,1.36 0.37,0.68 0.38,0.34 0.37,1.02 0.37,3.06 h 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,1.01 h 0.38 0.37 0.37 l 0.37,0.68 0.38,0.34 0.37,0.34 0.37,0.68 h 0.37 0.38 l 0.37,1.7 0.37,1.36 0.37,0.34 0.38,0.34 0.37,3.05 0.37,1.36 0.37,0.34 0.38,0.68 0.37,0.34 h 0.37 l 0.37,1.02 0.38,0.34 0.37,0.68 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,1.35 h 0.38 l 0.37,1.02 0.37,2.04 h 0.37 l 0.38,0.68 0.37,1.02 h 0.37 0.37 0.38 0.37 0.37 l 0.37,0.68 0.38,0.34 0.37,0.68 0.37,0.34 0.37,2.37 0.38,0.68 0.37,1.02 0.37,0.34 0.37,1.36 0.38,2.37 h 0.37 l 0.37,1.02 0.37,1.02 h 0.38 0.37 0.37 l 0.37,1.02 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,1.7 0.38,0.68 0.37,0.68 0.37,0.33 h 0.37 l 0.38,1.7 h 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 l 0.37,1.36 0.37,1.36 0.37,0.68 h 0.38 l 0.37,0.34 0.37,1.02 h 0.37 0.38 0.37 l 0.37,0.34 0.37,0.67 h 0.38 l 0.37,1.02 h 0.37 l 0.37,2.04 0.38,0.68 h 0.37 0.37 0.37 l 0.38,0.68 0.37,1.36 0.37,0.34 0.37,1.69 h 0.38 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 0.37,1.02 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.68 0.37,2.71 h 0.38 l 0.37,0.68 h 0.37 l 0.37,0.68 0.38,1.02 h 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 0.37,1.36 0.37,0.34 0.37,0.68 h 0.38 0.37 0.37 l 0.37,1.35 0.38,1.02 0.37,0.34 h 0.37 0.37 l 0.38,1.02 h 0.37 l 0.37,2.04 0.37,0.68 h 0.38 l 0.37,0.68 0.37,1.01 0.37,0.34 h 0.38 l 0.37,0.68 h 0.37 0.37 l 0.38,0.68 h 0.37 0.37 0.37 l 0.38,0.34 0.37,0.68 0.37,0.34 0.37,1.02 h 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.68 0.37,0.34 0.37,1.02 h 0.38 0.37 0.37 0.37 l 0.38,0.67 h 0.37 0.37 0.37 l 0.38,0.34 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,1.02 0.37,0.68 h 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 l 0.38,0.34 0.37,0.34 h 0.37 l 0.37,0.68 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 0.37,0.67 h 0.37 0.38 0.37 0.37 l 0.37,1.36 h 0.38 0.37 l 0.37,1.7 0.37,0.34 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,2.38 h 0.37 0.38 l 0.37,1.01 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.68 0.37,0.34 0.38,1.36 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.34 0.37,0.34 0.37,0.67 0.38,0.68 0.37,1.02 h 0.37 0.37 0.38 0.37 l 0.37,0.68 0.37,1.02 h 0.38 0.37 l 0.37,0.68 0.37,0.34 0.38,0.68 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,1.36 h 0.38 0.37 0.37 0.37 0.38 l 0.37,0.33 h 0.37 0.37 0.38 l 0.37,1.02 0.37,1.02 h 0.37 l 0.38,1.02 h 0.37 l 0.37,0.34 0.37,1.02 h 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.68 h 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,1.02 h 0.37 l 0.38,0.34 0.37,0.33 0.37,0.68 h 0.37 l 0.38,0.34 0.37,0.68 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.68 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 l 0.37,0.68 0.37,0.34 h 0.38 0.37 0.37 l 0.37,1.7 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.33 0.37,1.36 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.68 0.37,0.34 h 0.37 0.37 l 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 0.37 l 0.37,0.68 h 0.38 0.37 0.37 l 0.37,0.34 0.38,0.34 0.37,1.01 h 0.37 0.37 l 0.38,0.34 0.37,0.34 0.37,0.34 h 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,1.02 h 0.37 l 0.37,0.34 h 0.37 l 0.38,0.68 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 l 0.37,0.68 h 0.37 l 0.38,0.68 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.68 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.33 h 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.37 0.38 0.37 l 0.37,0.34 0.37,1.36 h 0.38 0.37 0.37 0.37 l 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.34 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 l 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 0.37,0.68 0.37,0.34 h 0.37 0.38 l 0.37,0.34 0.37,0.67 0.37,0.34 h 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.68 h 0.37 0.37 l 0.38,1.7 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.68 h 0.37 l 0.37,0.34 0.38,1.02 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.67 0.37,0.34 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 l 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 h 0.37 l 0.37,1.02 0.38,0.34 0.37,1.36 h 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 l 0.37,0.68 h 0.37 0.37 l 0.38,0.34 0.37,0.34 h 0.37 0.37 0.38 l 0.37,0.34 0.37,0.33 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,0.68 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 l 0.37,1.02 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.68 h 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.37 l 0.38,0.34 h 0.37 l 0.37,0.34 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 l 0.38,0.34 0.37,0.68 0.37,0.68 0.37,0.34 h 0.38 0.37 0.37 0.37 l 0.38,0.34 h 0.37 0.37 0.37 0.38 0.37 l 0.37,0.33 h 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 l 0.37,0.34 h 0.38 l 0.37,0.34 h 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 0.37 0.38 0.37 0.37 l 0.37,0.34 H 456"
               style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4693" />
            <path
               d="M 64.88,71.11 474.62,480.85"
               style="fill:none;stroke:#bebebe;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:1.07, 3.2;stroke-dashoffset:0;stroke-opacity:1"
               id="path4695" />
            <path
               d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
               style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path4697" />
          </g>
        </g>
        <g
           id="g4699" />
        <g
           id="g4701" />
        <g
           id="g4703" />
        <g
           id="g4705" />
        <g
           id="g4707" />
        <g
           id="g4709" />
        <g
           id="g4711" />
        <g
           id="g4713" />
        <g
           id="g4715" />
        <g
           id="g4717" />
        <g
           id="g4719" />
        <g
           id="g4721" />
        <g
           id="g4723" />
        <g
           id="g4725" />
        <g
           id="g4727" />
        <g
           id="g4729">
          <text
             transform="matrix(1,0,0,-1,42.44,86.5)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4733"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4731">0.00</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,179.62)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4737"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4735">0.25</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,272.75)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4741"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4739">0.50</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,365.87)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4745"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4743">0.75</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,458.99)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4749"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4747">1.00</tspan></text>
        </g>
        <g
           id="g4751" />
        <g
           id="g4753">
          <path
             d="m 62.14,89.73 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4755" />
          <path
             d="m 62.14,182.86 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4757" />
          <path
             d="m 62.14,275.98 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4759" />
          <path
             d="m 62.14,369.1 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4761" />
          <path
             d="m 62.14,462.22 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4763" />
        </g>
        <g
           id="g4765" />
        <g
           id="g4767" />
        <g
           id="g4769" />
        <g
           id="g4771" />
        <g
           id="g4773" />
        <g
           id="g4775" />
        <g
           id="g4777" />
        <g
           id="g4779" />
        <g
           id="g4781" />
        <g
           id="g4783">
          <path
             d="m 83.51,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4785" />
          <path
             d="m 176.63,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4787" />
          <path
             d="m 269.75,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4789" />
          <path
             d="m 362.87,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4791" />
          <path
             d="m 456,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4793" />
        </g>
        <g
           id="g4795" />
        <g
           id="g4797">
          <text
             transform="matrix(1,0,0,-1,74.75,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4801"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4799">0.00</tspan></text>
          <text
             transform="matrix(1,0,0,-1,167.87,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4805"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4803">0.25</tspan></text>
          <text
             transform="matrix(1,0,0,-1,260.99,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4809"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4807">0.50</tspan></text>
          <text
             transform="matrix(1,0,0,-1,354.12,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4813"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4811">0.75</tspan></text>
          <text
             transform="matrix(1,0,0,-1,447.24,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4817"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan4815">1.00</tspan></text>
        </g>
        <g
           id="g4819" />
        <g
           id="g4821" />
        <g
           id="g4823" />
        <g
           id="g4825" />
        <g
           id="g4827" />
        <g
           id="g4829">
          <text
             transform="matrix(1,0,0,-1,235.67,47.1)"
             style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4833"><tspan
               x="0 6.1160002 9.1739998 15.598 18.656 25.993 32.109001 38.224998 43.724998 46.167 49.224998 51.667 57.167 59.609001 62.667"
               y="0"
               sodipodi:role="line"
               id="tspan4831">1 − Specificity</tspan></text>
        </g>
        <g
           id="g4835" />
        <g
           id="g4837">
          <text
             transform="matrix(0,1,1,0,37.28,251.22)"
             style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4841"><tspan
               x="0 7.3369999 13.453 19.569 25.069 27.511 30.569 33.011002 38.511002 40.952999 44.011002"
               y="0"
               sodipodi:role="line"
               id="tspan4839">Sensitivity</tspan></text>
        </g>
        <g
           id="g4843" />
        <g
           id="g4845" />
        <g
           id="g4847" />
        <g
           id="g4849" />
        <g
           id="g4851" />
        <g
           id="g4853" />
        <g
           id="g4855" />
        <g
           id="g4857">
          <path
             d="M 146.93,5.48 H 392.56 V 33.72 H 146.93 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path4859" />
        </g>
        <g
           id="g4861" />
        <g
           id="g4863" />
        <g
           id="g4865" />
        <g
           id="g4867">
          <path
             d="m 157.89,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path4869" />
        </g>
        <g
           id="g4871" />
        <g
           id="g4873">
          <path
             d="m 159.62,19.6 h 13.83"
             style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4875" />
        </g>
        <g
           id="g4877" />
        <g
           id="g4879">
          <path
             d="m 264.66,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path4881" />
        </g>
        <g
           id="g4883" />
        <g
           id="g4885">
          <path
             d="m 266.39,19.6 h 13.82"
             style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4887" />
        </g>
        <g
           id="g4889" />
        <g
           id="g4891">
          <path
             d="m 343.83,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path4893" />
        </g>
        <g
           id="g4895" />
        <g
           id="g4897">
          <path
             d="m 345.56,19.6 h 13.82"
             style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path4899" />
        </g>
        <g
           id="g4901" />
        <g
           id="g4903" />
        <g
           id="g4905" />
        <g
           id="g4907">
          <text
             transform="matrix(1,0,0,-1,180.65,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4911"><tspan
               x="0 5.0040002 10.008 15.012 17.01 21.51 24.011999 26.01 30.51 33.012001 39.509998 44.514 49.518002 52.514999 57.519001 62.019001 66.518997 68.516998 73.521004"
               y="0"
               sodipodi:role="line"
               id="tspan4909">Logistic Regression</tspan></text>
        </g>
        <g
           id="g4913" />
        <g
           id="g4915" />
        <g
           id="g4917" />
        <g
           id="g4919" />
        <g
           id="g4921">
          <text
             transform="matrix(1,0,0,-1,287.42,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4925"><tspan
               x="0 4.4190001 7.4159999 12.42 17.424 19.926001 26.424 28.926001 31.427999 38.43 43.433998"
               y="0"
               sodipodi:role="line"
               id="tspan4923">Tree w/ GLM</tspan></text>
        </g>
        <g
           id="g4927" />
        <g
           id="g4929" />
        <g
           id="g4931" />
        <g
           id="g4933" />
        <g
           id="g4935">
          <text
             transform="matrix(1,0,0,-1,366.59,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4939"><tspan
               x="0 7.0019999 13.005"
               y="0"
               sodipodi:role="line"
               id="tspan4937">GAM</tspan></text>
        </g>
        <g
           id="g4941" />
        <g
           id="g4943" />
        <g
           id="g4945" />
        <g
           id="g4947" />
        <g
           id="g4949" />
        <g
           id="g4951" />
        <g
           id="g4953"
           transform="translate(0,5.3392969)">
          <text
             transform="matrix(1,0,0,-1,64.88,489.19)"
             style="font-variant:normal;font-weight:normal;font-size:13px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text4957"><tspan
               x="0 9.1260004 19.24 28.625999 32.240002 41.625999 48.854 53.573002 59.748001 66.975998 70.589996 78.181999 81.795998 88.179001 95.406998 101.907 105.521 109.135 118.521 125.749 129.36301"
               y="0"
               sodipodi:role="line"
               id="tspan4955">ROC Curve − Test Data</tspan></text>
        </g>
        <g
           id="g4959" />
        <g
           id="g4961" />
        <g
           id="g4963" />
        <g
           id="g4965" />
        <g
           id="g4967" />
        <g
           id="g4969" />
      </g>
      <g
         id="g5168"
         inkscape:label="testing_PR_AUC_plot"
         transform="translate(456.2,2.91968e-5)">
        <g
           id="g5170" />
        <g
           id="g5172" />
        <g
           id="g5174" />
        <g
           id="g5176" />
        <g
           id="g5178" />
        <g
           id="g5180" />
        <g
           id="g5182" />
        <g
           id="g5184" />
        <g
           id="g5186" />
        <g
           id="g5188" />
        <g
           id="g5190" />
        <g
           id="g5192" />
        <g
           id="g5194" />
        <g
           id="g5196" />
        <g
           id="g5198" />
        <g
           id="g5200" />
        <g
           id="g5202" />
        <g
           id="g5204" />
        <g
           id="g5206">
          <g
             id="g5208"
             clip-path="url(#clipPath5212)">
            <path
               d="M 23.9,0 H 480.1 V 504 H 23.9 Z"
               style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5214" />
          </g>
        </g>
        <g
           id="g5216" />
        <g
           id="g5218">
          <g
             id="g5220"
             clip-path="url(#clipPath5224)">
            <path
               d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
               style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
               id="path5226" />
            <path
               d="M 64.88,136.29 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5228" />
            <path
               d="M 64.88,229.42 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5230" />
            <path
               d="M 64.88,322.54 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5232" />
            <path
               d="M 64.88,415.66 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5234" />
            <path
               d="M 130.07,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5236" />
            <path
               d="M 223.19,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5238" />
            <path
               d="M 316.31,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5240" />
            <path
               d="M 409.44,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:0.53;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5242" />
            <path
               d="M 64.88,89.73 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5244" />
            <path
               d="M 64.88,182.86 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5246" />
            <path
               d="M 64.88,275.98 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5248" />
            <path
               d="M 64.88,369.1 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5250" />
            <path
               d="M 64.88,462.22 H 474.62"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5252" />
            <path
               d="M 83.51,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5254" />
            <path
               d="M 176.63,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5256" />
            <path
               d="M 269.75,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5258" />
            <path
               d="M 362.87,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5260" />
            <path
               d="M 456,71.11 V 480.85"
               style="fill:none;stroke:#ebebeb;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5262" />
            <path
               d="m 83.51,462.22 h 0.37 0.37 0.37 l 0.38,-69.13 0.37,11.69 0.37,8.31 0.37,6.21 0.38,4.82 0.37,3.84 0.37,-23.18 0.37,4.48 0.38,3.83 0.37,3.32 0.37,2.89 0.37,2.56 0.38,2.26 0.37,2.02 0.37,1.82 0.37,1.65 0.38,1.49 0.37,1.37 0.37,1.25 0.37,1.14 0.38,1.06 0.37,0.98 0.37,0.91 0.37,0.85 0.38,0.79 0.37,0.74 0.37,0.69 0.37,-8.85 0.38,0.87 0.37,0.82 0.37,0.77 0.37,0.74 0.38,0.69 0.37,0.67 0.37,0.63 0.37,0.59 0.38,0.58 0.37,0.54 0.37,0.52 0.37,0.5 0.38,-6.23 0.37,0.58 0.37,0.56 0.37,0.55 0.38,0.51 0.37,0.5 0.37,0.48 0.37,0.47 0.38,0.44 0.37,-5.08 0.37,0.5 0.37,0.49 0.38,0.47 0.37,0.45 0.37,0.44 0.37,0.43 0.38,0.41 0.37,0.41 0.37,-4.3 0.37,0.44 0.38,0.43 0.37,0.41 0.37,0.41 0.37,-3.89 0.38,0.44 0.37,0.42 0.37,0.41 0.37,0.41 0.38,0.39 0.37,0.38 0.37,-3.47 0.37,0.4 0.38,0.4 0.37,0.39 0.37,0.37 0.37,0.37 0.38,0.36 0.37,0.36 0.37,0.34 0.37,0.34 0.38,0.33 0.37,0.32 0.37,0.32 0.37,0.3 0.38,0.31 0.37,0.29 0.37,0.29 0.37,0.29 0.38,0.27 0.37,0.28 0.37,0.27 0.37,0.26 0.38,0.25 0.37,0.26 0.37,0.24 0.37,0.25 0.38,0.23 0.37,0.24 0.37,0.23 0.37,0.22 0.38,0.22 0.37,0.22 0.37,0.22 0.37,0.21 0.38,-2.54 0.37,0.23 0.37,0.22 0.37,0.22 0.38,0.21 0.37,-2.38 0.37,0.22 0.37,0.22 0.38,0.22 0.37,0.22 0.37,0.21 0.37,0.21 0.38,0.2 0.37,0.21 0.37,0.2 0.37,0.19 0.38,0.19 0.37,0.19 0.37,0.19 0.37,0.19 0.38,0.18 0.37,0.18 0.37,0.17 0.37,0.18 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.16 0.37,0.16 0.38,-1.97 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.16 0.37,-1.88 0.37,0.17 0.37,0.16 0.38,0.17 0.37,0.16 0.37,0.15 0.37,0.16 0.38,0.16 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.14 0.37,0.15 0.37,0.14 0.37,0.15 0.38,0.14 0.37,0.13 0.37,-1.69 0.37,0.15 0.38,0.14 0.37,-1.64 0.37,-3.34 0.37,0.16 0.38,0.17 0.37,0.16 0.37,0.16 0.37,-1.52 0.38,0.17 0.37,0.16 0.37,0.16 0.37,0.16 0.37,-1.46 0.38,-1.44 0.37,0.17 0.37,-1.4 0.37,0.17 0.38,-1.36 0.37,0.17 0.37,0.18 0.37,0.17 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.17 0.38,0.16 0.37,0.17 0.37,0.16 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.15 0.37,-1.27 0.38,0.16 0.37,0.15 0.37,0.16 0.37,0.16 0.38,-1.24 0.37,0.16 0.37,-1.22 0.37,0.16 0.38,0.16 0.37,0.16 0.37,-1.18 0.37,0.16 0.38,0.16 0.37,0.16 0.37,-1.15 0.37,-1.13 0.38,0.16 0.37,0.16 0.37,-3.61 0.37,0.17 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.17 0.38,0.17 0.37,-1.05 0.37,0.17 0.37,-2.21 0.38,0.17 0.37,-0.99 0.37,0.17 0.37,0.18 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,0.17 0.37,-5.4 0.37,0.19 0.37,0.18 0.38,0.18 0.37,-0.9 0.37,-0.89 0.37,-0.87 0.38,0.18 0.37,0.18 0.37,0.18 0.37,0.18 0.38,0.18 0.37,0.18 0.37,0.17 0.37,0.18 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.17 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,0.17 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.16 0.37,-2.76 0.38,0.16 0.37,0.16 0.37,0.17 0.37,0.16 0.38,-3.6 0.37,0.17 0.37,0.16 0.37,0.17 0.38,0.17 0.37,0.16 0.37,-0.75 0.37,0.17 0.38,0.16 0.37,0.16 0.37,-0.73 0.37,0.16 0.38,0.16 0.37,0.17 0.37,0.16 0.37,0.15 0.38,0.16 0.37,0.16 0.37,0.16 0.37,-0.72 0.38,0.16 0.37,0.15 0.37,-1.56 0.37,0.16 0.38,-0.69 0.37,0.16 0.37,0.15 0.37,0.16 0.38,-0.68 0.37,-0.67 0.37,0.16 0.37,0.15 0.38,0.16 0.37,0.15 0.37,0.16 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.14 0.38,0.15 0.37,-2.22 0.37,0.14 0.37,0.15 0.38,0.15 0.37,-2.17 0.37,-0.61 0.37,0.15 0.38,0.16 0.37,-0.61 0.37,-0.6 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,-0.59 0.37,-0.58 0.37,0.15 0.37,0.15 0.38,0.15 0.37,-0.58 0.37,-0.57 0.37,0.15 0.38,-0.56 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.14 0.37,0.15 0.37,0.14 0.37,-0.55 0.38,0.15 0.37,0.14 0.37,-1.92 0.37,-0.53 0.38,-0.53 0.37,-1.19 0.37,0.15 0.37,0.15 0.38,0.15 0.37,-1.83 0.37,-0.5 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,-0.5 0.38,0.15 0.37,0.15 0.37,-1.12 0.37,0.15 0.38,0.14 0.37,0.15 0.37,-0.48 0.37,-0.47 0.38,-0.47 0.37,0.14 0.37,0.15 0.37,0.14 0.38,0.15 0.37,0.14 0.37,-2.27 0.37,-0.45 0.38,0.15 0.37,-0.45 0.37,0.15 0.37,0.15 0.38,0.14 0.37,0.15 0.37,-0.44 0.37,0.14 0.38,0.14 0.37,0.15 0.37,0.14 0.37,-1.01 0.38,0.14 0.37,-1 0.37,0.15 0.37,0.14 0.38,-0.98 0.37,0.14 0.37,0.14 0.37,-0.97 0.38,0.15 0.37,0.14 0.37,0.14 0.37,0.15 0.38,-0.41 0.37,0.14 0.37,0.14 0.37,0.14 0.38,0.14 0.37,-0.4 0.37,0.14 0.37,-2.01 0.38,0.14 0.37,0.14 0.37,0.15 0.37,0.14 0.37,0.14 0.38,0.13 0.37,-0.38 0.37,-0.39 0.37,0.14 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-0.38 0.37,0.13 0.37,0.14 0.38,0.14 0.37,-0.89 0.37,0.13 0.37,-0.37 0.38,0.14 0.37,0.13 0.37,-0.37 0.37,-1.36 0.38,-0.36 0.37,0.14 0.37,-0.36 0.37,0.14 0.38,-0.85 0.37,-0.35 0.37,-0.82 0.37,-0.35 0.38,0.14 0.37,-1.29 0.37,-0.33 0.37,0.14 0.38,0.13 0.37,-0.33 0.37,-0.32 0.37,-0.79 0.38,-0.79 0.37,-0.31 0.37,-0.32 0.37,0.14 0.38,0.14 0.37,0.14 0.37,0.13 0.37,0.14 0.38,0.14 0.37,-0.31 0.37,0.13 0.37,0.14 0.38,-0.31 0.37,-0.76 0.37,0.14 0.37,-0.75 0.38,0.14 0.37,0.14 0.37,0.13 0.37,0.14 0.38,-1.17 0.37,0.13 0.37,0.14 0.37,0.13 0.38,-0.29 0.37,0.13 0.37,-0.29 0.37,0.13 0.38,0.14 0.37,-0.29 0.37,0.13 0.37,-0.29 0.38,-0.29 0.37,-0.28 0.37,0.13 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.14 0.37,-0.29 0.38,0.13 0.37,-0.69 0.37,0.13 0.37,0.13 0.38,0.13 0.37,-0.28 0.37,-0.28 0.37,0.13 0.38,-0.27 0.37,0.13 0.37,-0.28 0.37,-0.67 0.38,-1.07 0.37,0.14 0.37,-0.66 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.13 0.37,0.13 0.38,0.13 0.37,-0.66 0.37,-0.25 0.37,0.12 0.38,-1.02 0.37,-0.25 0.37,0.12 0.37,0.13 0.38,-0.63 0.37,-0.62 0.37,-0.62 0.37,-0.25 0.38,0.13 0.37,-0.24 0.37,0.13 0.37,0.13 0.38,-0.24 0.37,-0.24 0.37,0.13 0.37,-0.24 0.38,-0.24 0.37,0.13 0.37,-0.23 0.37,-0.96 0.38,-2.01 0.37,0.13 0.37,-0.22 0.37,0.13 0.38,0.13 0.37,0.12 0.37,0.13 0.37,0.13 0.38,0.13 0.37,0.13 0.37,-0.92 0.37,0.13 0.38,-0.22 0.37,0.13 0.37,0.12 0.37,0.13 0.38,0.13 0.37,0.12 0.37,0.13 0.37,0.12 0.38,0.13 0.37,-0.22 0.37,0.12 0.37,0.13 0.38,0.12 0.37,0.13 0.37,0.12 0.37,-0.22 0.38,-0.21 0.37,-0.22 0.37,0.13 0.37,-0.22 0.38,0.12 0.37,0.13 0.37,0.12 0.37,0.12 0.38,0.12 0.37,0.12 0.37,-0.21 0.37,0.12 0.38,-0.54 0.37,0.12 0.37,-0.21 0.37,0.12 0.38,-1.19 0.37,0.12 0.37,0.12 0.37,0.12 0.38,0.12 0.37,-0.2 0.37,0.12 0.37,-0.21 0.38,0.12 0.37,-0.2 0.37,0.12 0.37,-0.2 0.38,-0.2 0.37,-0.2 0.37,0.11 0.37,-0.19 0.38,0.11 0.37,0.12 0.37,0.12 0.37,0.12 0.38,-0.52 0.37,0.12 0.37,0.12 0.37,-0.2 0.38,-0.19 0.37,0.11 0.37,-0.5 0.37,-0.19 0.38,0.11 0.37,0.12 0.37,-0.5 0.37,0.12 0.38,-0.19 0.37,0.11 0.37,0.12 0.37,0.11 0.38,0.12 0.37,-0.49 0.37,-0.79 0.37,0.11 0.38,0.12 0.37,0.11 0.37,0.12 0.37,0.11 0.38,0.11 0.37,0.12 0.37,-0.19 0.37,-0.48 0.38,-0.18 0.37,0.11 0.37,-0.18 0.37,0.11 0.38,-0.47 0.37,-0.47 0.37,0.11 0.37,0.11 0.38,0.12 0.37,0.11 0.37,-0.47 0.37,0.11 0.38,0.12 0.37,0.11 0.37,0.11 0.37,0.11 0.38,0.11 0.37,0.11 0.37,-0.17 0.37,0.11 0.38,-0.18 0.37,-0.17 0.37,-0.18 0.37,0.11 0.38,0.11 0.37,-0.74 0.37,-0.73 0.37,0.11 0.37,-0.17 0.38,0.11 0.37,-0.44 0.37,-0.17 0.37,-0.17 0.38,-0.44 0.37,0.11 0.37,-0.16 0.37,0.11 0.38,-0.16 0.37,-0.44 0.37,-0.16 0.37,0.11 0.38,0.11 0.37,-0.43 0.37,0.11 0.37,-0.16 0.38,0.11 0.37,-0.43 0.37,-0.16 0.37,0.11 0.38,0.11 0.37,0.11 0.37,0.11 0.37,-0.16 0.38,0.11 0.37,0.11 0.37,-0.16 0.37,-0.16 0.38,0.11 0.37,-0.42 0.37,-0.15 0.37,-0.15 0.38,0.1 0.37,0.11 0.37,0.11 0.37,0.1 0.38,0.11 0.37,0.1 0.37,0.11 0.37,0.1 0.38,-0.15 0.37,0.11 0.37,0.1 0.37,-0.41 0.38,-0.41 0.37,0.11 0.37,-0.15 0.37,0.1 0.38,0.11 0.37,0.1 0.37,-0.65 0.37,0.1 0.38,-0.15 0.37,-0.4 0.37,0.11 0.37,0.1 0.38,0.11 0.37,-0.15 0.37,-0.64 0.37,0.1 0.38,-0.14 0.37,0.1 0.37,0.1 0.37,0.11 0.38,-0.15 0.37,-0.39 0.37,-0.38 0.37,0.1 0.38,-0.14 0.37,-1.83 0.37,-0.38 0.37,-0.84 0.38,0.1 0.37,-0.84 0.37,0.11 0.37,-0.14 0.38,-0.13 0.37,-0.12 0.37,0.1 0.37,-0.13 0.38,-0.59 0.37,0.1 0.37,-0.82 0.37,0.11 0.38,0.1 0.37,-0.12 0.37,-0.58 0.37,-0.13 0.38,-0.35 0.37,-0.12 0.37,-0.12 0.37,-0.12 0.38,-0.34 0.37,-0.12 0.37,0.1 0.37,-0.12 0.38,-0.12 0.37,-0.34 0.37,-0.12 0.37,0.11 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.11 0.38,-1.43 0.37,0.1 0.37,-1.62 0.37,0.1 0.38,0.11 0.37,0.1 0.37,0.1 0.37,-0.11 0.38,-0.32 0.37,0.1 0.37,-0.32 0.37,-0.53 0.38,0.1 0.37,0.1 0.37,-0.52 0.37,-0.32 0.38,-0.11 0.37,-0.1 0.37,0.1 0.37,-0.1 0.38,0.1 0.37,-2.56 0.37,0.1 0.37,-1.7 0.38,-0.1 0.37,0.11 0.37,-0.1 0.37,-1.08 0.38,-0.49 0.37,0.11 0.37,-0.1 0.37,-0.48 0.38,0.11 0.37,-0.09 0.37,-0.09 0.37,0.1 0.38,-0.67 0.37,0.1 0.37,-0.09 0.37,0.11 0.38,-0.47 0.37,-0.09 0.37,0.1 0.37,-0.09 0.38,0.11 0.37,-0.09 0.37,-0.28 0.37,-0.83 0.38,-0.65 0.37,-0.27 0.37,-0.08 0.37,0.1 0.38,-0.26 0.37,-0.82 0.37,-0.81 0.37,-0.08 0.38,0.11 0.37,0.1 0.37,-0.08 0.37,-0.08 0.38,0.1 0.37,-0.08 0.37,-0.43 0.37,0.1 0.38,-0.08 0.37,0.1 0.37,-0.07 0.37,-0.26 0.38,0.1 0.37,-0.43 0.37,-0.07 0.37,0.1 0.38,0.1 0.37,-0.25 0.37,0.1 0.37,-0.08 0.38,-1.12 0.37,-0.07 0.37,0.1 0.37,0.1 0.38,0.1 0.37,-0.07 0.37,0.1 0.37,-0.08 0.38,0.1 0.37,0.1 0.37,0.1 0.37,-0.24 0.38,0.1 0.37,0.1 0.37,-0.93 0.37,-0.24 0.38,-0.07 0.37,0.1 0.37,0.1 0.37,0.1 0.38,-0.07 0.37,-0.07 0.37,-0.07 0.37,-0.24 0.38,0.1 0.37,0.09 0.37,-0.57 0.37,-0.89 0.38,-0.23 0.37,-0.23 0.37,0.1 0.37,-0.07 0.38,-0.07 0.37,0.1 0.37,-0.06 0.37,0.09 0.38,-0.06 0.37,-1.04 0.37,-0.06 0.37,-0.22 0.38,0.09 0.37,-0.38 0.37,-0.54 0.37,0.1 0.38,-0.06 0.37,0.09 0.37,-0.37 0.37,0.09 0.38,-0.06 0.37,-0.06 0.37,-0.37 0.37,0.1 0.38,-1.31 0.37,0.1 0.37,-0.21 0.37,0.09 0.38,0.1 0.37,-0.06 0.37,-0.67 0.37,-0.05 0.37,0.09 0.38,0.1 0.37,0.1 0.37,-0.36 0.37,-0.36 0.38,0.09 0.37,-0.05 0.37,-0.36 0.37,-0.8 0.38,-0.2 0.37,0.1 0.37,-0.06 0.37,0.1 0.38,-0.05 0.37,-0.94 0.37,-0.05 0.37,-0.05 0.38,-0.05 0.37,0.09 0.37,-0.34 0.37,-0.48 0.38,-0.48 0.37,-0.2 0.37,-0.47 0.37,-0.34 0.38,-0.04 0.37,-0.05 0.37,-0.47 0.37,-0.19 0.38,-0.88 0.37,-0.05 0.37,-0.73 0.37,-0.04 0.38,-0.05 0.37,-0.18 0.37,0.1 0.37,-0.05 0.38,0.1 0.37,0.09 0.37,-0.04 0.37,-0.59 0.38,0.1 0.37,0.09 0.37,-0.04 0.37,-0.18 0.38,-0.45 0.37,-0.04 0.37,-0.31 0.37,-0.57 0.38,-1.36 0.37,-1.73 0.37,0.09 0.37,-0.03 0.38,0.09 0.37,-0.93 0.37,-0.29 0.37,-0.29 0.38,0.09 0.37,-0.91 0.37,0.09 0.37,-0.28 0.38,-0.16 0.37,-0.03 0.37,-0.03 0.37,-0.16 0.38,-0.52 0.37,-0.16 0.37,0.1 0.37,0.09 0.38,0.09 0.37,-0.76 0.37,-0.52 0.37,-0.03 0.38,-0.39 0.37,-0.27 0.37,0.1 0.37,0.09 0.38,-0.51 0.37,-0.15 0.37,0.1 0.37,-0.03 0.38,-0.15 0.37,-0.26 0.37,-0.03 0.37,-0.96 0.38,-1.08 0.37,-1.29 0.37,-0.82 0.37,-0.69 0.38,-1.03 0.37,-0.47 0.37,-1.66 0.37,-0.02 0.38,-0.01 0.37,0.09 0.37,-0.88 0.37,-0.02 0.38,-0.01 0.37,-0.98 0.37,-2.32 0.37,-0.12 0.38,-1.24 v -0.11"
               style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5264" />
            <path
               d="m 83.51,462.22 h 0.37 0.37 0.37 l 0.38,-69.13 0.37,11.69 0.37,8.31 0.37,6.21 0.38,4.82 0.37,3.84 0.37,-23.18 0.37,4.48 0.38,3.83 0.37,3.32 0.37,2.89 0.37,2.56 0.38,2.26 0.37,2.02 0.37,1.82 0.37,1.65 0.38,1.49 0.37,1.37 0.37,1.25 0.37,1.14 0.38,1.06 0.37,0.98 0.37,0.91 0.37,0.85 0.38,0.79 0.37,0.74 0.37,0.69 0.37,0.65 0.38,0.61 0.37,0.58 0.37,-8.23 0.37,0.74 0.38,0.69 0.37,0.67 0.37,0.63 0.37,-6.85 0.38,0.73 0.37,0.7 0.37,0.67 0.37,0.64 0.38,0.61 0.37,0.58 0.37,0.56 0.37,0.55 0.38,0.51 0.37,0.5 0.37,0.48 0.37,0.47 0.38,-5.17 0.37,0.53 0.37,0.5 0.37,0.49 0.38,0.47 0.37,0.45 0.37,0.44 0.37,0.43 0.38,0.41 0.37,0.41 0.37,0.38 0.37,-4.24 0.38,0.43 0.37,0.41 0.37,0.41 0.37,0.39 0.38,0.38 0.37,0.37 0.37,0.37 0.37,0.35 0.38,0.34 0.37,0.34 0.37,0.33 0.37,0.31 0.38,0.32 0.37,0.3 0.37,0.29 0.37,0.29 0.38,0.29 0.37,0.27 0.37,0.27 0.37,0.26 0.38,0.26 0.37,0.25 0.37,0.25 0.37,-3.18 0.38,0.27 0.37,0.26 0.37,-3 0.37,0.29 0.38,0.27 0.37,0.28 0.37,-2.82 0.37,0.29 0.38,0.29 0.37,0.27 0.37,0.28 0.37,0.27 0.38,0.26 0.37,0.26 0.37,-2.58 0.37,0.28 0.38,0.26 0.37,0.27 0.37,0.26 0.37,0.25 0.38,0.25 0.37,0.25 0.37,0.24 0.37,0.23 0.38,0.24 0.37,0.23 0.37,0.22 0.37,0.22 0.38,0.22 0.37,0.22 0.37,0.21 0.37,0.21 0.38,0.2 0.37,0.21 0.37,-2.2 0.37,0.22 0.38,0.2 0.37,0.21 0.37,0.2 0.37,0.2 0.38,0.2 0.37,0.2 0.37,0.19 0.37,0.19 0.38,0.18 0.37,0.19 0.37,0.18 0.37,0.18 0.38,0.17 0.37,0.18 0.37,0.17 0.37,0.17 0.38,0.17 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.16 0.37,-1.88 0.37,0.17 0.37,-1.83 0.38,-1.78 0.37,0.19 0.37,0.18 0.37,0.17 0.38,0.18 0.37,-1.71 0.37,0.19 0.37,0.18 0.38,-1.66 0.37,0.19 0.37,0.18 0.37,0.18 0.38,0.18 0.37,0.18 0.37,0.18 0.37,0.17 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,-1.54 0.37,0.18 0.37,0.16 0.37,0.17 0.38,0.17 0.37,0.16 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.16 0.37,0.15 0.37,0.15 0.37,0.16 0.38,0.15 0.37,-2.99 0.37,-1.38 0.37,0.16 0.38,0.17 0.37,0.16 0.37,0.16 0.37,-2.81 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.17 0.38,0.16 0.37,-2.69 0.37,0.18 0.37,0.17 0.38,0.17 0.37,-2.59 0.37,0.18 0.37,0.18 0.38,0.17 0.37,0.18 0.37,0.17 0.37,0.17 0.38,0.17 0.37,-3.76 0.37,0.18 0.37,0.18 0.38,0.18 0.37,0.17 0.37,0.18 0.37,0.17 0.38,0.18 0.37,0.17 0.37,-1.08 0.37,0.17 0.38,0.17 0.37,-1.05 0.37,0.17 0.37,-2.23 0.38,0.17 0.37,-1 0.37,0.18 0.37,-0.99 0.38,0.18 0.37,0.18 0.37,0.18 0.37,0.18 0.38,0.17 0.37,0.18 0.37,0.17 0.37,0.17 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,0.17 0.37,-0.94 0.37,-0.93 0.37,0.17 0.38,-1.99 0.37,0.17 0.37,-0.89 0.37,-0.87 0.38,-1.9 0.37,0.18 0.37,-0.84 0.37,0.18 0.38,0.18 0.37,0.17 0.37,-1.82 0.37,0.18 0.38,0.18 0.37,0.18 0.37,0.18 0.37,0.18 0.38,-0.8 0.37,-1.75 0.37,0.18 0.37,0.18 0.38,0.18 0.37,0.18 0.37,-0.77 0.37,0.18 0.38,0.18 0.37,0.17 0.37,0.18 0.37,-0.75 0.38,0.17 0.37,0.18 0.37,0.17 0.37,0.17 0.38,-0.74 0.37,0.18 0.37,-0.73 0.37,0.17 0.38,0.17 0.37,0.17 0.37,-0.71 0.37,0.17 0.38,0.17 0.37,0.16 0.37,0.17 0.37,0.17 0.38,0.16 0.37,0.17 0.37,0.16 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.15 0.37,0.16 0.37,0.16 0.37,0.15 0.38,-1.52 0.37,0.16 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.16 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.14 0.38,0.15 0.37,0.15 0.37,0.14 0.37,-1.45 0.38,0.14 0.37,0.15 0.37,0.15 0.37,0.14 0.38,0.15 0.37,0.14 0.37,0.14 0.37,0.15 0.38,0.14 0.37,0.14 0.37,0.14 0.37,-0.64 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.14 0.38,-0.62 0.37,0.13 0.37,-0.61 0.37,0.14 0.38,0.13 0.37,-1.35 0.37,0.14 0.37,-0.6 0.38,0.14 0.37,-0.59 0.37,0.14 0.37,-0.58 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.13 0.38,0.14 0.37,0.14 0.37,0.13 0.37,0.14 0.38,0.13 0.37,0.13 0.37,-1.27 0.37,0.14 0.38,0.13 0.37,0.14 0.37,0.13 0.37,0.13 0.38,0.13 0.37,0.14 0.37,0.13 0.37,-0.56 0.38,-1.23 0.37,0.14 0.37,-1.21 0.37,0.13 0.38,-1.85 0.37,0.14 0.37,0.13 0.37,0.14 0.38,0.13 0.37,0.14 0.37,-1.16 0.37,0.13 0.38,-1.14 0.37,0.13 0.37,-0.5 0.37,-1.12 0.38,-0.48 0.37,-0.48 0.37,-1.09 0.37,0.14 0.38,0.14 0.37,-0.47 0.37,-0.47 0.37,-0.46 0.38,0.14 0.37,-0.46 0.37,-0.45 0.37,0.14 0.38,0.14 0.37,-0.45 0.37,0.14 0.37,-0.44 0.38,-0.45 0.37,0.14 0.37,-0.43 0.37,0.14 0.38,-0.44 0.37,0.14 0.37,0.14 0.37,0.14 0.38,-0.43 0.37,0.14 0.37,-2.11 0.37,0.14 0.38,-0.41 0.37,0.14 0.37,0.14 0.37,0.14 0.38,-0.96 0.37,0.14 0.37,-0.4 0.37,0.14 0.38,0.14 0.37,0.14 0.37,0.13 0.37,-0.4 0.37,-0.92 0.38,0.13 0.37,-0.91 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-0.38 0.37,-0.9 0.37,0.14 0.38,-1.4 0.37,0.14 0.37,-0.37 0.37,0.14 0.38,-0.37 0.37,0.14 0.37,-0.36 0.37,0.14 0.38,0.14 0.37,0.13 0.37,-0.35 0.37,-0.36 0.38,-0.36 0.37,-0.35 0.37,0.14 0.37,0.14 0.38,0.13 0.37,0.14 0.37,-0.35 0.37,0.14 0.38,0.13 0.37,0.14 0.37,0.13 0.37,-0.82 0.38,0.13 0.37,0.14 0.37,0.13 0.37,0.13 0.38,-0.34 0.37,0.14 0.37,-1.75 0.37,0.14 0.38,0.13 0.37,-0.33 0.37,0.14 0.37,-0.33 0.38,-0.78 0.37,-0.33 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-0.77 0.37,0.13 0.37,0.13 0.38,-1.65 0.37,0.14 0.37,0.13 0.37,0.14 0.38,0.13 0.37,-1.62 0.37,-0.3 0.37,0.14 0.38,-0.3 0.37,-0.3 0.37,-0.29 0.37,-0.29 0.38,-0.29 0.37,0.13 0.37,-0.71 0.37,0.14 0.38,0.13 0.37,-1.12 0.37,-0.69 0.37,0.14 0.38,0.13 0.37,-0.68 0.37,0.13 0.37,0.13 0.38,0.14 0.37,0.13 0.37,0.13 0.37,0.14 0.38,0.13 0.37,-0.27 0.37,0.13 0.37,-0.27 0.38,-0.27 0.37,-0.67 0.37,0.13 0.37,0.13 0.38,-0.26 0.37,0.13 0.37,0.13 0.37,0.13 0.38,0.13 0.37,-0.26 0.37,0.13 0.37,-0.26 0.38,-0.65 0.37,0.13 0.37,0.13 0.37,-0.26 0.38,0.13 0.37,0.13 0.37,0.12 0.37,-0.25 0.38,-1.02 0.37,-0.63 0.37,0.13 0.37,0.13 0.38,-0.25 0.37,-0.24 0.37,-0.25 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.12 0.37,0.13 0.38,0.13 0.37,0.12 0.37,0.13 0.37,-0.98 0.38,0.13 0.37,-0.24 0.37,0.12 0.37,-0.6 0.38,0.13 0.37,-0.6 0.37,-0.94 0.37,-0.23 0.38,-0.59 0.37,0.13 0.37,0.13 0.37,0.12 0.38,0.13 0.37,0.12 0.37,0.13 0.37,0.12 0.38,-0.22 0.37,-0.23 0.37,-0.22 0.37,-0.22 0.38,0.12 0.37,0.12 0.37,-0.56 0.37,0.12 0.38,-0.22 0.37,0.13 0.37,-0.22 0.37,-0.22 0.38,0.13 0.37,0.12 0.37,0.12 0.37,0.12 0.38,-0.21 0.37,0.12 0.37,0.12 0.37,-0.21 0.38,0.12 0.37,-0.88 0.37,0.12 0.37,0.12 0.38,0.12 0.37,0.12 0.37,-0.21 0.37,-0.21 0.38,-0.21 0.37,0.12 0.37,0.12 0.37,0.12 0.38,0.12 0.37,0.12 0.37,-0.54 0.37,-0.2 0.38,0.12 0.37,-0.21 0.37,-0.2 0.37,-1.16 0.38,0.12 0.37,-0.2 0.37,0.12 0.37,0.12 0.38,-0.2 0.37,-0.2 0.37,0.12 0.37,-0.51 0.38,0.12 0.37,0.12 0.37,-0.2 0.37,-0.81 0.38,0.12 0.37,0.11 0.37,-0.49 0.37,0.11 0.38,0.12 0.37,-0.19 0.37,0.12 0.37,0.12 0.38,0.11 0.37,0.12 0.37,0.11 0.37,-0.18 0.38,0.11 0.37,-0.19 0.37,-0.18 0.37,0.11 0.38,0.12 0.37,0.11 0.37,0.12 0.37,-0.49 0.38,0.11 0.37,0.12 0.37,-0.19 0.37,0.12 0.38,0.11 0.37,0.11 0.37,-0.18 0.37,0.11 0.38,0.12 0.37,0.11 0.37,0.11 0.37,0.11 0.38,-0.18 0.37,0.11 0.37,-0.18 0.37,0.11 0.38,0.11 0.37,0.11 0.37,-0.18 0.37,0.11 0.38,-0.18 0.37,0.11 0.37,0.11 0.37,-0.18 0.38,0.11 0.37,-0.47 0.37,-1.03 0.37,0.11 0.38,0.11 0.37,0.11 0.37,0.1 0.37,0.11 0.37,-0.17 0.38,0.11 0.37,-0.74 0.37,-0.45 0.37,0.11 0.38,-0.17 0.37,-0.17 0.37,0.11 0.37,-0.72 0.38,-0.99 0.37,0.11 0.37,-0.17 0.37,-0.16 0.38,0.11 0.37,0.11 0.37,0.11 0.37,0.11 0.38,0.1 0.37,-0.43 0.37,-0.43 0.37,0.11 0.38,0.11 0.37,0.11 0.37,-0.96 0.37,0.1 0.38,0.11 0.37,0.11 0.37,0.11 0.37,0.1 0.38,0.11 0.37,0.11 0.37,0.1 0.37,-0.68 0.38,-0.68 0.37,0.11 0.37,0.11 0.37,0.1 0.38,0.11 0.37,-0.16 0.37,-0.41 0.37,0.11 0.38,-0.15 0.37,-0.16 0.37,-0.4 0.37,-0.15 0.38,0.1 0.37,-0.14 0.37,0.1 0.37,-0.15 0.38,0.11 0.37,-0.4 0.37,-0.65 0.37,0.1 0.38,-0.64 0.37,-0.39 0.37,-0.88 0.37,0.11 0.38,0.1 0.37,0.11 0.37,-0.14 0.37,-0.14 0.38,0.1 0.37,0.11 0.37,-0.39 0.37,-0.62 0.38,0.11 0.37,0.1 0.37,-0.37 0.37,-0.85 0.38,-1.08 0.37,-0.13 0.37,-0.37 0.37,0.11 0.38,-0.36 0.37,-0.83 0.37,-0.12 0.37,0.1 0.38,-0.81 0.37,-0.13 0.37,-0.12 0.37,-0.35 0.38,-0.12 0.37,-0.12 0.37,0.11 0.37,0.1 0.38,-0.12 0.37,0.11 0.37,0.1 0.37,0.1 0.38,0.11 0.37,-0.35 0.37,0.11 0.37,-0.12 0.38,0.1 0.37,0.11 0.37,0.1 0.37,-0.12 0.38,-0.34 0.37,0.1 0.37,-2.09 0.37,0.11 0.38,-0.12 0.37,-0.33 0.37,-0.32 0.37,0.1 0.38,-0.33 0.37,0.11 0.37,-0.11 0.37,-0.11 0.38,0.1 0.37,0.1 0.37,0.1 0.37,-0.32 0.38,0.11 0.37,-0.11 0.37,-0.32 0.37,-0.95 0.38,0.1 0.37,0.1 0.37,-0.1 0.37,-0.32 0.38,0.11 0.37,0.1 0.37,-0.11 0.37,-0.1 0.38,-0.72 0.37,-0.31 0.37,-0.1 0.37,-0.1 0.38,0.1 0.37,-0.71 0.37,-0.1 0.37,-0.7 0.38,-0.5 0.37,-0.69 0.37,-0.1 0.37,-0.68 0.38,-0.29 0.37,-0.09 0.37,0.1 0.37,-0.09 0.38,-0.29 0.37,0.1 0.37,-0.28 0.37,-0.86 0.38,-0.09 0.37,-0.09 0.37,-3.66 0.37,0.11 0.38,0.1 0.37,-0.08 0.37,-0.08 0.37,-0.08 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.11 0.38,0.1 0.37,-0.08 0.37,0.1 0.37,-0.63 0.38,0.1 0.37,0.1 0.37,-0.8 0.37,-0.44 0.38,-0.61 0.37,0.1 0.37,-0.79 0.37,-0.07 0.38,-0.08 0.37,-0.25 0.37,-0.42 0.37,0.1 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.1 0.38,0.11 0.37,0.1 0.37,0.1 0.37,-0.08 0.38,0.1 0.37,0.1 0.37,0.1 0.37,0.1 0.38,-0.07 0.37,0.1 0.37,0.1 0.37,-0.59 0.38,0.1 0.37,0.09 0.37,-0.93 0.37,-0.58 0.38,0.1 0.37,0.1 0.37,-0.07 0.37,0.1 0.38,0.1 0.37,-0.07 0.37,-0.07 0.37,-0.24 0.38,0.1 0.37,0.1 0.37,-0.57 0.37,0.09 0.38,-0.23 0.37,0.1 0.37,-0.4 0.37,-0.57 0.38,0.1 0.37,-0.39 0.37,-1.05 0.37,0.1 0.38,-0.06 0.37,0.1 0.37,-0.07 0.37,0.1 0.38,0.1 0.37,-0.07 0.37,0.1 0.37,0.1 0.38,0.1 0.37,-0.39 0.37,-0.22 0.37,0.09 0.38,-0.54 0.37,0.1 0.37,0.1 0.37,0.09 0.38,-1.01 0.37,-0.21 0.37,0.09 0.37,-0.84 0.38,0.1 0.37,-0.53 0.37,-0.05 0.37,-0.98 0.38,-0.36 0.37,0.09 0.37,-0.66 0.37,0.1 0.37,0.09 0.38,0.1 0.37,-0.66 0.37,-0.35 0.37,0.1 0.38,0.09 0.37,-0.35 0.37,0.1 0.37,0.09 0.38,-0.35 0.37,-0.78 0.37,-0.2 0.37,0.09 0.38,0.1 0.37,0.09 0.37,-0.05 0.37,-0.05 0.38,0.1 0.37,0.09 0.37,-0.92 0.37,-1.05 0.38,0.09 0.37,-0.05 0.37,-0.33 0.37,0.1 0.38,0.09 0.37,-0.47 0.37,0.09 0.37,-0.04 0.38,-0.75 0.37,0.09 0.37,-0.18 0.37,-0.6 0.38,0.09 0.37,-0.6 0.37,-0.18 0.37,-1.27 0.38,0.09 0.37,-0.04 0.37,0.1 0.37,-0.59 0.38,0.1 0.37,0.09 0.37,-0.04 0.37,-0.84 0.38,-0.04 0.37,-0.04 0.37,0.09 0.37,-0.04 0.38,0.1 0.37,-0.44 0.37,0.1 0.37,-0.96 0.38,-0.82 0.37,-0.04 0.37,-1.45 0.37,-0.67 0.38,-1.04 0.37,-0.03 0.37,0.09 0.37,-0.28 0.38,0.09 0.37,-0.28 0.37,-0.52 0.37,0.09 0.38,-0.28 0.37,-0.03 0.37,-0.27 0.37,-0.16 0.38,0.1 0.37,-0.15 0.37,-0.15 0.37,-0.03 0.38,0.09 0.37,-0.03 0.37,0.09 0.37,-0.27 0.38,0.09 0.37,0.1 0.37,-0.63 0.37,-0.03 0.38,-0.51 0.37,-0.5 0.37,-1.56 0.37,0.09 0.38,-0.49 0.37,-0.71 0.37,0.09 0.37,-1.17 0.38,-3.05 0.37,-0.02 0.37,-1.44 0.37,-0.12 0.38,-0.24 0.37,-0.01 0.37,-0.34 0.37,-0.02 0.38,-0.44 0.37,-0.66 0.37,-2.42 0.37,-0.12 0.38,-0.93 v -0.1 -0.11"
               style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5266" />
            <path
               d="m 83.51,462.22 h 0.37 0.37 0.37 0.38 l 0.37,-57.44 0.37,8.31 0.37,6.21 0.38,4.82 0.37,3.84 0.37,3.14 0.37,-21.84 0.38,3.83 0.37,3.32 0.37,2.89 0.37,2.56 0.38,2.26 0.37,2.02 0.37,1.82 0.37,1.65 0.38,1.49 0.37,1.37 0.37,1.25 0.37,1.14 0.38,1.06 0.37,0.98 0.37,0.91 0.37,0.85 0.38,0.79 0.37,0.74 0.37,0.69 0.37,0.65 0.38,0.61 0.37,0.58 0.37,0.55 0.37,0.51 0.38,0.49 0.37,0.46 0.37,-7.51 0.37,0.59 0.38,0.58 0.37,0.54 0.37,-6.46 0.37,0.64 0.38,0.61 0.37,0.58 0.37,0.56 0.37,0.55 0.38,0.51 0.37,0.5 0.37,-5.32 0.37,0.56 0.38,0.54 0.37,0.53 0.37,0.5 0.37,0.49 0.38,0.47 0.37,0.45 0.37,0.44 0.37,0.43 0.38,0.41 0.37,-4.35 0.37,0.46 0.37,0.44 0.38,0.43 0.37,0.41 0.37,0.41 0.37,0.39 0.38,0.38 0.37,0.37 0.37,0.37 0.37,0.35 0.38,0.34 0.37,0.34 0.37,0.33 0.37,0.31 0.38,0.32 0.37,0.3 0.37,-3.47 0.37,0.33 0.38,0.33 0.37,0.31 0.37,0.31 0.37,0.3 0.38,-3.16 0.37,0.32 0.37,0.32 0.37,0.3 0.38,0.31 0.37,0.29 0.37,0.29 0.37,0.29 0.38,0.27 0.37,0.28 0.37,0.27 0.37,0.26 0.38,0.25 0.37,0.26 0.37,0.24 0.37,0.25 0.38,0.23 0.37,0.24 0.37,0.23 0.37,0.22 0.38,0.22 0.37,0.22 0.37,0.22 0.37,0.21 0.38,0.2 0.37,0.2 0.37,0.2 0.37,0.2 0.38,0.19 0.37,0.19 0.37,0.19 0.37,0.18 0.38,0.18 0.37,0.18 0.37,-2.36 0.37,0.19 0.38,0.19 0.37,0.18 0.37,0.18 0.37,0.18 0.38,0.18 0.37,0.17 0.37,0.17 0.37,-2.17 0.38,0.18 0.37,0.18 0.37,0.17 0.37,0.18 0.38,0.17 0.37,0.17 0.37,0.17 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.15 0.37,-1.94 0.38,0.16 0.37,0.16 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.14 0.37,0.15 0.38,0.14 0.37,0.14 0.37,-1.79 0.37,0.15 0.38,-1.75 0.37,0.16 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.14 0.37,0.15 0.38,0.14 0.37,0.14 0.37,0.14 0.37,-1.62 0.38,0.14 0.37,0.15 0.37,0.14 0.37,0.14 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.13 0.37,-1.53 0.38,0.14 0.37,-1.5 0.37,0.15 0.37,0.14 0.38,0.15 0.37,0.14 0.37,0.14 0.37,0.13 0.38,-1.43 0.37,-1.4 0.37,0.14 0.37,0.15 0.38,0.15 0.37,0.14 0.37,0.14 0.37,0.15 0.38,-1.35 0.37,0.14 0.37,0.15 0.37,-1.32 0.38,0.15 0.37,0.15 0.37,0.14 0.37,0.14 0.38,0.15 0.37,-1.27 0.37,0.14 0.37,0.15 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.14 0.38,0.14 0.37,0.13 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-2.53 0.37,0.15 0.37,0.14 0.38,0.13 0.37,0.14 0.37,-1.15 0.37,-1.13 0.38,0.14 0.37,-1.11 0.37,0.15 0.37,0.14 0.38,-1.09 0.37,0.15 0.37,0.15 0.37,0.14 0.38,0.14 0.37,-1.06 0.37,0.15 0.37,0.14 0.38,-1.03 0.37,0.14 0.37,0.15 0.37,0.14 0.38,0.14 0.37,-5.54 0.37,-0.95 0.37,0.16 0.38,0.17 0.37,0.15 0.37,-2.01 0.37,-0.9 0.38,0.16 0.37,0.17 0.37,0.16 0.37,0.16 0.38,0.16 0.37,0.16 0.37,0.16 0.37,0.16 0.38,0.15 0.37,0.16 0.37,0.15 0.37,0.16 0.38,0.15 0.37,-0.87 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,-1.85 0.37,0.16 0.38,0.15 0.37,-0.83 0.37,-1.78 0.37,-0.79 0.38,0.16 0.37,0.15 0.37,0.16 0.37,0.16 0.38,0.15 0.37,0.16 0.37,-0.78 0.37,0.15 0.38,-0.76 0.37,0.15 0.37,0.16 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.16 0.37,0.15 0.38,0.14 0.37,0.15 0.37,-2.52 0.37,0.16 0.38,0.15 0.37,-2.45 0.37,0.16 0.37,0.15 0.38,0.16 0.37,0.15 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,-0.69 0.37,-1.51 0.38,0.15 0.37,0.15 0.37,-0.66 0.37,0.15 0.38,-0.66 0.37,-0.65 0.37,0.15 0.37,0.16 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.14 0.37,-0.63 0.37,-1.4 0.37,0.15 0.38,-0.62 0.37,0.15 0.37,-0.6 0.37,-1.35 0.38,0.15 0.37,0.15 0.37,0.15 0.37,-0.59 0.38,0.15 0.37,0.15 0.37,0.15 0.37,0.15 0.38,0.14 0.37,-0.58 0.37,0.15 0.37,0.15 0.38,0.14 0.37,0.15 0.37,0.14 0.37,0.14 0.38,0.15 0.37,0.14 0.37,0.14 0.37,-0.57 0.38,-0.56 0.37,0.14 0.37,0.14 0.37,0.15 0.38,0.13 0.37,-0.55 0.37,0.14 0.37,0.14 0.38,0.14 0.37,0.14 0.37,0.13 0.37,-1.22 0.38,0.14 0.37,0.14 0.37,0.13 0.37,-0.53 0.38,0.14 0.37,0.13 0.37,0.14 0.37,0.14 0.38,0.13 0.37,0.13 0.37,0.14 0.37,-0.53 0.38,-0.52 0.37,0.14 0.37,0.13 0.37,0.13 0.38,0.13 0.37,-0.51 0.37,-2.42 0.37,-0.49 0.38,-0.49 0.37,-1.72 0.37,0.14 0.37,0.14 0.38,0.14 0.37,-1.08 0.37,0.14 0.37,0.13 0.38,0.14 0.37,0.14 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-1.06 0.37,0.14 0.37,-0.45 0.38,-0.46 0.37,0.14 0.37,0.14 0.37,0.13 0.38,0.14 0.37,-2.18 0.37,0.14 0.37,-1 0.38,0.14 0.37,0.14 0.37,0.13 0.37,0.14 0.38,0.14 0.37,0.13 0.37,0.14 0.37,-0.43 0.38,-0.42 0.37,-0.97 0.37,0.14 0.37,0.14 0.38,0.13 0.37,0.14 0.37,0.13 0.37,0.14 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.14 0.37,0.13 0.38,0.13 0.37,-0.41 0.37,-0.4 0.37,0.13 0.38,-0.94 0.37,-0.39 0.37,-1.44 0.37,-0.39 0.38,0.14 0.37,-0.9 0.37,0.14 0.37,-1.4 0.38,0.14 0.37,0.13 0.37,0.14 0.37,-0.37 0.38,0.13 0.37,-0.36 0.37,0.13 0.37,0.13 0.38,0.14 0.37,0.13 0.37,-0.36 0.37,0.13 0.38,0.13 0.37,0.13 0.37,-0.36 0.37,0.13 0.38,0.13 0.37,-0.84 0.37,0.13 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.13 0.37,0.13 0.38,-0.35 0.37,0.12 0.37,0.13 0.37,-0.35 0.38,-1.28 0.37,-0.34 0.37,-2.65 0.37,-0.33 0.38,0.13 0.37,0.14 0.37,0.13 0.37,0.13 0.38,-0.32 0.37,0.13 0.37,0.13 0.37,0.13 0.38,0.13 0.37,-2.99 0.37,0.13 0.37,0.14 0.38,0.13 0.37,0.13 0.37,0.13 0.37,0.13 0.38,0.13 0.37,0.13 0.37,0.13 0.37,-0.74 0.38,0.13 0.37,0.13 0.37,0.12 0.37,-2.43 0.38,0.13 0.37,-1.55 0.37,0.14 0.37,0.13 0.38,0.13 0.37,0.13 0.37,-0.29 0.37,0.13 0.38,-0.28 0.37,0.13 0.37,0.13 0.37,-0.28 0.38,-0.28 0.37,0.13 0.37,-0.28 0.37,0.13 0.38,0.13 0.37,-0.28 0.37,0.13 0.37,0.13 0.38,0.12 0.37,-0.27 0.37,0.13 0.37,-0.28 0.38,-0.27 0.37,0.13 0.37,-0.27 0.37,0.13 0.38,0.12 0.37,0.13 0.37,0.12 0.37,0.13 0.38,0.12 0.37,0.12 0.37,0.13 0.37,-0.66 0.38,-0.65 0.37,0.12 0.37,-1.41 0.37,0.13 0.38,-0.26 0.37,-0.25 0.37,-0.25 0.37,0.12 0.38,-0.62 0.37,0.12 0.37,0.12 0.37,0.13 0.38,-0.25 0.37,0.12 0.37,0.13 0.37,0.12 0.38,-0.25 0.37,-0.24 0.37,-0.25 0.37,0.12 0.38,0.13 0.37,0.12 0.37,0.12 0.37,0.12 0.38,0.12 0.37,0.12 0.37,-0.24 0.37,0.12 0.38,0.12 0.37,0.12 0.37,-0.24 0.37,-0.25 0.38,-0.23 0.37,-0.24 0.37,0.12 0.37,0.12 0.38,-0.24 0.37,-0.24 0.37,0.12 0.37,-0.58 0.38,-1.28 0.37,0.12 0.37,0.12 0.37,-0.23 0.38,0.12 0.37,0.12 0.37,-0.23 0.37,0.12 0.38,0.12 0.37,0.11 0.37,0.12 0.37,0.12 0.38,-0.57 0.37,-0.22 0.37,0.12 0.37,0.11 0.38,-1.56 0.37,0.12 0.37,-0.22 0.37,-0.21 0.38,0.11 0.37,-0.21 0.37,-0.21 0.37,0.11 0.38,0.12 0.37,0.12 0.37,0.11 0.37,0.12 0.38,-0.21 0.37,0.11 0.37,-0.21 0.37,0.12 0.38,-0.21 0.37,-0.21 0.37,0.11 0.37,0.12 0.38,0.11 0.37,-0.21 0.37,0.12 0.37,0.11 0.38,0.11 0.37,0.12 0.37,0.11 0.37,-0.21 0.38,0.12 0.37,0.11 0.37,-0.21 0.37,0.11 0.38,0.12 0.37,-1.15 0.37,0.11 0.37,0.11 0.38,-0.51 0.37,-1.44 0.37,0.11 0.37,0.11 0.38,0.12 0.37,0.11 0.37,-0.2 0.37,0.12 0.38,-0.2 0.37,0.11 0.37,-0.19 0.37,-0.49 0.38,0.11 0.37,0.11 0.37,0.11 0.37,-1.09 0.38,-0.19 0.37,-0.78 0.37,0.11 0.37,0.12 0.38,0.11 0.37,-0.19 0.37,0.11 0.37,0.12 0.38,-0.19 0.37,0.11 0.37,-0.18 0.37,-0.18 0.38,0.11 0.37,0.11 0.37,-1.05 0.37,-0.18 0.38,-0.46 0.37,0.11 0.37,0.11 0.37,-0.17 0.38,0.11 0.37,0.11 0.37,0.1 0.37,0.11 0.37,0.11 0.38,-0.17 0.37,-1.02 0.37,0.11 0.37,-0.17 0.38,0.11 0.37,0.11 0.37,0.1 0.37,-0.17 0.38,-0.17 0.37,0.11 0.37,0.11 0.37,0.11 0.38,-0.45 0.37,-0.16 0.37,-0.72 0.37,-1.51 0.38,-0.17 0.37,0.11 0.37,-0.16 0.37,0.11 0.38,0.11 0.37,-0.43 0.37,-0.68 0.37,-0.16 0.38,0.11 0.37,-0.16 0.37,0.11 0.37,0.11 0.38,0.11 0.37,0.1 0.37,0.11 0.37,0.11 0.38,-0.42 0.37,0.11 0.37,-0.41 0.37,0.1 0.38,-0.15 0.37,0.11 0.37,0.1 0.37,-0.41 0.38,0.11 0.37,0.1 0.37,-1.17 0.37,-0.14 0.38,0.1 0.37,0.11 0.37,0.1 0.37,-0.14 0.38,-0.15 0.37,0.11 0.37,-0.65 0.37,0.1 0.38,0.11 0.37,-0.14 0.37,0.1 0.37,0.1 0.38,-0.14 0.37,-0.64 0.37,0.11 0.37,0.1 0.38,-0.38 0.37,0.1 0.37,0.1 0.37,0.11 0.38,0.1 0.37,-0.14 0.37,0.1 0.37,-0.14 0.38,0.11 0.37,-0.14 0.37,0.1 0.37,0.1 0.38,-0.14 0.37,-0.38 0.37,-0.61 0.37,0.1 0.38,-0.61 0.37,-0.14 0.37,0.1 0.37,-0.13 0.38,-0.13 0.37,0.1 0.37,0.1 0.37,-0.37 0.38,-1.29 0.37,-0.13 0.37,0.1 0.37,-0.13 0.38,-0.13 0.37,0.1 0.37,0.11 0.37,-0.82 0.38,0.1 0.37,-0.8 0.37,-0.35 0.37,-0.8 0.38,-1.01 0.37,-0.12 0.37,0.11 0.37,0.1 0.38,-0.12 0.37,0.1 0.37,-0.34 0.37,-0.11 0.38,-1.21 0.37,-0.11 0.37,-0.11 0.37,0.1 0.38,-0.12 0.37,-0.11 0.37,-1.6 0.37,-0.11 0.38,0.1 0.37,-0.95 0.37,0.11 0.37,0.1 0.38,-0.32 0.37,0.11 0.37,0.1 0.37,0.1 0.38,0.1 0.37,-0.1 0.37,-0.11 0.37,-0.72 0.38,0.1 0.37,-0.31 0.37,0.1 0.37,0.1 0.38,0.11 0.37,0.1 0.37,0.1 0.37,-0.31 0.38,0.1 0.37,0.1 0.37,-0.3 0.37,-0.91 0.38,-0.7 0.37,-0.1 0.37,-0.5 0.37,-0.09 0.38,-0.49 0.37,0.1 0.37,-0.1 0.37,-0.09 0.38,0.1 0.37,0.1 0.37,-1.27 0.37,-0.67 0.38,-0.47 0.37,-0.66 0.37,-0.09 0.37,-0.09 0.38,-0.09 0.37,0.1 0.37,-0.08 0.37,0.1 0.38,-0.09 0.37,0.1 0.37,-0.65 0.37,0.1 0.38,-0.09 0.37,0.1 0.37,0.1 0.37,-0.45 0.38,0.1 0.37,-0.09 0.37,-0.08 0.37,-0.09 0.38,-1.36 0.37,-0.63 0.37,0.1 0.37,0.1 0.38,0.1 0.37,-0.98 0.37,-0.44 0.37,0.1 0.38,0.1 0.37,-0.07 0.37,0.1 0.37,-0.26 0.38,0.1 0.37,0.1 0.37,-0.25 0.37,-0.08 0.38,0.1 0.37,0.1 0.37,-0.26 0.37,-0.42 0.38,-0.43 0.37,0.1 0.37,-0.25 0.37,0.1 0.38,0.1 0.37,-0.07 0.37,-0.08 0.37,-0.07 0.38,0.1 0.37,-0.42 0.37,0.1 0.37,-0.25 0.38,-0.58 0.37,-0.07 0.37,0.1 0.37,-0.92 0.38,0.1 0.37,-0.24 0.37,0.1 0.37,-0.73 0.38,0.09 0.37,0.1 0.37,0.1 0.37,0.1 0.38,-0.24 0.37,-0.39 0.37,-0.07 0.37,-0.07 0.38,0.1 0.37,0.09 0.37,-0.06 0.37,-0.23 0.38,-0.56 0.37,-0.71 0.37,0.1 0.37,-0.07 0.38,-0.22 0.37,-0.06 0.37,-0.54 0.37,-0.07 0.38,-0.53 0.37,0.09 0.37,-0.53 0.37,-0.06 0.38,-0.06 0.37,0.09 0.37,0.1 0.37,-0.22 0.38,-0.06 0.37,-0.06 0.37,-0.83 0.37,0.1 0.37,0.09 0.38,-0.06 0.37,-0.21 0.37,0.1 0.37,-0.37 0.38,-0.36 0.37,0.1 0.37,-0.06 0.37,0.1 0.38,-0.97 0.37,0.1 0.37,-0.06 0.37,-0.65 0.38,-0.05 0.37,-0.5 0.37,-0.05 0.37,-0.2 0.38,-0.94 0.37,-0.05 0.37,0.1 0.37,0.09 0.38,-0.63 0.37,-0.05 0.37,-0.34 0.37,-0.48 0.38,-0.05 0.37,-0.04 0.37,-0.48 0.37,-0.33 0.38,-2.01 0.37,0.1 0.37,-0.05 0.37,0.1 0.38,-0.05 0.37,-0.32 0.37,-0.04 0.37,0.09 0.38,-0.04 0.37,-1.54 0.37,-0.59 0.37,-0.71 0.38,0.1 0.37,-0.31 0.37,0.1 0.37,0.09 0.38,0.1 0.37,0.09 0.37,-0.04 0.37,-0.3 0.38,-0.44 0.37,-0.17 0.37,-0.03 0.37,0.09 0.38,0.09 0.37,-0.82 0.37,0.09 0.37,-0.03 0.38,-0.04 0.37,0.09 0.37,-0.55 0.37,-0.55 0.38,-0.29 0.37,-0.16 0.37,0.09 0.37,0.1 0.38,-0.04 0.37,-0.03 0.37,0.09 0.37,0.09 0.38,0.09 0.37,-0.28 0.37,-0.92 0.37,0.09 0.38,-0.15 0.37,-0.04 0.37,-0.4 0.37,-0.03 0.38,-0.78 0.37,0.1 0.37,-0.77 0.37,0.09 0.38,-1.48 0.37,-2.05 0.37,-1.2 0.37,-2.78 0.38,0.1 0.37,-0.81 0.37,-0.14 0.37,-1.35 0.38,-0.24 0.37,-0.35 0.37,-0.01 0.37,-0.02 0.38,0.09 0.37,-0.02 0.37,-0.34 0.37,-0.56 0.38,-2.26 0.37,-0.12 0.37,-0.12 0.37,-0.12 0.38,-2.7 v -0.11"
               style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5268" />
            <path
               d="M 64.88,294.04 H 474.62"
               style="fill:none;stroke:#bebebe;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:1.07, 3.2;stroke-dashoffset:0;stroke-opacity:1"
               id="path5270" />
            <path
               d="M 64.88,71.11 H 474.62 V 480.85 H 64.88 Z"
               style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
               id="path5272" />
          </g>
        </g>
        <g
           id="g5274" />
        <g
           id="g5276" />
        <g
           id="g5278" />
        <g
           id="g5280" />
        <g
           id="g5282" />
        <g
           id="g5284" />
        <g
           id="g5286" />
        <g
           id="g5288" />
        <g
           id="g5290" />
        <g
           id="g5292" />
        <g
           id="g5294" />
        <g
           id="g5296" />
        <g
           id="g5298" />
        <g
           id="g5300" />
        <g
           id="g5302" />
        <g
           id="g5304">
          <text
             transform="matrix(1,0,0,-1,42.44,86.5)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5308"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5306">0.00</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,179.62)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5312"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5310">0.25</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,272.75)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5316"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5314">0.50</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,365.87)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5320"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5318">0.75</tspan></text>
          <text
             transform="matrix(1,0,0,-1,42.44,458.99)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5324"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5322">1.00</tspan></text>
        </g>
        <g
           id="g5326" />
        <g
           id="g5328">
          <path
             d="m 62.14,89.73 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5330" />
          <path
             d="m 62.14,182.86 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5332" />
          <path
             d="m 62.14,275.98 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5334" />
          <path
             d="m 62.14,369.1 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5336" />
          <path
             d="m 62.14,462.22 h 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5338" />
        </g>
        <g
           id="g5340" />
        <g
           id="g5342" />
        <g
           id="g5344" />
        <g
           id="g5346" />
        <g
           id="g5348" />
        <g
           id="g5350" />
        <g
           id="g5352" />
        <g
           id="g5354" />
        <g
           id="g5356" />
        <g
           id="g5358">
          <path
             d="m 83.51,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5360" />
          <path
             d="m 176.63,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5362" />
          <path
             d="m 269.75,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5364" />
          <path
             d="m 362.87,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5366" />
          <path
             d="m 456,68.37 v 2.74"
             style="fill:none;stroke:#333333;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5368" />
        </g>
        <g
           id="g5370" />
        <g
           id="g5372">
          <text
             transform="matrix(1,0,0,-1,74.75,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5376"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5374">0.00</tspan></text>
          <text
             transform="matrix(1,0,0,-1,167.87,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5380"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5378">0.25</tspan></text>
          <text
             transform="matrix(1,0,0,-1,260.99,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5384"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5382">0.50</tspan></text>
          <text
             transform="matrix(1,0,0,-1,354.12,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5388"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5386">0.75</tspan></text>
          <text
             transform="matrix(1,0,0,-1,447.24,59.71)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#4d4d4d;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5392"><tspan
               x="0 5.0040002 7.506 12.51"
               y="0"
               sodipodi:role="line"
               id="tspan5390">1.00</tspan></text>
        </g>
        <g
           id="g5394" />
        <g
           id="g5396" />
        <g
           id="g5398" />
        <g
           id="g5400" />
        <g
           id="g5402" />
        <g
           id="g5404">
          <text
             transform="matrix(1,0,0,-1,254.47,47.1)"
             style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5408"><tspan
               x="0 7.9419999 14.058 19.558001 25.674 28.115999"
               y="0"
               sodipodi:role="line"
               id="tspan5406">Recall</tspan></text>
        </g>
        <g
           id="g5410" />
        <g
           id="g5412">
          <text
             transform="matrix(0,1,1,0,37.28,253.36)"
             style="font-variant:normal;font-weight:normal;font-size:11px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5416"><tspan
               x="0 7.3369999 11 17.115999 22.615999 25.058001 30.558001 33 39.116001"
               y="0"
               sodipodi:role="line"
               id="tspan5414">Precision</tspan></text>
        </g>
        <g
           id="g5418" />
        <g
           id="g5420" />
        <g
           id="g5422" />
        <g
           id="g5424" />
        <g
           id="g5426" />
        <g
           id="g5428" />
        <g
           id="g5430" />
        <g
           id="g5432">
          <path
             d="M 146.93,5.48 H 392.56 V 33.72 H 146.93 Z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path5434" />
        </g>
        <g
           id="g5436" />
        <g
           id="g5438" />
        <g
           id="g5440" />
        <g
           id="g5442">
          <path
             d="m 157.89,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path5444" />
        </g>
        <g
           id="g5446" />
        <g
           id="g5448">
          <path
             d="m 159.62,19.6 h 13.83"
             style="fill:none;stroke:#f8766d;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5450" />
        </g>
        <g
           id="g5452" />
        <g
           id="g5454">
          <path
             d="m 264.66,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path5456" />
        </g>
        <g
           id="g5458" />
        <g
           id="g5460">
          <path
             d="m 266.39,19.6 h 13.82"
             style="fill:none;stroke:#00ba38;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5462" />
        </g>
        <g
           id="g5464" />
        <g
           id="g5466">
          <path
             d="m 343.83,10.96 h 17.28 v 17.28 h -17.28 z"
             style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="path5468" />
        </g>
        <g
           id="g5470" />
        <g
           id="g5472">
          <path
             d="m 345.56,19.6 h 13.82"
             style="fill:none;stroke:#619cff;stroke-width:1.07;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1"
             id="path5474" />
        </g>
        <g
           id="g5476" />
        <g
           id="g5478" />
        <g
           id="g5480" />
        <g
           id="g5482">
          <text
             transform="matrix(1,0,0,-1,180.65,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5486"><tspan
               x="0 5.0040002 10.008 15.012 17.01 21.51 24.011999 26.01 30.51 33.012001 39.509998 44.514 49.518002 52.514999 57.519001 62.019001 66.518997 68.516998 73.521004"
               y="0"
               sodipodi:role="line"
               id="tspan5484">Logistic Regression</tspan></text>
        </g>
        <g
           id="g5488" />
        <g
           id="g5490" />
        <g
           id="g5492" />
        <g
           id="g5494" />
        <g
           id="g5496">
          <text
             transform="matrix(1,0,0,-1,287.42,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5500"><tspan
               x="0 4.4190001 7.4159999 12.42 17.424 19.926001 26.424 28.926001 31.427999 38.43 43.433998"
               y="0"
               sodipodi:role="line"
               id="tspan5498">Tree w/ GLM</tspan></text>
        </g>
        <g
           id="g5502" />
        <g
           id="g5504" />
        <g
           id="g5506" />
        <g
           id="g5508" />
        <g
           id="g5510">
          <text
             transform="matrix(1,0,0,-1,366.59,16.37)"
             style="font-variant:normal;font-weight:normal;font-size:9px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5514"><tspan
               x="0 7.0019999 13.005"
               y="0"
               sodipodi:role="line"
               id="tspan5512">GAM</tspan></text>
        </g>
        <g
           id="g5516" />
        <g
           id="g5518" />
        <g
           id="g5520" />
        <g
           id="g5522" />
        <g
           id="g5524" />
        <g
           id="g5526" />
        <g
           id="g5528"
           transform="translate(0,5.3456445)">
          <text
             transform="matrix(1,0,0,-1,64.88,489.19)"
             style="font-variant:normal;font-weight:normal;font-size:13px;font-family:Garamond;-inkscape-font-specification:Garamond;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
             id="text5532"><tspan
               x="0 8.6709995 18.056999 21.671 31.056999 38.285 43.004002 49.179001 56.407001 60.021 67.612999 71.226997 77.610001 84.837997 91.337997 94.952003 98.566002 107.952 115.18 118.794"
               y="0"
               sodipodi:role="line"
               id="tspan5530">PR Curve − Test Data</tspan></text>
        </g>
        <g
           id="g5534" />
        <g
           id="g5536" />
        <g
           id="g5538" />
        <g
           id="g5540" />
        <g
           id="g5542" />
        <g
           id="g5544" />
      </g>
    </g>
  </g>
</svg>



 
```r
format.auc.results <- function(x){
  return( data.table(   ROC.AUC = auc(x)[4][,1][1] , PR.AUC = auc(x)[4][,1][2]  )     )}

train.model.synthetic.glm<- evalmod(scores= predict(synth.model.glm , type = "response" ), labels  =    synth.model.glm$data$Class    )
train.model.synthetic.mixed<- evalmod(scores= predict(synth.model.three.deep , type = "response")  , labels  =  synth.model.three.deep$data$Class      )
train.model.synthetic.gam<- evalmod(scores= predict(synth.gam, type = "response" ), labels  =    synth.model.three.deep$data$Class    )

train.model.banking.glm<- evalmod(scores= predict(banking.mode.glm, type = "response" ), labels  =   banking.mode.four.deep$data$y     )
train.model.banking.mixed<- evalmod(scores= predict(banking.mode.four.deep , type = "response"), labels  =    banking.mode.four.deep$data$y    )
train.model.banking.gam<- evalmod(scores= predict(banking.gam , type = "response"), labels  =    banking.mode.four.deep$data$y    )

train.model.adult.glm<- evalmod(scores= predict(adult.mode.glm, type = "response" ), labels  =   adult.mode.four.deep$data$Class     )
train.model.adult.mixed<- evalmod(scores= predict(adult.mode.four.deep, type = "response" ), labels  =   adult.mode.four.deep$data$Class     )
train.model.adult.gam<- evalmod(scores= predict(adult.gam, type = "response" ), labels  = adult.mode.four.deep$data$Class       )

format.auc.results <- function(x){
  return( data.table(   ROC.AUC = auc(x)[4][,1][1] , PR.AUC = auc(x)[4][,1][2]  )     )}
training.results.synthetic <- lapply(X = list(train.model.synthetic.glm,train.model.synthetic.mixed,train.model.synthetic.gam),format.auc.results) %>% rbindlist
training.results.banking <- lapply(X = list(train.model.banking.glm,train.model.banking.mixed,train.model.banking.gam),format.auc.results) %>% rbindlist
training.results.adult <- lapply(X = list(train.model.adult.glm,train.model.adult.mixed,train.model.adult.gam),format.auc.results) %>% rbindlist
training.results.synthetic$Model <- c("Logistic","GLM + Tree","GAM") 
training.results.banking$Model <- c("Logistic","GLM + Tree","GAM")
training.results.adult$Model <- c("Logistic","GLM + Tree","GAM")
training.results.synthetic$Data <- "Synthetic"
training.results.banking$Data <- "Banking"
training.results.adult$Data <- "Adult"
training.list <-rbindlist( list(training.results.synthetic,training.results.banking,training.results.adult))
training.list$Label <- "In-Sample"

test.model.synthetic.glm<- evalmod(scores= predict(synth.model.glm,newdata = test.data$Synthetic, type = "response" ), labels  =    test.data$Synthetic$Class    )
test.model.synthetic.mixed<- evalmod(scores= predict(synth.model.three.deep , type = "response")  , labels  =  synth.model.three.deep$data$Class      )
test.model.synthetic.gam<- evalmod(scores= predict(synth.gam, type = "response" ), labels  =    synth.model.three.deep$data$Class    )
test.model.banking.glm<- evalmod(scores= predict(banking.mode.glm, type = "response" ), labels  =   banking.mode.four.deep$data$y     )
test.model.banking.mixed<- evalmod(scores= predict(banking.mode.four.deep , type = "response"), labels  =    banking.mode.four.deep$data$y    )
test.model.banking.gam<- evalmod(scores= predict(banking.gam , type = "response"), labels  =    banking.mode.four.deep$data$y    )
test.model.adult.glm<- evalmod(scores= predict(adult.mode.glm, type = "response" ), labels  =   adult.mode.four.deep$data$Class     )
test.model.adult.mixed<- evalmod(scores= predict(adult.mode.four.deep, type = "response" ), labels  =   adult.mode.four.deep$data$Class     )
test.model.adult.gam<- evalmod(scores= predict(adult.gam, type = "response" ), labels  = adult.mode.four.deep$data$Class       )
testing.results.synthetic <- lapply(X = list(test.model.synthetic.glm,test.model.synthetic.mixed,test.model.synthetic.gam),format.auc.results) %>% rbindlist
testing.results.banking <- lapply(X = list(test.model.banking.glm,test.model.banking.mixed,test.model.banking.gam),format.auc.results) %>% rbindlist
testing.results.adult <- lapply(X = list(test.model.adult.glm,test.model.adult.mixed,test.model.adult.gam),format.auc.results) %>% rbindlist
testing.results.synthetic$Model <- c("Logistic","GLM + Tree","GAM") 
testing.results.banking$Model <- c("Logistic","GLM + Tree","GAM")
testing.results.adult$Model <- c("Logistic","GLM + Tree","GAM")
testing.results.synthetic$Data <- "Synthetic"
testing.results.banking$Data <- "Banking"
testing.results.adult$Data <- "Adult"
testing.list <- rbindlist(list(testing.results.synthetic,testing.results.banking,testing.results.adult))
testing.list$Label <- "Out-of-Sample"
```


```r
results.list <- rbindlist(list(training.list,testing.list))
results.list <- results.list[,c(5,4,3,1,2)]
results.list <- melt(data = results.list,id.vars = c("Data","Model","Label"))
results.list <-dcast(data = results.list,formula = Data + Model ~ Label + variable,fun.aggregate = function(x){return(sum(x,na.rm=F))},value.var = "value")
reactable(data = results.list,groupBy = c("Data"),
            columns = list( `In-Sample_ROC.AUC` = colDef(name = "ROC.AUC",format =colFormat(digits = 3) ),
                            `In-Sample_PR.AUC` = colDef(name = "PR.AUC",format =colFormat(digits = 3)),
                            `Out-of-Sample_ROC.AUC` = colDef(name = "ROC.AUC",format =colFormat(digits = 3)),
                            `Out-of-Sample_PR.AUC` = colDef(name = "PR.AUC",format =colFormat(digits = 3)) ),
    columnGroups = list( colGroup( name = "In-Sample", columns = c("In-Sample_ROC.AUC","In-Sample_PR.AUC")    ),
                               colGroup( name = "Out-of-Sample", columns = c("Out-of-Sample_ROC.AUC","Out-of-Sample_PR.AUC"  ) )),defaultExpanded = T, defaultSortOrder = "desc")      
```
# Conclusion:
I compared the results from two different approaches to coping with nonlinear data. The first was used a decision tree model as a feature engineering step to create a set of one-hot encoders which are then used in a logistic regression model. The second method was a generalized additive model which incorporates the potential nonlinearity of the data into the model specification. This eliminates the assumption of a linear relationship between the dependent and independent variables. These two methodologies were compared on three data-sets to see if there was a consistent different in their performances. The tests show that GAM outperformed the GLM+Tree models in every instead. Both models outperformed simple logistic regression models.

The fact that GAMs outperformed the Tree+GLM models is significant since it also eliminates the feature engineering step and the ambiguity about the depth of the tree being used to generate the processed variables. 
