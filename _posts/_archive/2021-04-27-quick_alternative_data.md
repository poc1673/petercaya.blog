---
title: "Defining and working with alternative data"
date: 2021-04-27 
---

The advent of the internet of things, improvements in deep learning, and the widespread use of software for tasks like webscraping mean a surge in the availability of previously unconventional data in the financial industry. In fact, it's already happening. This is often referred to as "alternative data". At this time, I haven't had the chance to actually work with alternative data, but it's clear that it will become more common and that it will require special scrutiny by analysts using or reviewing the data.

# Defining alternative data

The title of "alternative data" is ambiguous, so it's necessary to define terms. According to Alexander Denev and Saeed Amen's [book](https://www.amazon.com/dp/B08C4NSBWF/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1), alternative data has all or some of the following features:

-   Not commonly used by market participants

-   Costly to collect

-   Usually outside of financial markets

-   Has a short history

-   Is more difficult to use

Some examples of alternative data include:

-   Satellite images

-   Output from a webscraper

-   [Geolocation data](https://www.retaildive.com/ex/mobilecommercedaily/u-s-bank-marries-geolocation-with-fraud-prevention-for-visa-cardholders)

-   Automobile traffic data

The edge provided by alternative data wanes over time as more people become able to access it. If it becomes common enough, it may cease to be alternative data altogether.

# An example: Car counts to predict earnings

To me, the classic example of alternative data is the parking lot car-count data-set created by Tom and Alex Diamond. The two brothers obtained a set of pictures of retailer parking lots across the country and counted the number of cars to create a time series. You can find an Atlantic article reporting on this [here](https://www.theatlantic.com/magazine/archive/2019/05/stock-value-satellite-images-investing/586009/). Using this unconventional source of information reaped big benefits: When researchers relied on the data to decide which stock to buy the week before quarterly earnings, they were able to outpace benchmark strategies by 4.7%!

# Why it poses a compliance problem

Bad development data is a common causes of poor model performance and can come in a number of different forms ranging from noisy/inaccurate data, to biased or incomplete data. In the most simple scenario, the data quality can reduce accuracy in a trackable manner, but the structural issues with the data may go unobserved in the validation process and only become apparent when it is put into production.

# Handling alternative data in development

The difficulty of finding, producing, and maintaining alternative data means that quality control strategies becomes an even higher priority than before. Conventional data typically is in a neater, more widely understood format and its mainstream use means that flaws will be identified. However, alternative data involves more sophisticated steps which introduces ambiguity and error in the results. In the example given in the Atlantic article human error could be introduced simple miscounting. The Diamond brothers could have plausibly trained a neural network to count the cars, but then how is the performance of the neural network accounted for?

After the additional scrutiny on the collection methods is complete, the analyst must also consider more common problems such as data imputation. If these steps aren't taken, it's possible that the result could be a model built on noisy, spurious data.

Here, benchmarking the alternative data can be valuable to improve confidence in it. A good example of benchmarking alternative data is from the Billion Price Project paper available [here](https://www.aeaweb.org/articles?id=10.1257/jep.30.2.151). The authors of the paper sought to build a representation of how prices vary by scraping price data from the internet. This data was then compared to the CPI. We can see that here is a correspondence between the two which lends some credibility:

![Taken from Cavallo and Rigobon's 2016 paper](https://github.com/poc1673/poc1673.github.io/blob/main/rigobon%20figure.png?raw=true)

With this in mind, it's also important to note that it is considered alternative for a reason: It is either unusual, messy, or difficult to collect. These problems can be overcome, but raise new questions about the necessary compliance tasks. Thankfully, we don't have to reinvent the wheel and can lean on old methodologies for data control without having to reinvent the wheel. We only need to apply greater scrutiny.
