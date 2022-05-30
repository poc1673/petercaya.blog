---
layout: post
title:  "Using a bank data to generate a call report application"
date:   2022-05-01  
categories: programming, R, python
---
|Header photo here|
Monitoring and accounting guidelines from the past few years have increased the depth of analysis that banks need to use on their loan portfolios. This includes the comprehensive credit loss (CECL) methodology which requires that banks project out the expected lifetime loan losses for their portfolios. Additionally, banks can use the information on their loan portfolios and compare it to their peers to determine if there are any definite trends in loan performance.

It turns out that the information necessary to track the performance of loans at individual banks is available publicly through the FFIEC website. Since this information was so available, and was already being used to develop loan loss forecasting solutions in the industry, I've decided to try my hand at developing a web application that organizes call report data related to loan losses, makes projections based on past macro-economic trends, and augments the FFIEC data with additional data.

At this time, I'v completed the first step of my project which was to:
1. Collect bank call report data,
2. Clean and organize the data and,
3. Create a dashboard in Shiny to aggregate default data.

I'm using this blog post to introduce and document my process but this post won't include fine details or extensive code. I will eventually load my project as-is to Github and share a link, but someone with motivation can replicate my steps using the resources in this post, Selenium, and R Shiny. I am in the process of reorganizing my project before moving forward with it and will update this post when I load it to Github.

# Creating the call report data-set
Our keystone data resource is the call report data available on the FFIEC website. For those unfamiliar, I've copied the definition of call reports from [Investopedia](https://www.investopedia.com/terms/c/callreport.asp) below:

> A call report is a regulatory report that must be filed by banks in the U.S. on a quarterly basis with the FDIC. A call report contains information about the bank's financial health, and by examining multiple call reports it can provide insight regarding the welfare of the U.S. banking system more broadly. 

## Accessing call report data

The reports can be reviewed and downloaded [here](https://cdr.ffiec.gov/public/ManageFacsimiles.aspx). I utilized the bulk call report downloader available [one this page](https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx). For this example, I selected the Single Period Bulk Call Report and downloaded it as a tab delimited file. It should be noted that you can download the call report history for each instution, but this leads to a large number of requests to the FFIEC website which makes the data scraping process error prone. 

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/CallReportPost%20-%20bulk_call_report%20picture.png)

I used the information above to create a Selenium program that would download the single period bulk call report for every date available.

We can see the download is a zipped file which contains two types of text files:
1. Bulk POR 
2. Call Schedule - A file containing the actual banking data. The tsv files are separates based on the code.

This is a lot of information to parse on its own; But we have one of these for each date! This leads to our next task:

## Organizing the data

Downloading the zip files leads to the following problems:
1. How to organize the results.
2. Providing names to the columns of each text file.
3. Mapping the bank ID to the Bank's actual name.

In my case, I simply used the date to organize the call reports by adding a column. The names of the columns *are* available, but are in the second row, so light  reformatting was required to label the columns.

**| PICTURE of data frame preformatting here|**

It's also necessary to map the RSSD to the bank names. I used the relevant POR file which provides details for each bank in the call report. This file contains the relevant bank information for each RSSD which could be joined to the call report information. Results were saved to a database.

## Subsetting variables

We've succesfully downloaded and labeled the data, but one issue remains: The database that was created can be accessed easily but is still 21GB! This is clearly too large to be deploy will cause issues with the Shiny app that I inded to use this for.

Thankfully, the fields that are provided by call reports aren't relevant to our application. I filtered the information from the database using regex so only fields relevant to the following loan types are selected:

* Multifamily
* Credit cards
* Commercial and industrial loans
* Agricultural loans
* Single family loans
 
From here we select call report codes relevant to the following categories:
- Total loans in category
- Loans that are 90 days past due
- Loans that are in nonaccrual 
- Charge-offs

# Creating the webapplication
 
I've built this application with Shiny due to the sheer simplicity. I considered trying to build it with Django - but Shiny's direct and easy to test framework made it more appealing to me. At this time, it has three primary tabs (along with an About tab) which are:

- Bank Review
- RSSD Lookup
- Data Download

## Bank review information

The heart of the application is the bank review tab which allows the user to select a bank's RSSD ID and compare the loan performance against peers and other banks in the state. The call report data fields are in dollars so I scaled the results by taking the relevant field (90 days past due, nonaccrual, and charge offs) as a percent of the relevant product type (multifamily loans, C&I, et cetera). 

The same aggregation procedure is used for the bank being reviewed and the state aggregate amounts. For the peer results, the loan amounts and the field are combined together as a group before the percentage value is created. This amounts to creating a balance weighted average of the loan performance of the peer group. One issue that there was in the original data is that there isn't a simple one-to-one mapping available in the call reports for the RSSD ID and the name of the actual bank. 

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/CallReportPost%20-%20bank%20review%20tab.PNG) 

To help the user identify the bank of interest as well as potential peers, I added an "RSSD Lookup" tab:

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/CallReportPost%20-%20bank%20lookup%20info.PNG)

A final touch I added was to allow the user to download the data from the app table. Eventually, I may expand this piece out to give the user more options, but at this time, it only provides the data used in the chart for the bank review tab.

# Next steps

The first step for me is to add in macro-economic data to the charts and analysis being used in the report. I think that being able to compare something like the percentage of days-paste due for a bank and its peers to the economic cycle would be helpful to a user.

I'd also like to add a predictive component based off of the peer and macroeconomic data. This will provide the user some "value add" for the results. 

Finally, I'd like to expand out the data offered and used by the app so that it provides more than just the product segments that it does now. The project is currently as a simple "pilot" app and there is a lot that can be done to expand its scope.






