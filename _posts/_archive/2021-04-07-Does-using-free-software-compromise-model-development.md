---
layout: post
title:  "Does using free software compromise model development?"
date:   2021-04-13 11:34:12 -0400
tags: programming
---
A topic that gets raised when I speak about model development with business users is how free programming languages like R and Python compare to paid software like SAS or MATLAB. The discussion generally turns to compromises made by choosing free platforms for development. This is understandable: generally choosing the free option comes at a significant non-monetary cost later on.

In the realm of development though, picking a tool like SAS over free languages like R and Python implies a tradeoff on features instead of a compromise in quality. The tradeoff is based on one choice: Do you prefer to have the support and infrastructure delivered directly to you, or do you desire more flexibility and customization with your implementation? In this post, I'll summarize the most common concerns raised, explain them, and examine how true they are in practice.

My own background involves a mix of paid software packages and free programming languages. I've validated several statistical models employing SAS, and have conducted mathematical research using MATLAB.

## Cases made against free software
The main concerns about using free software are:

1. Paid programs/packages will be faster
2. Paid programs/packages are easier to implement with a GUI.
3. Paid programs/packages come with support.
4. Free programming languages aren't fully tested and packages can be unstable.

### Paid software will be faster

Conventionally, R and Python perform more slowly than platforms SAS at large scale data manipulation since SAS can work on data row-by-row while R and Python must load it into memory. For small-scale data, this actually works in R and Python's favor, but when data moves to several gigabytes SAS will be faster.

The change in computing power doesn’t become a concern until the user moves into the realm of big data. One of the great strengths of SAS and MATLAB is their ability to work with data that can't fit in memory. In fact, SAS can work with data row-by-row which allows it to quickly work on very large data sets without additional software extensions. MATLAB has a similar ability.

There are some alternatives for languages like R and Python. If the main concern is simply importing and transforming data SQL can be used. I personally use SQLite because of the lightweight installation. There is also the alternative of the [HDF5 format](https://pypi.org/project/h5df/) which creates a heavily compressed data-set that can be called into memory as needed. Importing GZIP files is also an alternative.

For cases where a model must be trained on a data-set that won't fit in memory, it's necessary to draw upon a service like [Spark](https://spark.apache.org/) or [H2O](https://www.h2o.ai). This is especially true when the model is fit on training data that won't fit into memory. These services do charge fees for their use, but they are generally much lower than buying a license. The downside is that the charges will be variable based on the amount of computing time you take among other things.

### Paid packages make development easier

When this is raised it's generally related to the use of GUIs like SAS e-miner or MATLAB's machine learning GUI. Both of these products are easy to use and can allow a developer to create a high-quality model with no code. They even help with exploratory data analysis and output performance metrics! 

R and Python do offer their own equivalents. [Orange](https://orangedatamining.com/) is based in Python and provides an attractive interface for exploring machine learning models but has the limitation of not allowing the Python code to be exported. [R Commander](https://en.wikipedia.org/wiki/R_Commander) provides a similar suite of tools that can output the code at the end of the process.

Before moving on from this point, I want to highlight one point on using a GUI for model development. The ability to rapidly develop a model in a GUI means that the developer isn't working directly with the data or code that ultimately composes the actual model. Unusual results or data issues can be easily overlooked by the program and may go undocumented by the developer. Additionally, the code being generated should be checked for quality but the automated process can output messy, badly organized code which can complicate tasks like audit, validation, and impact analysis. A model developer may create an overfit, overcomplicated model and then externalize its issues to a user who will have trouble understanding how the program is working.

### Paid packages have better support

When you're working on a deadline the support from a paid package can be a lifesaver and that is part of what you're paying for. Paid programs include excellent client support which can help users identify a solution quickly. Additionally, the software providers generally provide bespoke guidance on how to best solve a problem.

Free programs don’t have the same level of individual support and instead rely on an active support community. Most problems can be solved by checking messageboards like StackExchange or asking questions on mailing lists. That said, this doesn’t provide the same level of personalized support.
 
### Stability of numerical results

I’ve encountered this a few times. The first was at a conference where I was told that a pharmaceutical company was required to use SAS because it had been fully validated.

In theory, this concern makes sense, but I've never encountered this as a problem in practice. The most common statistical methods have been studied for decades and are well-posed problems IE: They have a unique solution that doesn’t vary based on the starting point. There isn’t a way to arrive at a “wrong” solution to logistic regression because of some hiccup in the software. My own experience here has been that whenever I’ve needed to replicate a result in MATLAB/SAS in another language, I’ve been able to without issue.

## Conclusion:

I hope this has helped provide some background on the tradeoffs that come with picking free software. The emphasis shouldn't be on whether one set of packages is universally superior, but what the needs of the user/organization are. The heuristic we can use is this: If the organization contains few coders, paid packages may be the best option. They provide very effective GUIs and great support. If you have more technical expertise, this may not be as useful since developers will have the background to implement from scratch and troubleshoot their own problems.



*Special thanks to my friend Aleda Leis who gave her thoughts on my assessment. She has a deeper experience with SAS than I do and getting her second opinion helped me fine-tune my views in this post. You can find her LinkedIn profile [here](https://www.linkedin.com/in/aledalthompson/).*



