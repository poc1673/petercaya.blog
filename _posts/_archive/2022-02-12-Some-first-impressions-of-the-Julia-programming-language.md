---
layout: post
title:  "Some first impressions of the Julia programming language"
date:   2022-02-12  
categories: programming, julia
---

![](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/julia_header.PNG)

I had some free time last Wednesday, so I began teaching myself the Julia programming language.

This framework has been at the back of my mind for a time now. I first came across it at a conference at Cornell University where some of the developers showcased it in comparison to R, Python and C++. The point they wanted to drive home is that Julia offered a way to use literature programming while maintaining the high performance of C and FORTRAN. Since then, it's repeatedly come up on my periphery including [Peter Norvig mentioning that he would prefer if Julia was the main language for AI](https://juliacomputing.com/blog/2020/12/newsletter-december/). It's remained a curiosity for me

In this post, I'll step through some facile observations I've had on Julia when coming from a background in R. This isn't supposed to be a full-scope guide. If you are looking for that, I recommend reading the [docs](https://docs.julialang.org/en/v1/) or [this](https://docs.julialang.org/en/v1/manual/noteworthy-differences/) handy comparison to other common data science languages.



# Package management is as easy as R

One of the first features that I ran into in Julia was the Pkg utility that comes with the language. This is Julia's equivalent to R's library/CRAN facility or Python's pip. To use it, simply type Pkg at the Julia command line, and then download the libraries that you need:


Whereas pip has given me problems in the past, I found Pkg to be as easy to use as R's library function. You simply type "]" to enter pkg:


![ ](https://raw.githubusercontent.com/poc1673/petercaya.com/main/_posts/blog_pkg_figure.png)

Granted, the ecosystem for Julia is still immature compared to R and Python, but the next point helps compensate for this problem...

# Julia Makes Calling Other Languages Simple

One item that was featured early on for me was the supposed relative ease that one could integrate R and Python code into Julia. Again, this is the case with Python and R, but the basic package to draw from these languages appears to be more straightfoward to use than the PyCall and RCall packages.

The encouragement I've seen for this means that someone using Julia might be able to easily draw from the extensive environments of both R and Python. That said, this presumes that the programmer already knows one or both of those languages. Julia also encourages drawing from a wide range of other languages (which you can find discussed here) but this is the extent of my investigation.


# Julia may be able to prevent 

Often, it's necessary to convert some code from a high level, more friendly language like Python and R to a faster language such as C++. This can be time consuming and requires additional expertise in additional programming languages which may be problematic if you are in a small team. 

This is where I believe Julia might be a boon to teams that want to avoid this bottleneck. It's especially valuable for solo programmers like myself who need to avoid additional implementation steps where possible. Julia was created with high speed performance in mind and the literature seems to corroborate that well-written Julia approachs C. To see more, you can review the micro-benchmark article here. 


# A note on multiple dispatch

Before beginning with Julia, I wasn't familiar with the concept of multiple dispatch but after doing some background reading on the subject, it's evident that it value is evident for both code clarity and efficiency. Multiple dispatch allows the programmer to define the same function with different behaviors based on input type. This is in contrast to cases I often see where a function uses an if-statement to deal with different input types. Like the first example below.

```
function adder_example(a,b)
    if typeof(a)==String
        return string(a,b)
    elseif (typeof(a) == Int64)|(typeof(a) == Float64)
        return a+b
    else
        return "NA"
    end 
end    
```
As a multiple dispatch implementation, this becomes the more terse code below:
```
adder_example(a::String ,b::String ) = string(a,b)
adder_example(a::Int64, b::Int64) = a+b
adder_example(a::Float64, b::Float64) = a+b 
```

This is more clear, but how is this different from overloading a function? The difference is when the function is evaluated: Multiple dispatch is evaluated at runtime - which provides a speed advantage over the compile time decision used in overloading. [This](https://medium.com/swlh/how-julia-uses-multiple-dispatch-to-beat-python-8fab888bb4d8#:~:text=To%20explain%20this%20speedup%2C%20Julia,millions%20of%20if%2Dstatement%20evaluations.) post provides a simple example where multiple dispatch Julia ran 50 times faster than the equivalent code written in Python.



# Flux.jl for machine learning

![](https://github.com/poc1673/petercaya.com/blob/main/_posts/flux_logo.png)

My experience with Julia during this past week has been experimenting some of examples given in [Neural Networks and Deep Learning by Charu Agarwal](https://link.springer.com/book/10.1007/978-3-319-94463-0). When I was planning to do this, I had a choice between:
* TensorFlow
* Jax
* Flux

Since the main purpose of the exercise was to get down to the level of the mathematical implementation of neural networks, I decided against TensorFlow since in my experience it aims to provide an efficient pipeline for implementation but whose code is less clear and less open to a new person modifying it.

Flux and Jax provide a closer result and I decided to use Flux to satisfy my curiosity. So far, I've found the framework appealing for my own use - specifically its autodifferentiation facility is very straightforward to use and may find its way into some of my work on optimization:



```
# The purpose of this script is to act as an example of solving a simple optimization problem in Julia using the Flux language.
using Flux
test_function(x) = x^3;
# Example of a single calculation of the gradient
# We can use two inputs into the gradient function: The function and an argument:
function df(x)
    gradient(test_function,x)[1]
end
# Train the model using gradient descent:
x = 5
cur_fn_value = test_function(x)
tol = 1e-6
while cur_fn_value > tol
    x = x - .05df(x)
    cur_fn_value = abs(test_function(x))    
    println(string("Current result is: ",cur_fn_value,"."))
end
println(join(["Solution is :",x,"."]))
# Close enough!
```


This is an extremely basic example but Flux provides facilities for building the layers of a neural network in the same manner as other libraries. I find it appealing because its flexibility (and that of the underlying Zygote.jl library) make it helpful for more conventional numerical analysis tasks.



# Conclusion

This concludes a short post on my ~10 days of experience with Julia. I'm finding the paradigm to have some definite advantages over what I'm used to seeing as a heavy R user. The speed that it has is appealing for more difficult data problems, and while there is a smaller ecosystem, Julia seems to make it easy to draw from R and Python and integrate it into Julia very easy.

I'll post from time to time on this, but in the future it's more likely you'll simply see the outputs of my work with Julia if I do it as opposed to give further thoughts on it. 


