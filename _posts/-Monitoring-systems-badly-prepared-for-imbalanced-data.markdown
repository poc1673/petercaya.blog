<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1"><!-- Begin Jekyll SEO tag v2.7.1 -->
<title>Optimization Problems in Tensorflow </title>
<meta name="generator" content="Jekyll v3.8.7" />
<meta property="og:title" content="Optimization Problems in Tensorflow" />
<meta property="og:locale" content="en_US" />
<meta name="description" content="Write an awesome description for your new site here. You can edit this line in _config.yml. It will appear in your document head meta (for Google search results) and in your feed.xml site description." />
<meta property="og:description" content="Write an awesome description for your new site here. You can edit this line in _config.yml. It will appear in your document head meta (for Google search results) and in your feed.xml site description." />
<link rel="canonical" href="http://localhost:4000/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data" />
<meta property="og:url" content="http://localhost:4000/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data" />
<meta property="og:site_name" content="Risk Matters" />
<meta property="og:type" content="article" />
<meta property="article:published_time" content="2021-04-02T11:34:12-04:00" />
<meta name="twitter:card" content="summary" />
<meta property="twitter:title" content="Optimization Problems in Tensorflow" />
<script type="application/ld+json">
{"headline":"Optimization Problems in Tensorflow","dateModified":"2021-04-02T11:34:12-04:00","datePublished":"2021-04-02T11:34:12-04:00","@type":"BlogPosting","url":"http://localhost:4000/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data","mainEntityOfPage":{"@type":"WebPage","@id":"http://localhost:4000/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data"},"description":"Write an awesome description for your new site here. You can edit this line in _config.yml. It will appear in your document head meta (for Google search results) and in your feed.xml site description.","@context":"https://schema.org"}</script>
<!-- End Jekyll SEO tag -->
<link rel="stylesheet" href="/assets/main.css"><link type="application/atom+xml" rel="alternate" href="http://localhost:4000/feed.xml" title="Your awesome title" /></head>
<body><header class="site-header" role="banner">

  <div class="wrapper"><a class="site-title" rel="author" href="/">Your awesome title</a><nav class="site-nav">
        <input type="checkbox" id="nav-trigger" class="nav-trigger" />
        <label for="nav-trigger">
          <span class="menu-icon">
            <svg viewBox="0 0 18 15" width="18px" height="15px">
              <path d="M18,1.484c0,0.82-0.665,1.484-1.484,1.484H1.484C0.665,2.969,0,2.304,0,1.484l0,0C0,0.665,0.665,0,1.484,0 h15.032C17.335,0,18,0.665,18,1.484L18,1.484z M18,7.516C18,8.335,17.335,9,16.516,9H1.484C0.665,9,0,8.335,0,7.516l0,0 c0-0.82,0.665-1.484,1.484-1.484h15.032C17.335,6.031,18,6.696,18,7.516L18,7.516z M18,13.516C18,14.335,17.335,15,16.516,15H1.484 C0.665,15,0,14.335,0,13.516l0,0c0-0.82,0.665-1.483,1.484-1.483h15.032C17.335,12.031,18,12.695,18,13.516L18,13.516z"/>
            </svg>
          </span>
        </label>

        <div class="trigger"><a class="page-link" href="/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data">Optimization Problems in Tensorflow</a><a class="page-link" href="/about/">About</a></div>
      </nav></div>
</header>
<main class="page-content" aria-label="Content">
      <div class="wrapper">
        <article class="post h-entry" itemscope itemtype="http://schema.org/BlogPosting">

  <header class="post-header">
    <h1 class="post-title p-name" itemprop="name headline">Optimization Problems in Tensorflow</h1>
    <p class="post-meta">
      <time class="dt-published" datetime="2021-04-02T11:34:12-04:00" itemprop="datePublished">Apr 2, 2021
      </time></p>
  </header>

  <div class="post-content e-content" itemprop="articleBody">
    You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

<figure class="highlight"><pre><code class="language-ruby" data-lang="ruby"><span class="k">def</span> <span class="nf">print_hi</span><span class="p">(</span><span class="nb">name</span><span class="p">)</span>
  <span class="nb">puts</span> <span class="s2">"Hi, </span><span class="si">#{</span><span class="nb">name</span><span class="si">}</span><span class="s2">"</span>
<span class="k">end</span>
<span class="n">print_hi</span><span class="p">(</span><span class="s1">'Tom'</span><span class="p">)</span>
<span class="c1">#=&gt; prints 'Hi, Tom' to STDOUT.</span></code></pre></figure>

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

  </div><a class="u-url" href="/2021-05-01-Monitoring-systems-badly-prepared-for-imbalanced-data" hidden></a>
</article>

      </div>
    </main><footer class="site-footer h-card">
  <data class="u-url" href="/"></data>

  <div class="wrapper">

    <h2 class="footer-heading">Your awesome title</h2>

    <div class="footer-col-wrapper">
      <div class="footer-col footer-col-1">
        <ul class="contact-list">
          <li class="p-name">Your awesome title</li><li><a class="u-email" href="mailto:your-email@example.com">your-email@example.com</a></li></ul>
      </div>

      <div class="footer-col footer-col-2"><ul class="social-media-list"><li><a href="https://github.com/jekyll"><svg class="svg-icon"><use xlink:href="/assets/minima-social-icons.svg#github"></use></svg> <span class="username">jekyll</span></a></li><li><a href="https://www.twitter.com/jekyllrb"><svg class="svg-icon"><use xlink:href="/assets/minima-social-icons.svg#twitter"></use></svg> <span class="username">jekyllrb</span></a></li></ul>
</div>

      <div class="footer-col footer-col-3">
        <p>Write an awesome description for your new site here. You can edit this line in _config.yml. It will appear in your document head meta (for Google search results) and in your feed.xml site description.</p>
      </div>
    </div>

  </div>

</footer>
</body>

</html>
