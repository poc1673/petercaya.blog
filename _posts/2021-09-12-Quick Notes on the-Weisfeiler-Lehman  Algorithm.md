# Weisfeiller-Lehman Algorithm

While I was reading Thomas Kipf's [blog](https://tkipf.github.io/graph-convolutional-networks/) and his high level description of graph neural networks and the similarity he noted beetween them and the strategy used by the Weisfeiller-Lehman (WL) algorithm where he notes:

> A recent paper on a model called DeepWalk (Perozzi et al., KDD 2014) showed that they can learn a very similar embedding in a complicated unsupervised training procedure. How is it possible to get such an embedding more or less "for free" using our simple untrained GCN model?

>We can shed some light on this by interpreting the GCN model as a generalized, differentiable version of the well-known Weisfeiler-Lehman algorithm on graphs. Before reviewing this blog post, I was unfamiliar with the WL algorithm and its ability to find the canonical form of a graph so I decided to do a quick, simple implementation in Python from scratch.

## What is the context of Weisfeiller-Lehman's algorithm?
The WL algorithm was originally described in the [THE REDUCTION OF A GRAPH TO CANONICAL FORM AND THE
ALGEBRA WHICH APPEARS THEREIN](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf) by Boris Weisfeiler and Andrei Leman. Consider the case where we have some graph    <img src="https://latex.codecogs.com/gif.latex?G(V,E)" /> where <img src="https://latex.codecogs.com/gif.latex?V" /> are the vertices/nodes of the graph and <img src="https://latex.codecogs.com/gif.latex?E" /> represents the edges. Each node <img src="https://latex.codecogs.com/gif.latex?V_i" /> of the graph has a set of neighboring nodes. I've taken the example I've illustrated and notated below directly from Michael Bronstein's article [here](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49). 

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="8in"
   height="5in"
   viewBox="0 0 280.45834 182.26459"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="graph_example_1.svg">
  <title
     id="title10">graph_example_1</title>
  <defs
     id="defs2">
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1030"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1026"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1022"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1018"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect916"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect892"
       is_visible="true"
       lpeversion="1"
       pattern="M 0,5 C 0,2.24 2.24,0 5,0 7.76,0 10,2.24 10,5 10,7.76 7.76,10 5,10 2.24,10 0,7.76 0,5 Z"
       copytype="single_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       hide_knot="false"
       fuse_tolerance="0" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect882"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect878"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect874"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect870"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect866"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect862"
       is_visible="true"
       lpeversion="1" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.98994949"
     inkscape:cx="606.41549"
     inkscape:cy="355.99917"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     inkscape:document-rotation="0"
     showgrid="true"
     inkscape:window-width="2880"
     inkscape:window-height="1526"
     inkscape:window-x="2869"
     inkscape:window-y="-11"
     inkscape:window-maximized="1"
     units="in"
     inkscape:object-paths="true" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title>graph_example_1</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-27.999999)">
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12"
       r="15"
       cy="65.399567"
       cx="25.132292" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-8"
       cx="25.132292"
       cy="115.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-81"
       cx="25.132292"
       cy="170.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-2"
       cx="65.132294"
       cy="90.132294"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-1"
       cx="65.132294"
       cy="140.13229"
       r="15" />
    <text
       xml:space="preserve"
       style="font-size:11.2889px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.264583"
       x="9.4763451"
       y="35.342194"
       id="text924"><tspan
         sodipodi:role="line"
         id="tspan922"
         x="9.4763451"
         y="35.342194"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Garamond;-inkscape-font-specification:Garamond;stroke-width:0.264583">Graph 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:11.2889px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.264583"
       x="199.47635"
       y="35.342194"
       id="text924-4"><tspan
         sodipodi:role="line"
         id="tspan922-4"
         x="199.47635"
         y="35.342194"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Garamond;-inkscape-font-specification:Garamond;stroke-width:0.264583">Graph 2</tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,169.33333 Z"
       id="path976" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 38.364583,71.437499 C 52.916666,82.020832 52.916666,82.020832 52.916666,82.020832"
       id="path984" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,79.374999 V 100.54167"
       id="path986" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,156.10416 V 129.64583"
       id="path988" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 39.6875,170.65625 26.458332,-15.875"
       id="path990" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 64.822916,104.51042 v 21.16666"
       id="path992" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 64.822916,125.67708 V 104.51042"
       id="path898"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,170.65625 C 9.8617493,170.06016 9.1372967,169.4617 8.7270708,169.05263 8.316845,168.64357 8.2223497,168.42308 7.639781,167.28914 7.0572123,166.1552 5.9862798,164.10783 5.3245726,162.73767 c -0.6617072,-1.37015 -0.9136924,-2.06311 -1.0558682,-2.56723 -0.1421757,-0.50411 -0.1736747,-0.8191 -0.2995728,-1.66949 -0.1258981,-0.85039 -0.3463837,-2.2363 -0.1894062,-3.56007 0.1569774,-1.32377 0.6924437,-2.58369 0.9761861,-4.66189 0.2837424,-2.07821 0.3152405,-4.97603 0.3310037,-6.92889 0.015763,-1.95286 0.015763,-2.9608 0.078676,-4.39406 0.062913,-1.43325 0.1889049,-3.29163 0.2519842,-7.73276 0.063079,-4.44112 0.063079,-11.46518 0.094561,-17.54431 0.031482,-6.07913 0.094478,-11.21331 0.110268,-14.630807 0.01579,-3.417495 -0.015708,-5.118387 -0.1098928,-6.33087 C 5.4183272,91.50481 5.2608374,90.780357 5.1660964,90.166008 5.0713554,89.551658 5.0398574,89.04769 4.7881147,87.693411 4.536372,86.339131 4.0639021,84.134271 3.827338,82.606404 3.5907739,81.078537 3.5907739,80.228092 3.7481373,78.353802 3.9055006,76.479513 4.2204808,73.581695 4.3780975,72.038449 4.5357142,70.495204 4.5357142,70.306218 4.645771,70.053176 4.7558277,69.800134 4.976313,69.485155 5.1023128,69.233978 5.2283127,68.9828 5.2598108,68.793811 5.6226803,68.509003 5.9855498,68.224195 6.678506,67.846219 7.0888396,67.657484 c 0.4103337,-0.188735 0.5363259,-0.188735 0.724348,-0.267703 0.1880222,-0.07897 0.4400064,-0.236458 0.7085439,-0.330858 0.2685375,-0.0944 0.5520195,-0.125901 0.8977054,-0.267741 0.3456859,-0.14184 0.7551601,-0.393824 1.0080981,-0.519589 0.252938,-0.125765 0.347432,-0.125765 0.347431,-0.125765 -10e-7,0 -0.09449,0 -0.191633,0"
       id="path914"
       inkscape:path-effect="#path-effect916"
       inkscape:original-d="m 10.583333,170.65625 c -0.7218083,-0.59582 -1.4462626,-1.19428 -2.1733629,-1.79539 -0.091848,-0.21784 -0.1863421,-0.43833 -0.283482,-0.66146 -1.0682867,-2.04472 -2.1392192,-4.09209 -3.2127976,-6.14211 -0.2493383,-0.69031 -0.5013224,-1.38327 -0.7559524,-2.07887 -0.028852,-0.31233 -0.06035,-0.62732 -0.094494,-0.94494 -0.2178402,-1.38327 -0.4383264,-2.76918 -0.6614583,-4.15774 0.538112,-1.25727 1.0735784,-2.51719 1.6063986,-3.77976 0.034144,-2.89517 0.065642,-5.79299 0.094494,-8.69345 0.00265,-1.00529 0.00265,-2.01323 0,-3.02381 0.1286381,-1.85574 0.25463,-3.71412 0.3779764,-5.57515 0.00265,-7.02141 0.00265,-14.04547 0,-21.07217 0.065642,-5.13153 0.1286377,-10.26571 0.1889879,-15.402533 C 5.6407907,95.63062 5.6092925,93.929728 5.5751489,92.22619 5.4203044,91.50438 5.2628143,90.779927 5.1026784,90.052825 5.0738264,89.551504 5.0423283,89.047536 5.0081843,88.540922 4.53836,86.338705 4.0658899,84.133845 3.5907739,81.926339 c 0.00265,-0.847802 0.00265,-1.698247 0,-2.55134 0.3176259,-2.895171 0.632606,-5.792989 0.9449403,-8.693452 0.00265,-0.186343 0.00265,-0.375329 0,-0.566965 0.2231318,-0.312335 0.4436181,-0.627313 0.6614583,-0.94494 0.034144,-0.186341 0.065642,-0.37533 0.094494,-0.566965 0.6956023,-0.37533 1.3885585,-0.753306 2.0788691,-1.133928 0.1286377,0.0026 0.2546299,0.0026 0.377976,0 0.25463,-0.154845 0.5066141,-0.312335 0.7559524,-0.472469 0.2861281,-0.02885 0.5696101,-0.06035 0.8504465,-0.0945 0.4121201,-0.249338 0.8215945,-0.501322 1.2284225,-0.755952 0.09714,0.0026 0.191634,0.0026 0.283482,0 -0.09185,0.0026 -0.186342,0.0026 -0.283482,0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,42.333333 H 280.45833"
       id="path937" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0"
       r="15"
       cy="65.132294"
       cx="215.13229" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-4"
       r="15"
       cy="115.13229"
       cx="215.13229" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-6"
       r="15"
       cy="195.13229"
       cx="250.13229" />
    <ellipse
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264585;stroke-opacity:0.190871"
       id="path12-0-8"
       cy="150.13229"
       cx="250.13251"
       rx="15.000208"
       ry="14.999999" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-3"
       r="15"
       cy="150.13229"
       cx="180.13229" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 216.95833,78.052082 V 101.86458"
       id="path1008"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 187.85416,138.90625 17.19792,-14.55208"
       id="path1010"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 224.89583,124.35417 18.52083,14.55208"
       id="path1012"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 250.03125,164.04166 V 182.5625"
       id="path1014"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 264.58333,195.79166 c 1.51848,-3.30072 3.03751,-6.60263 4.22021,-9.5229 1.18269,-2.92028 2.02905,-5.45933 2.58598,-7.77533 0.55693,-2.31599 0.8242,-4.40962 1.00225,-5.5461 0.17806,-1.13649 0.26717,-1.3147 0.35627,-1.55935 0.0891,-0.24464 0.17819,-0.55646 0.31198,-1.89247 0.13379,-1.33601 0.31197,-3.69688 0.66824,-6.52563 0.35628,-2.82874 0.89081,-6.12506 1.15826,-9.15387 0.26744,-3.02881 0.26744,-5.79059 0.26744,-10.28963 0,-4.49904 0,-10.73532 0,-15.72435 0,-4.98904 0,-8.73081 0,-11.38123 0,-2.65043 0,-4.2095 -0.60061,-6.28057 -0.60061,-2.07106 -1.80332,-4.65467 -2.60518,-6.41419 -0.80187,-1.75952 -1.20277,-2.69496 -1.8706,-3.853204 -0.66783,-1.15824 -1.60328,-2.539147 -2.36075,-3.786342 -0.75747,-1.247194 -1.33655,-2.36081 -2.24949,-3.808595 -0.91293,-1.447786 -2.16019,-3.229584 -4.29822,-5.969165 -2.13804,-2.739582 -5.16709,-6.436807 -8.39631,-9.59981 -3.22921,-3.163002 -6.65917,-5.791155 -8.75266,-7.239431 -2.09349,-1.448276 -2.85075,-1.715544 -4.343,-2.272312 -1.49226,-0.556767 -3.71951,-1.403121 -5.64358,-2.067518 -1.92408,-0.664396 -3.54495,-1.14656 -5.16898,-1.629664"
       id="path1024"
       inkscape:path-effect="#path-effect1026"
       inkscape:original-d="m 264.58333,195.79166 c 1.52167,-3.29925 3.0407,-6.60116 4.55708,-9.90571 0.84899,-2.53641 1.69534,-5.07547 2.53907,-7.61718 0.2699,-2.09096 0.53718,-4.18458 0.80179,-6.28083 0.0917,-0.17553 0.18085,-0.35372 0.26729,-0.53454 0.0917,-0.30917 0.18081,-0.62098 0.26725,-0.93544 0.18085,-2.35824 0.35902,-4.71912 0.53454,-7.08265 0.53718,-3.29368 1.07172,-6.59 1.60361,-9.88897 0.003,-2.75914 0.003,-5.52092 0,-8.28535 0.003,-6.23365 0.003,-12.46993 0,-18.70887 0.003,-3.73913 0.003,-7.4809 0,-11.22532 0.003,-1.55643 0.003,-3.1155 0,-4.67722 -1.20004,-2.58096 -2.40276,-5.16456 -3.60812,-7.75081 -0.39825,-0.9328 -0.79917,-1.86824 -1.20272,-2.80633 -0.93278,-1.378249 -1.86825,-2.759144 -2.80633,-4.142682 -0.57644,-1.110977 -1.15551,-2.224598 -1.73725,-3.340867 -1.24461,-1.779151 -2.49187,-3.560948 -3.74177,-5.34539 -3.02641,-3.694584 -6.05546,-7.391813 -9.08717,-11.091685 -3.42731,-2.625506 -6.85727,-5.253657 -10.28987,-7.884451 -0.75462,-0.264626 -1.51188,-0.531895 -2.27179,-0.80181 -2.2246,-0.843708 -4.45185,-1.690063 -6.68174,-2.53906 -1.61823,-0.47952 -3.2391,-0.961684 -4.86262,-1.446496" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 166.6875,149.48958 c -0.36959,0.1045 -0.74093,0.2095 -1.97386,-3.56705 -1.23292,-3.77655 -3.32653,-11.43828 -4.37377,-17.87527 -1.04723,-6.43698 -1.04723,-11.64874 -0.023,-16.50486 1.02426,-4.85612 3.07332,-9.35516 6.54789,-15.279926 3.47458,-5.924765 8.37452,-13.274675 12.76285,-18.019265 4.38833,-4.744591 8.26374,-6.882748 10.8479,-8.085418 2.58416,-1.20267 3.87597,-1.469941 5.42852,-1.982836 1.55255,-0.512894 3.36656,-1.271281 5.17927,-2.029123"
       id="path1028"
       inkscape:path-effect="#path-effect1030"
       inkscape:original-d="m 166.6875,149.48958 c -0.3687,0.10764 -0.74004,0.21264 -1.11403,0.31499 -2.09097,-7.65908 -4.18458,-15.32081 -6.28083,-22.98518 0.003,-5.20911 0.003,-10.42087 0,-15.63527 2.05171,-4.49639 4.10077,-8.99543 6.14719,-13.49711 4.90259,-7.347265 9.80253,-14.697176 14.69983,-22.049735 3.87805,-2.13551 7.75346,-4.273666 11.62622,-6.414468 1.29445,-0.264623 2.58626,-0.531895 3.87541,-0.801809 1.81666,-0.755743 3.63067,-1.514131 5.44204,-2.275166" />
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="19.597795"
       y="58.136337"
       id="text1038-3"><tspan
         sodipodi:role="line"
         id="tspan1036-9"
         x="19.597795"
         y="58.136337"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="20.544983"
       y="110.01562"
       id="text1038-3-2"><tspan
         sodipodi:role="line"
         id="tspan1036-9-8"
         x="20.544983"
         y="110.01562"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="18.927937"
       y="164.38885"
       id="text1038-3-27"><tspan
         sodipodi:role="line"
         id="tspan1036-9-3"
         x="18.927937"
         y="164.38885"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="59.938354"
       y="85.013847"
       id="text1038-3-278"><tspan
         sodipodi:role="line"
         id="tspan1036-9-2"
         x="59.938354"
         y="85.013847"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="60.058556"
       y="133.534"
       id="text1038-3-4"><tspan
         sodipodi:role="line"
         id="tspan1036-9-89"
         x="60.058556"
         y="133.534"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="209.59779"
       y="58.136337"
       id="text1038-3-0"><tspan
         sodipodi:role="line"
         id="tspan1036-9-1"
         x="209.59779"
         y="58.136337"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="209.59779"
       y="108.13634"
       id="text1038-3-6"><tspan
         sodipodi:role="line"
         id="tspan1036-9-7"
         x="209.59779"
         y="108.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="174.59779"
       y="143.13634"
       id="text1038-3-66"><tspan
         sodipodi:role="line"
         id="tspan1036-9-4"
         x="174.59779"
         y="143.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="244.59779"
       y="143.13634"
       id="text1038-3-45"><tspan
         sodipodi:role="line"
         id="tspan1036-9-84"
         x="244.59779"
         y="143.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="244.59779"
       y="188.13634"
       id="text1038-3-3"><tspan
         sodipodi:role="line"
         id="tspan1036-9-5"
         x="244.59779"
         y="188.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.73022px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.134302"
       x="212.79546"
       y="65.701317"
       id="text1210-9"><tspan
         sodipodi:role="line"
         id="tspan1208-8"
         x="212.79546"
         y="65.701317"
         style="stroke-width:0.134302">A</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.79841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.1359"
       x="213.49359"
       y="117.86392"
       id="text1214-3"><tspan
         sodipodi:role="line"
         id="tspan1212-7"
         x="213.49359"
         y="117.86392"
         style="stroke-width:0.1359">B</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="177.52696"
       y="150.80841"
       id="text1218-0"><tspan
         sodipodi:role="line"
         id="tspan1216-1"
         x="177.52696"
         y="150.80841"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.96357px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139771"
       x="247.55394"
       y="152.72256"
       id="text1222-5"><tspan
         sodipodi:role="line"
         id="tspan1220-4"
         x="247.55394"
         y="152.72256"
         style="stroke-width:0.139771">D</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:6.73684px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.157894"
       x="248.05713"
       y="196.66643"
       id="text1226-8"><tspan
         sodipodi:role="line"
         id="tspan1224-5"
         x="248.05713"
         y="196.66643"
         style="stroke-width:0.157894">E</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="206.33916"
       y="73.879242"
       id="text1350"><tspan
         sodipodi:role="line"
         id="tspan1348"
         x="206.33916"
         y="73.879242"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="208.03545"
       y="124.80482"
       id="text1350-2"><tspan
         sodipodi:role="line"
         id="tspan1348-2"
         x="208.03545"
         y="124.80482"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="173.59436"
       y="158.28972"
       id="text1354-2"><tspan
         sodipodi:role="line"
         id="tspan1352-5"
         x="173.59436"
         y="158.28972"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="243.70894"
       y="159.61264"
       id="text1354-2-5"><tspan
         sodipodi:role="line"
         id="tspan1352-5-3"
         x="243.70894"
         y="159.61264"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="243.96278"
       y="204.3111"
       id="text1354-2-2"><tspan
         sodipodi:role="line"
         id="tspan1352-5-5"
         x="243.96278"
         y="204.3111"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.73022px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.134302"
       x="22.933922"
       y="66.205948"
       id="text1210-9-4"><tspan
         sodipodi:role="line"
         id="tspan1208-8-0"
         x="22.933922"
         y="66.205948"
         style="stroke-width:0.134302">A</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.79841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.1359"
       x="24.467724"
       y="117.86392"
       id="text1214-3-0"><tspan
         sodipodi:role="line"
         id="tspan1212-7-9"
         x="24.467724"
         y="117.86392"
         style="stroke-width:0.1359">B</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="241.02696"
       y="169.32924"
       id="text1218-0-8"><tspan
         sodipodi:role="line"
         id="tspan1216-1-8"
         x="241.02696"
         y="169.32924"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.96357px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139771"
       x="62.986919"
       y="92.509705"
       id="text1222-5-5"><tspan
         sodipodi:role="line"
         id="tspan1220-4-2"
         x="62.986919"
         y="92.509705"
         style="stroke-width:0.139771">D</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="22.466202"
       y="172.88365"
       id="text1218-0-5"><tspan
         sodipodi:role="line"
         id="tspan1216-1-2"
         x="22.466202"
         y="172.88365"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:6.73684px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.157894"
       x="63.249706"
       y="142.65382"
       id="text1226-8-0"><tspan
         sodipodi:role="line"
         id="tspan1224-5-7"
         x="63.249706"
         y="142.65382"
         style="stroke-width:0.157894">E</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.233006"
       y="75.055092"
       id="text1350-6"><tspan
         sodipodi:role="line"
         id="tspan1348-3"
         x="17.233006"
         y="75.055092"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.36664"
       y="125.32593"
       id="text1350-1"><tspan
         sodipodi:role="line"
         id="tspan1348-7"
         x="17.36664"
         y="125.32593"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="17.231424"
       y="181.83562"
       id="text1354-2-8"><tspan
         sodipodi:role="line"
         id="tspan1352-5-4"
         x="17.231424"
         y="181.83562"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.930302"
       y="150.08495"
       id="text1354-2-50"><tspan
         sodipodi:role="line"
         id="tspan1352-5-7"
         x="59.930302"
         y="150.08495"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.262127"
       y="100.05452"
       id="text1354-2-3"><tspan
         sodipodi:role="line"
         id="tspan1352-5-47"
         x="59.262127"
         y="100.05452"
         style="stroke-width:0.140125">{1,1}</tspan></text>
  </g>
</svg>


We have a few terms to define in the above figure:

* <img src="https://latex.codecogs.com/gif.latex?L" /> is the current label for each node <img src="https://latex.codecogs.com/gif.latex?i" />. In the two graphs above, <img src="https://latex.codecogs.com/gif.latex?L_i" /> is displayed inside each circle. All nodes starte off as being labeled as 1.
* Each node has a multiset (a list of elements that doesn't take order into account) of associated labels corresponding to the <img src="https://latex.codecogs.com/gif.latex?L_i" /> value of the connected nodes. For instance, node A of the above graph has the multiset of {1,1,1} because <img src="https://latex.codecogs.com/gif.latex?L_2 = L_3 = L_5 =1" />. For instance, if one has the multiset {1,2}, it's treated as equivalent to {2,1}.

We can see that the separate graph on the right has the same number of nodes and they each start with the same labeling. However, it isn't clear if there is some equivalence between the graphon the right and the graph on the left. The purpose of the WL algorithm is to help determine this equivalence.

The WL algorithm works by defining a new label for each node based on the labels of its neighbor. There is a good writeup in [David Bieber's blog](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/) and elsewhere, but here I wanted to concentrate on explaining the steps at a simple level:

First, the multiset of labels associated with each node is taken. In the chart above, we can see that both graphs have two kinds of multi-sets: {1,1,1} and {1,1}. 
2. Create a new set of labels <img src="https://latex.codecogs.com/gif.latex?C_i" /> which will go on to replace <img src="https://latex.codecogs.com/gif.latex?L_i" />. In this case, there will be two.
3. Note all nodes containing the {1,1,1} multiset; change the label on these nodes to <img src="https://latex.codecogs.com/gif.latex?L_i = 1" />. Afterwards, find all nodes where the multiset is {1,1} and change the labeling on these nodes so that <img src="https://latex.codecogs.com/gif.latex?L_i = 2" /> . 

This step is performed on both graphs and results in the next iteration which we see below:

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="8in"
   height="5in"
   viewBox="0 0 280.45834 182.26459"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="graph_example_1.svg">
  <title
     id="title10">graph_example_1</title>
  <defs
     id="defs2">
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1030"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1026"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1022"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1018"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect916"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect892"
       is_visible="true"
       lpeversion="1"
       pattern="M 0,5 C 0,2.24 2.24,0 5,0 7.76,0 10,2.24 10,5 10,7.76 7.76,10 5,10 2.24,10 0,7.76 0,5 Z"
       copytype="single_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       hide_knot="false"
       fuse_tolerance="0" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect882"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect878"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect874"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect870"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect866"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect862"
       is_visible="true"
       lpeversion="1" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.98994949"
     inkscape:cx="606.41549"
     inkscape:cy="355.99917"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     inkscape:document-rotation="0"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="986"
     inkscape:window-x="-11"
     inkscape:window-y="-11"
     inkscape:window-maximized="1"
     units="in"
     inkscape:object-paths="true" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title>graph_example_1</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-27.999999)">
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12"
       r="15"
       cy="65.399567"
       cx="25.132292" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-8"
       cx="25.132292"
       cy="115.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-81"
       cx="25.132292"
       cy="170.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-2"
       cx="65.132294"
       cy="90.132294"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-1"
       cx="65.132294"
       cy="140.13229"
       r="15" />
    <text
       xml:space="preserve"
       style="font-size:11.2889px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.264583"
       x="9.4763451"
       y="35.342194"
       id="text924"><tspan
         sodipodi:role="line"
         id="tspan922"
         x="9.4763451"
         y="35.342194"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Garamond;-inkscape-font-specification:Garamond;stroke-width:0.264583">Graph 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:11.2889px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.264583"
       x="199.47635"
       y="35.342194"
       id="text924-4"><tspan
         sodipodi:role="line"
         id="tspan922-4"
         x="199.47635"
         y="35.342194"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Garamond;-inkscape-font-specification:Garamond;stroke-width:0.264583">Graph 2</tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,169.33333 Z"
       id="path976" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 38.364583,71.437499 C 52.916666,82.020832 52.916666,82.020832 52.916666,82.020832"
       id="path984" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,79.374999 V 100.54167"
       id="path986" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,156.10416 V 129.64583"
       id="path988" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 39.6875,170.65625 26.458332,-15.875"
       id="path990" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 64.822916,104.51042 v 21.16666"
       id="path992" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 64.822916,125.67708 V 104.51042"
       id="path898"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,170.65625 C 9.8617493,170.06016 9.1372967,169.4617 8.7270708,169.05263 8.316845,168.64357 8.2223497,168.42308 7.639781,167.28914 7.0572123,166.1552 5.9862798,164.10783 5.3245726,162.73767 c -0.6617072,-1.37015 -0.9136924,-2.06311 -1.0558682,-2.56723 -0.1421757,-0.50411 -0.1736747,-0.8191 -0.2995728,-1.66949 -0.1258981,-0.85039 -0.3463837,-2.2363 -0.1894062,-3.56007 0.1569774,-1.32377 0.6924437,-2.58369 0.9761861,-4.66189 0.2837424,-2.07821 0.3152405,-4.97603 0.3310037,-6.92889 0.015763,-1.95286 0.015763,-2.9608 0.078676,-4.39406 0.062913,-1.43325 0.1889049,-3.29163 0.2519842,-7.73276 0.063079,-4.44112 0.063079,-11.46518 0.094561,-17.54431 0.031482,-6.07913 0.094478,-11.21331 0.110268,-14.630807 0.01579,-3.417495 -0.015708,-5.118387 -0.1098928,-6.33087 C 5.4183272,91.50481 5.2608374,90.780357 5.1660964,90.166008 5.0713554,89.551658 5.0398574,89.04769 4.7881147,87.693411 4.536372,86.339131 4.0639021,84.134271 3.827338,82.606404 3.5907739,81.078537 3.5907739,80.228092 3.7481373,78.353802 3.9055006,76.479513 4.2204808,73.581695 4.3780975,72.038449 4.5357142,70.495204 4.5357142,70.306218 4.645771,70.053176 4.7558277,69.800134 4.976313,69.485155 5.1023128,69.233978 5.2283127,68.9828 5.2598108,68.793811 5.6226803,68.509003 5.9855498,68.224195 6.678506,67.846219 7.0888396,67.657484 c 0.4103337,-0.188735 0.5363259,-0.188735 0.724348,-0.267703 0.1880222,-0.07897 0.4400064,-0.236458 0.7085439,-0.330858 0.2685375,-0.0944 0.5520195,-0.125901 0.8977054,-0.267741 0.3456859,-0.14184 0.7551601,-0.393824 1.0080981,-0.519589 0.252938,-0.125765 0.347432,-0.125765 0.347431,-0.125765 -10e-7,0 -0.09449,0 -0.191633,0"
       id="path914"
       inkscape:path-effect="#path-effect916"
       inkscape:original-d="m 10.583333,170.65625 c -0.7218083,-0.59582 -1.4462626,-1.19428 -2.1733629,-1.79539 -0.091848,-0.21784 -0.1863421,-0.43833 -0.283482,-0.66146 -1.0682867,-2.04472 -2.1392192,-4.09209 -3.2127976,-6.14211 -0.2493383,-0.69031 -0.5013224,-1.38327 -0.7559524,-2.07887 -0.028852,-0.31233 -0.06035,-0.62732 -0.094494,-0.94494 -0.2178402,-1.38327 -0.4383264,-2.76918 -0.6614583,-4.15774 0.538112,-1.25727 1.0735784,-2.51719 1.6063986,-3.77976 0.034144,-2.89517 0.065642,-5.79299 0.094494,-8.69345 0.00265,-1.00529 0.00265,-2.01323 0,-3.02381 0.1286381,-1.85574 0.25463,-3.71412 0.3779764,-5.57515 0.00265,-7.02141 0.00265,-14.04547 0,-21.07217 0.065642,-5.13153 0.1286377,-10.26571 0.1889879,-15.402533 C 5.6407907,95.63062 5.6092925,93.929728 5.5751489,92.22619 5.4203044,91.50438 5.2628143,90.779927 5.1026784,90.052825 5.0738264,89.551504 5.0423283,89.047536 5.0081843,88.540922 4.53836,86.338705 4.0658899,84.133845 3.5907739,81.926339 c 0.00265,-0.847802 0.00265,-1.698247 0,-2.55134 0.3176259,-2.895171 0.632606,-5.792989 0.9449403,-8.693452 0.00265,-0.186343 0.00265,-0.375329 0,-0.566965 0.2231318,-0.312335 0.4436181,-0.627313 0.6614583,-0.94494 0.034144,-0.186341 0.065642,-0.37533 0.094494,-0.566965 0.6956023,-0.37533 1.3885585,-0.753306 2.0788691,-1.133928 0.1286377,0.0026 0.2546299,0.0026 0.377976,0 0.25463,-0.154845 0.5066141,-0.312335 0.7559524,-0.472469 0.2861281,-0.02885 0.5696101,-0.06035 0.8504465,-0.0945 0.4121201,-0.249338 0.8215945,-0.501322 1.2284225,-0.755952 0.09714,0.0026 0.191634,0.0026 0.283482,0 -0.09185,0.0026 -0.186342,0.0026 -0.283482,0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,42.333333 H 280.45833"
       id="path937" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0"
       r="15"
       cy="65.132294"
       cx="215.13229" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-4"
       r="15"
       cy="115.13229"
       cx="215.13229" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-6"
       r="15"
       cy="195.13229"
       cx="250.13229" />
    <ellipse
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264585;stroke-opacity:0.190871"
       id="path12-0-8"
       cy="150.13229"
       cx="250.13251"
       rx="15.000208"
       ry="14.999999" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#ff3b00;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-0-3"
       r="15"
       cy="150.13229"
       cx="180.13229" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 216.95833,78.052082 V 101.86458"
       id="path1008"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 187.85416,138.90625 17.19792,-14.55208"
       id="path1010"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 224.89583,124.35417 18.52083,14.55208"
       id="path1012"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 250.03125,164.04166 V 182.5625"
       id="path1014"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 264.58333,195.79166 c 1.51848,-3.30072 3.03751,-6.60263 4.22021,-9.5229 1.18269,-2.92028 2.02905,-5.45933 2.58598,-7.77533 0.55693,-2.31599 0.8242,-4.40962 1.00225,-5.5461 0.17806,-1.13649 0.26717,-1.3147 0.35627,-1.55935 0.0891,-0.24464 0.17819,-0.55646 0.31198,-1.89247 0.13379,-1.33601 0.31197,-3.69688 0.66824,-6.52563 0.35628,-2.82874 0.89081,-6.12506 1.15826,-9.15387 0.26744,-3.02881 0.26744,-5.79059 0.26744,-10.28963 0,-4.49904 0,-10.73532 0,-15.72435 0,-4.98904 0,-8.73081 0,-11.38123 0,-2.65043 0,-4.2095 -0.60061,-6.28057 -0.60061,-2.07106 -1.80332,-4.65467 -2.60518,-6.41419 -0.80187,-1.75952 -1.20277,-2.69496 -1.8706,-3.853204 -0.66783,-1.15824 -1.60328,-2.539147 -2.36075,-3.786342 -0.75747,-1.247194 -1.33655,-2.36081 -2.24949,-3.808595 -0.91293,-1.447786 -2.16019,-3.229584 -4.29822,-5.969165 -2.13804,-2.739582 -5.16709,-6.436807 -8.39631,-9.59981 -3.22921,-3.163002 -6.65917,-5.791155 -8.75266,-7.239431 -2.09349,-1.448276 -2.85075,-1.715544 -4.343,-2.272312 -1.49226,-0.556767 -3.71951,-1.403121 -5.64358,-2.067518 -1.92408,-0.664396 -3.54495,-1.14656 -5.16898,-1.629664"
       id="path1024"
       inkscape:path-effect="#path-effect1026"
       inkscape:original-d="m 264.58333,195.79166 c 1.52167,-3.29925 3.0407,-6.60116 4.55708,-9.90571 0.84899,-2.53641 1.69534,-5.07547 2.53907,-7.61718 0.2699,-2.09096 0.53718,-4.18458 0.80179,-6.28083 0.0917,-0.17553 0.18085,-0.35372 0.26729,-0.53454 0.0917,-0.30917 0.18081,-0.62098 0.26725,-0.93544 0.18085,-2.35824 0.35902,-4.71912 0.53454,-7.08265 0.53718,-3.29368 1.07172,-6.59 1.60361,-9.88897 0.003,-2.75914 0.003,-5.52092 0,-8.28535 0.003,-6.23365 0.003,-12.46993 0,-18.70887 0.003,-3.73913 0.003,-7.4809 0,-11.22532 0.003,-1.55643 0.003,-3.1155 0,-4.67722 -1.20004,-2.58096 -2.40276,-5.16456 -3.60812,-7.75081 -0.39825,-0.9328 -0.79917,-1.86824 -1.20272,-2.80633 -0.93278,-1.378249 -1.86825,-2.759144 -2.80633,-4.142682 -0.57644,-1.110977 -1.15551,-2.224598 -1.73725,-3.340867 -1.24461,-1.779151 -2.49187,-3.560948 -3.74177,-5.34539 -3.02641,-3.694584 -6.05546,-7.391813 -9.08717,-11.091685 -3.42731,-2.625506 -6.85727,-5.253657 -10.28987,-7.884451 -0.75462,-0.264626 -1.51188,-0.531895 -2.27179,-0.80181 -2.2246,-0.843708 -4.45185,-1.690063 -6.68174,-2.53906 -1.61823,-0.47952 -3.2391,-0.961684 -4.86262,-1.446496" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 166.6875,149.48958 c -0.36959,0.1045 -0.74093,0.2095 -1.97386,-3.56705 -1.23292,-3.77655 -3.32653,-11.43828 -4.37377,-17.87527 -1.04723,-6.43698 -1.04723,-11.64874 -0.023,-16.50486 1.02426,-4.85612 3.07332,-9.35516 6.54789,-15.279926 3.47458,-5.924765 8.37452,-13.274675 12.76285,-18.019265 4.38833,-4.744591 8.26374,-6.882748 10.8479,-8.085418 2.58416,-1.20267 3.87597,-1.469941 5.42852,-1.982836 1.55255,-0.512894 3.36656,-1.271281 5.17927,-2.029123"
       id="path1028"
       inkscape:path-effect="#path-effect1030"
       inkscape:original-d="m 166.6875,149.48958 c -0.3687,0.10764 -0.74004,0.21264 -1.11403,0.31499 -2.09097,-7.65908 -4.18458,-15.32081 -6.28083,-22.98518 0.003,-5.20911 0.003,-10.42087 0,-15.63527 2.05171,-4.49639 4.10077,-8.99543 6.14719,-13.49711 4.90259,-7.347265 9.80253,-14.697176 14.69983,-22.049735 3.87805,-2.13551 7.75346,-4.273666 11.62622,-6.414468 1.29445,-0.264623 2.58626,-0.531895 3.87541,-0.801809 1.81666,-0.755743 3.63067,-1.514131 5.44204,-2.275166" />
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="19.597795"
       y="58.136337"
       id="text1038-3"><tspan
         sodipodi:role="line"
         id="tspan1036-9"
         x="19.597795"
         y="58.136337"
         style="stroke-width:0.10269">L = 2</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="20.544983"
       y="110.01562"
       id="text1038-3-2"><tspan
         sodipodi:role="line"
         id="tspan1036-9-8"
         x="20.544983"
         y="110.01562"
         style="stroke-width:0.10269">L = 2</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="18.927937"
       y="164.38885"
       id="text1038-3-27"><tspan
         sodipodi:role="line"
         id="tspan1036-9-3"
         x="18.927937"
         y="164.38885"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="59.938354"
       y="85.013847"
       id="text1038-3-278"><tspan
         sodipodi:role="line"
         id="tspan1036-9-2"
         x="59.938354"
         y="85.013847"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="60.058556"
       y="133.534"
       id="text1038-3-4"><tspan
         sodipodi:role="line"
         id="tspan1036-9-89"
         x="60.058556"
         y="133.534"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="209.59779"
       y="58.136337"
       id="text1038-3-0"><tspan
         sodipodi:role="line"
         id="tspan1036-9-1"
         x="209.59779"
         y="58.136337"
         style="stroke-width:0.10269">L = 2</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="209.59779"
       y="108.13634"
       id="text1038-3-6"><tspan
         sodipodi:role="line"
         id="tspan1036-9-7"
         x="209.59779"
         y="108.13634"
         style="stroke-width:0.10269">L = 2</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="174.59779"
       y="143.13634"
       id="text1038-3-66"><tspan
         sodipodi:role="line"
         id="tspan1036-9-4"
         x="174.59779"
         y="143.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="244.59779"
       y="143.13634"
       id="text1038-3-45"><tspan
         sodipodi:role="line"
         id="tspan1036-9-84"
         x="244.59779"
         y="143.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="244.59779"
       y="188.13634"
       id="text1038-3-3"><tspan
         sodipodi:role="line"
         id="tspan1036-9-5"
         x="244.59779"
         y="188.13634"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.73022px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.134302"
       x="212.79546"
       y="65.701317"
       id="text1210-9"><tspan
         sodipodi:role="line"
         id="tspan1208-8"
         x="212.79546"
         y="65.701317"
         style="stroke-width:0.134302">A</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.79841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.1359"
       x="213.49359"
       y="117.86392"
       id="text1214-3"><tspan
         sodipodi:role="line"
         id="tspan1212-7"
         x="213.49359"
         y="117.86392"
         style="stroke-width:0.1359">B</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="177.52696"
       y="150.80841"
       id="text1218-0"><tspan
         sodipodi:role="line"
         id="tspan1216-1"
         x="177.52696"
         y="150.80841"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.96357px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139771"
       x="247.55394"
       y="152.72256"
       id="text1222-5"><tspan
         sodipodi:role="line"
         id="tspan1220-4"
         x="247.55394"
         y="152.72256"
         style="stroke-width:0.139771">D</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:6.73684px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.157894"
       x="248.05713"
       y="196.66643"
       id="text1226-8"><tspan
         sodipodi:role="line"
         id="tspan1224-5"
         x="248.05713"
         y="196.66643"
         style="stroke-width:0.157894">E</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="206.33916"
       y="73.879242"
       id="text1350"><tspan
         sodipodi:role="line"
         id="tspan1348"
         x="206.33916"
         y="73.879242"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="208.03545"
       y="124.80482"
       id="text1350-2"><tspan
         sodipodi:role="line"
         id="tspan1348-2"
         x="208.03545"
         y="124.80482"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="173.59436"
       y="158.28972"
       id="text1354-2"><tspan
         sodipodi:role="line"
         id="tspan1352-5"
         x="173.59436"
         y="158.28972"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="243.70894"
       y="159.61264"
       id="text1354-2-5"><tspan
         sodipodi:role="line"
         id="tspan1352-5-3"
         x="243.70894"
         y="159.61264"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="243.96278"
       y="204.3111"
       id="text1354-2-2"><tspan
         sodipodi:role="line"
         id="tspan1352-5-5"
         x="243.96278"
         y="204.3111"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.73022px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.134302"
       x="22.933922"
       y="66.205948"
       id="text1210-9-4"><tspan
         sodipodi:role="line"
         id="tspan1208-8-0"
         x="22.933922"
         y="66.205948"
         style="stroke-width:0.134302">A</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.79841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.1359"
       x="24.467724"
       y="117.86392"
       id="text1214-3-0"><tspan
         sodipodi:role="line"
         id="tspan1212-7-9"
         x="24.467724"
         y="117.86392"
         style="stroke-width:0.1359">B</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="241.02696"
       y="169.32924"
       id="text1218-0-8"><tspan
         sodipodi:role="line"
         id="tspan1216-1-8"
         x="241.02696"
         y="169.32924"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.96357px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139771"
       x="62.986919"
       y="92.509705"
       id="text1222-5-5"><tspan
         sodipodi:role="line"
         id="tspan1220-4-2"
         x="62.986919"
         y="92.509705"
         style="stroke-width:0.139771">D</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="22.466202"
       y="172.88365"
       id="text1218-0-5"><tspan
         sodipodi:role="line"
         id="tspan1216-1-2"
         x="22.466202"
         y="172.88365"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:6.73684px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.157894"
       x="63.249706"
       y="142.65382"
       id="text1226-8-0"><tspan
         sodipodi:role="line"
         id="tspan1224-5-7"
         x="63.249706"
         y="142.65382"
         style="stroke-width:0.157894">E</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.233006"
       y="75.055092"
       id="text1350-6"><tspan
         sodipodi:role="line"
         id="tspan1348-3"
         x="17.233006"
         y="75.055092"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.36664"
       y="125.32593"
       id="text1350-1"><tspan
         sodipodi:role="line"
         id="tspan1348-7"
         x="17.36664"
         y="125.32593"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="17.231424"
       y="181.83562"
       id="text1354-2-8"><tspan
         sodipodi:role="line"
         id="tspan1352-5-4"
         x="17.231424"
         y="181.83562"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.930302"
       y="150.08495"
       id="text1354-2-50"><tspan
         sodipodi:role="line"
         id="tspan1352-5-7"
         x="59.930302"
         y="150.08495"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.262127"
       y="100.05452"
       id="text1354-2-3"><tspan
         sodipodi:role="line"
         id="tspan1352-5-47"
         x="59.262127"
         y="100.05452"
         style="stroke-width:0.140125">{1,1}</tspan></text>
  </g>
</svg>

These steps are repeated until the labels on both graphs stop changing. Once that is accomplished, the number and typers of labels can be compared between the two of them to determine whether the graphs are isomorphic or not. For a complete walkthrough of the iterations for this example, please see the original [post here](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/).  We eventually arrive at the two graphs below:

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="3.4023879in"
   height="6.1915193in"
   viewBox="0 0 86.420651 157.26459"
   version="1.1"
   id="svg8"
   inkscape:version="1.0.2-2 (e86c870879, 2021-01-15)"
   sodipodi:docname="graph_example_3.svg">
  <title
     id="title10">graph_example_1</title>
  <defs
     id="defs2">
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1030"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1026"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1022"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect1018"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect916"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect892"
       is_visible="true"
       lpeversion="1"
       pattern="M 0,5 C 0,2.24 2.24,0 5,0 7.76,0 10,2.24 10,5 10,7.76 7.76,10 5,10 2.24,10 0,7.76 0,5 Z"
       copytype="single_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       hide_knot="false"
       fuse_tolerance="0" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect882"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect878"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect874"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect870"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="bspline"
       id="path-effect866"
       is_visible="true"
       lpeversion="1"
       weight="33.333333"
       steps="2"
       helper_size="0"
       apply_no_weight="true"
       apply_with_weight="true"
       only_selected="false" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect862"
       is_visible="true"
       lpeversion="1" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4"
     inkscape:cx="307.82637"
     inkscape:cy="301.27052"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     inkscape:document-rotation="0"
     showgrid="true"
     inkscape:window-width="2880"
     inkscape:window-height="1526"
     inkscape:window-x="2869"
     inkscape:window-y="-11"
     inkscape:window-maximized="1"
     units="in"
     inkscape:object-paths="true" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title>graph_example_1</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-27.999999)">
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12"
       r="15"
       cy="65.399567"
       cx="25.132292" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-8"
       cx="25.132292"
       cy="115.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-81"
       cx="25.132292"
       cy="170.13229"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-2"
       cx="65.132294"
       cy="90.132294"
       r="15" />
    <circle
       style="opacity:0.25;mix-blend-mode:normal;fill:#00a400;fill-opacity:1;stroke:#000000;stroke-width:0.264583;stroke-opacity:0.190871"
       id="path12-1"
       cx="65.132294"
       cy="140.13229"
       r="15" />
    <text
       xml:space="preserve"
       style="font-size:11.2889px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.264583"
       x="9.4763451"
       y="35.342194"
       id="text924"><tspan
         sodipodi:role="line"
         id="tspan922"
         x="9.4763451"
         y="35.342194"
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:Garamond;-inkscape-font-specification:Garamond;stroke-width:0.264583">Graph 1</tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,169.33333 Z"
       id="path976" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 38.364583,71.437499 C 52.916666,82.020832 52.916666,82.020832 52.916666,82.020832"
       id="path984" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,79.374999 V 100.54167"
       id="path986" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 25.135416,156.10416 V 129.64583"
       id="path988" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 39.6875,170.65625 26.458332,-15.875"
       id="path990" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="m 64.822916,104.51042 v 21.16666"
       id="path992" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 64.822916,125.67708 V 104.51042"
       id="path898"
       inkscape:connector-type="polyline"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.264583px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 10.583333,170.65625 C 9.8617493,170.06016 9.1372967,169.4617 8.7270708,169.05263 8.316845,168.64357 8.2223497,168.42308 7.639781,167.28914 7.0572123,166.1552 5.9862798,164.10783 5.3245726,162.73767 c -0.6617072,-1.37015 -0.9136924,-2.06311 -1.0558682,-2.56723 -0.1421757,-0.50411 -0.1736747,-0.8191 -0.2995728,-1.66949 -0.1258981,-0.85039 -0.3463837,-2.2363 -0.1894062,-3.56007 0.1569774,-1.32377 0.6924437,-2.58369 0.9761861,-4.66189 0.2837424,-2.07821 0.3152405,-4.97603 0.3310037,-6.92889 0.015763,-1.95286 0.015763,-2.9608 0.078676,-4.39406 0.062913,-1.43325 0.1889049,-3.29163 0.2519842,-7.73276 0.063079,-4.44112 0.063079,-11.46518 0.094561,-17.54431 0.031482,-6.07913 0.094478,-11.21331 0.110268,-14.630807 0.01579,-3.417495 -0.015708,-5.118387 -0.1098928,-6.33087 C 5.4183272,91.50481 5.2608374,90.780357 5.1660964,90.166008 5.0713554,89.551658 5.0398574,89.04769 4.7881147,87.693411 4.536372,86.339131 4.0639021,84.134271 3.827338,82.606404 3.5907739,81.078537 3.5907739,80.228092 3.7481373,78.353802 3.9055006,76.479513 4.2204808,73.581695 4.3780975,72.038449 4.5357142,70.495204 4.5357142,70.306218 4.645771,70.053176 4.7558277,69.800134 4.976313,69.485155 5.1023128,69.233978 5.2283127,68.9828 5.2598108,68.793811 5.6226803,68.509003 5.9855498,68.224195 6.678506,67.846219 7.0888396,67.657484 c 0.4103337,-0.188735 0.5363259,-0.188735 0.724348,-0.267703 0.1880222,-0.07897 0.4400064,-0.236458 0.7085439,-0.330858 0.2685375,-0.0944 0.5520195,-0.125901 0.8977054,-0.267741 0.3456859,-0.14184 0.7551601,-0.393824 1.0080981,-0.519589 0.252938,-0.125765 0.347432,-0.125765 0.347431,-0.125765 -10e-7,0 -0.09449,0 -0.191633,0"
       id="path914"
       inkscape:path-effect="#path-effect916"
       inkscape:original-d="m 10.583333,170.65625 c -0.7218083,-0.59582 -1.4462626,-1.19428 -2.1733629,-1.79539 -0.091848,-0.21784 -0.1863421,-0.43833 -0.283482,-0.66146 -1.0682867,-2.04472 -2.1392192,-4.09209 -3.2127976,-6.14211 -0.2493383,-0.69031 -0.5013224,-1.38327 -0.7559524,-2.07887 -0.028852,-0.31233 -0.06035,-0.62732 -0.094494,-0.94494 -0.2178402,-1.38327 -0.4383264,-2.76918 -0.6614583,-4.15774 0.538112,-1.25727 1.0735784,-2.51719 1.6063986,-3.77976 0.034144,-2.89517 0.065642,-5.79299 0.094494,-8.69345 0.00265,-1.00529 0.00265,-2.01323 0,-3.02381 0.1286381,-1.85574 0.25463,-3.71412 0.3779764,-5.57515 0.00265,-7.02141 0.00265,-14.04547 0,-21.07217 0.065642,-5.13153 0.1286377,-10.26571 0.1889879,-15.402533 C 5.6407907,95.63062 5.6092925,93.929728 5.5751489,92.22619 5.4203044,91.50438 5.2628143,90.779927 5.1026784,90.052825 5.0738264,89.551504 5.0423283,89.047536 5.0081843,88.540922 4.53836,86.338705 4.0658899,84.133845 3.5907739,81.926339 c 0.00265,-0.847802 0.00265,-1.698247 0,-2.55134 0.3176259,-2.895171 0.632606,-5.792989 0.9449403,-8.693452 0.00265,-0.186343 0.00265,-0.375329 0,-0.566965 0.2231318,-0.312335 0.4436181,-0.627313 0.6614583,-0.94494 0.034144,-0.186341 0.065642,-0.37533 0.094494,-0.566965 0.6956023,-0.37533 1.3885585,-0.753306 2.0788691,-1.133928 0.1286377,0.0026 0.2546299,0.0026 0.377976,0 0.25463,-0.154845 0.5066141,-0.312335 0.7559524,-0.472469 0.2861281,-0.02885 0.5696101,-0.06035 0.8504465,-0.0945 0.4121201,-0.249338 0.8215945,-0.501322 1.2284225,-0.755952 0.09714,0.0026 0.191634,0.0026 0.283482,0 -0.09185,0.0026 -0.186342,0.0026 -0.283482,0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:0.146871px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,42.333333 H 86.420653"
       id="path937" />
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="19.597795"
       y="58.136337"
       id="text1038-3"><tspan
         sodipodi:role="line"
         id="tspan1036-9"
         x="19.597795"
         y="58.136337"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="20.544983"
       y="110.01562"
       id="text1038-3-2"><tspan
         sodipodi:role="line"
         id="tspan1036-9-8"
         x="20.544983"
         y="110.01562"
         style="stroke-width:0.10269">L = 2</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="18.927937"
       y="164.38885"
       id="text1038-3-27"><tspan
         sodipodi:role="line"
         id="tspan1036-9-3"
         x="18.927937"
         y="164.38885"
         style="stroke-width:0.10269">L = 1</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="59.938354"
       y="85.013847"
       id="text1038-3-278"><tspan
         sodipodi:role="line"
         id="tspan1036-9-2"
         x="59.938354"
         y="85.013847"
         style="stroke-width:0.10269">L = 3</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:4.38146px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.10269"
       x="60.058556"
       y="133.534"
       id="text1038-3-4"><tspan
         sodipodi:role="line"
         id="tspan1036-9-89"
         x="60.058556"
         y="133.534"
         style="stroke-width:0.10269">L = 3</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.73022px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.134302"
       x="22.933922"
       y="66.205948"
       id="text1210-9-4"><tspan
         sodipodi:role="line"
         id="tspan1208-8-0"
         x="22.933922"
         y="66.205948"
         style="stroke-width:0.134302">A</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.79841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.1359"
       x="24.467724"
       y="117.86392"
       id="text1214-3-0"><tspan
         sodipodi:role="line"
         id="tspan1212-7-9"
         x="24.467724"
         y="117.86392"
         style="stroke-width:0.1359">B</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.96357px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139771"
       x="62.986919"
       y="92.509705"
       id="text1222-5-5"><tspan
         sodipodi:role="line"
         id="tspan1220-4-2"
         x="62.986919"
         y="92.509705"
         style="stroke-width:0.139771">D</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.70841px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.13379"
       x="22.466202"
       y="172.88365"
       id="text1218-0-5"><tspan
         sodipodi:role="line"
         id="tspan1216-1-2"
         x="22.466202"
         y="172.88365"
         style="stroke-width:0.13379">C</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:6.73684px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.157894"
       x="63.249706"
       y="142.65382"
       id="text1226-8-0"><tspan
         sodipodi:role="line"
         id="tspan1224-5-7"
         x="63.249706"
         y="142.65382"
         style="stroke-width:0.157894">E</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.233006"
       y="75.055092"
       id="text1350-6"><tspan
         sodipodi:role="line"
         id="tspan1348-3"
         x="17.233006"
         y="75.055092"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.94351px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.139301"
       x="17.36664"
       y="125.32593"
       id="text1350-1"><tspan
         sodipodi:role="line"
         id="tspan1348-7"
         x="17.36664"
         y="125.32593"
         style="stroke-width:0.139301">{1,1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="17.231424"
       y="181.83562"
       id="text1354-2-8"><tspan
         sodipodi:role="line"
         id="tspan1352-5-4"
         x="17.231424"
         y="181.83562"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.930302"
       y="150.08495"
       id="text1354-2-50"><tspan
         sodipodi:role="line"
         id="tspan1352-5-7"
         x="59.930302"
         y="150.08495"
         style="stroke-width:0.140125">{1,1}</tspan></text>
    <text
       xml:space="preserve"
       style="font-size:5.97871px;line-height:1.25;font-family:Gadugi;-inkscape-font-specification:'Gadugi, Normal';stroke-width:0.140125"
       x="59.262127"
       y="100.05452"
       id="text1354-2-3"><tspan
         sodipodi:role="line"
         id="tspan1352-5-47"
         x="59.262127"
         y="100.05452"
         style="stroke-width:0.140125">{1,1}</tspan></text>
  </g>
</svg>


We can see that in the chart above, we have three different groups: 1,2 and 3.We can see that they both have the exact same <img src="https://latex.codecogs.com/gif.latex?L_i " /> values, with the exact same quantity. Therefore, the two graphs are isomorphic.

This turns out to have fairly wide applications spanning from cryptography, social media, and chemistry. In our example, we see this algorithm come up in context to the initiated graph neural networks.

## How does it related to GNN?
 
The purpose of the WL algorithm is to reduce the graph to its canonical form using the labeling of its neighbors. This is exactly what's being done by a GNN only in the case of the GNN, we are fitting a neural network in order to perform this. This adds an additional difference which is that the GNN utilizes differentiable operations in order to perform the grouping as opposed to WL which is discrete. However, both methods generate similar results.

## Implementation

I've provided most of the code for my Python implementation directly below. We begin our python implementation by defining a graph class in python:

```
# Purpose: The purpose of this script is to provide the structures necessary for working with graphs. 
# Comparison method taken from StackOverflow below:
#https://stackoverflow.com/questions/9623114/check-if-two-unordered-lists-are-equal#:~:text=Python%20has%20a%20built%2Din,the%20comparison%20will%20be%20unordered.&text=If%20you%20don't%20want,always%20be%20faster%20then%20collections.
import collections
from typing import Counter
# compare
# Inputs:
# • x/y - a list of elements to be reviewed
# Outputs:
# • TRUE/FALSE for whether the same elements and the same number of elements are included in both lists.
compare = lambda x, y: collections.Counter(x) == collections.Counter(y) 

# Get a list of labels for each node in the graph:
def get_neighbor_labels(graph_object):
    def get_node_neighbor_labels(graph_object,node_key):
        current_node = graph_object[node_key]
        current_neighbors = current_node["Neighbor"]    
        neighbor_labels = [graph_object[current_neighbors[x]]["Label"] for x in range(len(current_neighbors))     ]
        # We sort the details of each list to ensure that the orderings performed later work properly.
        neighbor_labels.sort()
        return(neighbor_labels)
    graph_with_label_values = {x:get_node_neighbor_labels(graph_object=graph_object, node_key = x) for x in graph_object.keys()        }
    return(graph_with_label_values)

def gen_new_labels(graph_results):
    # Convert each object generated by the counter function into a character object. This is for the purpose of matching later on.
    all_labels = [str(collections.Counter(graph_results[x])) for x in graph_results.keys()] 
    # Get the unique values from all_labels:
    unique_labels = [str(x) for x in collections.Counter(all_labels).keys()]
    # Get the range of labels from the unique_labels object. This is to be used to generate new indices for the results.
    new_labels = range(len(unique_labels))
    # Placeholder dictionary: This carries the new labels for each node in the graph that we've described above:
    new_labels_for_replacement = {x:"NA" for x in graph_results.keys()}
    # The function below uses the actual group type to generate new labels for each node. This number will be the same as the number of unique values.
    for i in range(len(unique_labels)):
        unique_ind = i
        unique_match_bools = [ unique_labels[i] == x for x in all_labels  ]
        new_label_keys = list(new_labels_for_replacement.keys())
        for orig_labels in range( len(new_labels_for_replacement)):
            if unique_match_bools[orig_labels]:
                new_labels_for_replacement[new_label_keys[orig_labels]]=i    
    return(new_labels_for_replacement)

# This function relabels a graph:
def relabel_graph(graph_for_relabeling):
    # Get the associated labels for each node of the graph:
    graph_labelings = get_neighbor_labels(graph_for_relabeling)
    # Generate new labelings for each node:
    new_labels = gen_new_labels(graph_labelings)
    # Relabel the original graph using the new_labels object:
    for cur_label in graph_for_relabeling.keys():
        graph_for_relabeling[cur_label]["Label"] = new_labels[cur_label]
    return(graph_for_relabeling)

# Quickly extract the labels of each graph node:
def get_labels(graph_object):
    ret_dictionary = {x:graph_object[x]["Label"] for x in graph_object.keys()}
    return(ret_dictionary)

# The purpose of the function below is to test whether the labels in the grpah have changed or not from the previous graph:
# previous_graph_iteration is the label from the previous graph that's being used.
# current_graph_iteration is the lables from the current graph
def change_label_test(current_graph_iteration, previous_graph_iteration):
    test_match = [previous_graph_iteration[x]==current_graph_iteration[x] for x in previous_graph_iteration.keys()]
    # Test to see if all labels are the same from the previous iteration:
    test_result = all(x for x in test_match)
    return(test_result)
 
# The final function used to generate the canonical form of the graph input. For the structure of the graph input that needs to be used, see the dictionary object defined below as graph_example_1:
def create_graph_canonical_form(graph_for_relabeling):
    i = 0
    print("Running iteration " + str(i) + ".")
    label_test_results = False
    while not(label_test_results):
        previous_graph_label = get_labels(graph_for_relabeling)
        current_graph = relabel_graph(graph_for_relabeling)
        current_graph_labels = get_labels(current_graph)
        label_test_results = change_label_test(current_graph_iteration=current_graph_labels,previous_graph_iteration=previous_graph_label)
        i+=1
        print("Running iteration " + str(i) + ".")
    return(current_graph)

# We define a simple example taken from the Weisfeiller-Lehman algorithm blog post. We can see that this generates the same form/labelings that were generated in that blog post:
graph_example_1 = {1:{"Label":1,"Neighbor":[2,3,5]},
              2:{"Label":1,"Neighbor":[1,3]},
              3:{"Label":1,"Neighbor":[1,2,4]},
              4:{"Label":1,"Neighbor":[3,5]},
              5:{"Label":1,"Neighbor":[1,4]}}

# When we run the create_graph_canonical form function on the example above, we see that we get the same labels as would be generated if we proceeded manually.
create_graph_canonical_form(graph_example_1)
```

The code above includes a graph_example_1 variable which represents Graph 1 from the example above. If we run it through the create_graph_canonical_form function we arrive at the graph labelings from the end of the simple example above:

{1: {'Label': 0, 'Neighbor': [2, 3, 5]}, 
 2: {'Label': 1, 'Neighbor': [1, 3]}, 
 3: {'Label': 0, 'Neighbor': [1, 2, 4]}, 
 4: {'Label': 2, 'Neighbor': [3, 5]}, 
 5: {'Label': 2, 'Neighbor': [1, 4]}}

# Further reading

I meant this to be a very tentative examination of this topic since other people have done a very complete job of workin through the problem and explaining it. I only aimed to provide a very simple example that one could use or work through for their own understanding. For a full rigorous discussion of it, there is the original paper [here](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf) . I also highly recommend the blog post  [here](https://tkipf.github.io/graph-convolutional-networks/) which kicked off my reading in this area. Finally, there are two very good blog posts [here](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/) and [here](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49) which discuss the theory more.



