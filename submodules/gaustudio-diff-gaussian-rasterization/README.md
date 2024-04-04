# Differential Gaussian Rasterization for [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio)
This software is used as the rasterization engine in [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio), and supports:

* Analytical gradient for rendered opacity.
* Inference median depthf thanks to [JonathonLuiten](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth)
* Analytical gradient for rendered depth thanks to [ingra14m](https://github.com/ingra14m/depth-diff-gaussian-rasterization) and[slothfulxtx](https://github.com/slothfulxtx).

The code is built on top of the original Differential Gaussian Rasterization used in "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields".

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>