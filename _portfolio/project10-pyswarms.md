---
title: "PySwarms - Open Source Software Contribution"
excerpt: "To learn more about collaborative development with version control systems as well as contributing to open source projects on GitHub, I made two contributions to the PySwarms project. PySwarms is a Python module for particle swarm optimization research.<br/><img src='/images/pyswarms/eggholder.jpg'>"
collection: portfolio
---

To learn more about collaborative development with version control systems as well as contributing to open source projects on GitHub, I made two contributions to the PySwarms project. PySwarms is a Python module for particle swarm optimization research.

# Tools Used

* Python
* Jupyter Notebooks
* Unit Testing with PyTest
* Git & GitHub

# My Contributions

I made the following contributions to the project:
* [Improved the Plotter Module](#improved-the-plotter-module)
* [Added Objective Functions](#added-objective-functions)

## Improved The Plotter Module

[View Pull Request on GitHub.](https://github.com/ljvmiranda921/pyswarms/pull/172){:target="_blank"}

Bug fix: Added 'z-axis' to formatters/Designer attribute 'labels' to prevent index out of range error when plotting in 3D. Also added a third dimension to 'limits' for the same reason.

New feature: I added a colormaps property to the Designer class in formatters.py to allow the user to choose whichever one they want. It uses the same style as other properties of the class, and the module imports matplotlib.cm to validate both types of colormaps in matplotlib. I edited plotters/plot_countour to initialize a Designer object with a default colormap if none is provided. I've tested with every category of colormap as well as custom colormaps and verified that it is working as intended.

Default parameter change: Mesher.delta from 0.001 to 0.1. This makes it safe to run many of the functions on an average computer. With 0.001, the mesher object grows very quickly.

[](/images/pyswarms/ploter.jpg)

## Added Objective Functions

[View Objective Function Commit on GitHub.](https://github.com/ljvmiranda921/pyswarms/pull/168/commits/b5a3afdb6a3087cce64ec08f554ae034936eb553){:target="_blank"}

[View Pull Request on GitHub.](https://github.com/ljvmiranda921/pyswarms/pull/168)

[Download a demo notebook.](/images/pyswarms/Objective Functions Demo.ipynb)

I cleaned up and fixed existing functions with TODO tags and implemented Cross-in-Tray, Easom, Eggholder, Himmelblau's, Holder Table, and Three Hump Camel functions. The file was getting long so I also alphabetized the functions and made a simple table of contents. I also added unit tests for the new objective functions.

### Cross-in-Tray Function

![](/images/pyswarms/cross-in-tray.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/cross_in_tray.mp4" type = "video/mp4" > frameborder="0" style="border:0"></iframe>
</div>

### Eggholder Function

![](/images/pyswarms/eggholder.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/eggholder.mp4" type = "video/mp4"  > frameborder="0" style="border:0"></iframe>
</div>

### Easom Function

![](/images/pyswarms/easom.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/easom.mp4" type = "video/mp4" > frameborder="0" style="border:0"></iframe>
</div>

### Himmelblau Function

![](/images/pyswarms/himmelblau.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/himmelblau.mp4" type = "video/mp4"  > frameborder="0" style="border:0"></iframe>
</div>

### Holder Table Function

![](/images/pyswarms/holder_table.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/holder_table.mp4" type = "video/mp4"  > frameborder="0" style="border:0"></iframe>
</div>

### Three Hump Camel Function

![](/images/pyswarms/three_hump_camel.png)

<div class="flexible-container">
<iframe src = "/images/pyswarms/three_hump_camel.mp4" type = "video/mp4">  frameborder="0" style="border:0"></iframe>
</div>
