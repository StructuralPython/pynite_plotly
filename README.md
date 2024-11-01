# pynite_plotly: Plotly visualization for PyNiteFEA

## Installation

```
pip install pynite-plotly
```

## Basic Usage

`pynite-plotly` can be used as a drop-in replacement for PyNite's own PyVista rendering module.

So, simply replace:

```python
from PyNite.Rendering import Renderer
```

with this:

```python
from pynite_plotly import Renderer
```

And then visualize your model according to the API for the PyNite renderer:

```python
from PyNite import FEModel3D

model = FEModel3D()

### ... build your model here...

vis = Renderer(model)
vis.render_model()
```

## Additional Features
The `Renderer` class has a few additional attributes beyond the attributes in the PyNite `Renderer`. These are useful for modifying colors and line weights of the elements in the plot.

* `.colors` - A `dict` that has keys corresponding to the different elements in the plot
* `.line_widths` - A `dict` that has keys corresponding to the different linear elements in the plot

Change the values of the keys and run `.update` to see how they affect the plot!

## Current Limitations

2024-11-01: I have only implemented rendering frames and loads. I have not implemented plates/quads/area loads. It will take a special effort because plotly renderers meshes (the plates) as triangles but PyNite uses a quad mesh for FEA. 

So, to render everything like PyVista will require remeshing the quad mesh as triangles and having the ability to plot the lines of the quad mesh overtop of the triangulated quad mesh which can also have gradient shading according to the results being plotted.

Additionally, I have not implemented plotting of nodes and node labels yet. No real reason but I noticed that they were not implemented in the PyNite Renderer so I just kept chill on it...for now.


## Motivation

[PyNiteFEA](https://github.com/jwock82/pynite) is excellent and has been becoming more so. In v0.0.94 @JWock82 released PyVista visualization to complement the existing VTK visualization. This has been a great improvement in usability since PyVista can run within a Jupyter notebook as opposed to launching a separate operating system window for the visualization (VTK).

To get PyVista running in Jupyter, requires Trame and a load of other dependencies. These dependencies typically lag behind the latest Python version. Additionally, PyVista does not run everywhere on the web yet (like streamlit).

However, plotly has similar 3D plotting capability to PyVista, runs everywhere, and has a light dependency load.

So, using a helper library I created, [plotly_3d_primitives](https://github.com/structuralpython/plotly_3d_primitives), I then copied the original PyNite `Rendering` module and swapped out the PyVista method names with my new plotly function names. Did a little bit of massaging and...voila! A new rendering module for PyNite!

