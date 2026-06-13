.. _advanced-topics:

===============
Advanced Topics
===============

Customizing Plot Appearance
+++++++++++++++++++++++++++

To customize plots beyond the options available in
:py:func:`p4.plot_layout` and the various class plotting methods, you must work
with Plotly's classes and methods. Title text and font (of axes and figures),
plot sizes and margins, and other non-data plot elements can all be customized
via the `layout API`_. Basic examples of these tasks are provided below.

.. _layout API: https://plotly.com/python/reference/layout/


Customizing the plot
--------------------

Customize plot size (same for 3D and 2D)

.. code-block::

    figure_3d.update_layout(width=300, height=300)

Customize margin sizes (same for 3D and 2D)

.. code-block::

    figure_3d.update_layout(
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=4
        )
    )

Customizing camera position (3D only)

.. code-block::

    figure_3d.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(
                    x=2,
                    y=2,
                    z=2
                )
            )
        )
    )


Customizing the axes
--------------------

Customize axis title text (3D vs 2D)

.. code-block::

    figure_2d.update_layout(
        yaxis=dict(
            title=dict(
                text="custom Y"
            )
        )
    )

    figure_3d.update_layout(
        scene=dict(
            yaxis=dict(
                title=dict(
                    text="custom Y"
                )
            )
        )
    )

Customize axis title font (3D vs 2D)

.. code-block::

    figure_2d.update_layout(
        yaxis=dict(
            title=dict(
                font=dict(
                    weight=10,
                    size=32
                )
            )
        )
    )

    figure_3d.update_layout(
        scene=dict(
            yaxis=dict(
                title=dict(
                    font=dict(
                        weight=10,
                        size=32
                    )
                )
            )
        )
    )

Customizing axis limits (3D vs 2D)

.. code-block::

    figure_2d.update_layout(
        yaxis=dict(
            range=[-1, 1]
        )
    )

    figure_3d.update_layout(
        scene=dict(
            yaxis=dict(
                range=[-1, 1]
            )
        )
    )

Merging and Concatenating Plots
+++++++++++++++++++++++++++++++

To place data from one figure into another figure, use Plotly's
`Figure.add_trace()`_ method. (This was demonstrated
in :doc:`measuring-fields-body`.) Only 3D data can be added to 3D plots and
only 2D data can be added to 2D plots.

.. _Figure.add_trace(): https://plotly.com/python-api-reference/generated/generated/plotly.graph_objects.Figure.add_traces.html)

.. code-block::

    figure_2d_a.add_trace(trace_2d_b)

To concatenate two figures together under a single parent figure, use Plotly's
`subplots API`_.

.. note::
    Plotly's subplots system uses 1-based indexing for rows and columns.

.. _subplots API: https://plotly.com/python/subplots/

.. code-block::
    
    import plotly.subplots

    parent_figure = plotly.subplots.make_subplots(rows=1, cols=2)
    parent_figure.add_trace(trace_3d, row=1, col=1)
    parent_figure.add_trace(trace_2d, row=1, col=2)


Automating Workflows
++++++++++++++++++++

TODO

Demonstrate Signac integration

This introduces JSON serialization, which is then picked up in the next
section.


How to Publish Models
+++++++++++++++++++++

Particle and interaction models can be published by saving them to JSON and
including those files in your supplementary information. If you publish models
in this way, please include a citation and link for p4.


How to Publish Plots
++++++++++++++++++++

Plotly figures can be published in two ways:

* As interactive HTML files that you can put in your supplementary information
* As static images that you can put in your main text

To save a Plotly figure as an interactive HTML file, call its ``write_html()``
method. See examples here: https://plotly.com/python/interactive-html-export/.

.. code-block::

    figure.write_html("filename.html")

To save a Plotly figure as a static image file, you must install an additional
dependency. Follow the instructions in Plotly's official documentation:
https://plotly.com/python/static-image-export/.
