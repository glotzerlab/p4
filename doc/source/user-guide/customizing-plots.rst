.. _customizing-plots:

=================
Customizing Plots
=================

To customize plots beyond the options available in p4's various plotting
methods, you must work with Plotly's Python API. Basic examples of common tasks
are provided below.

Customizing Appearance
++++++++++++++++++++++

Title text and font (of axes and figures), figure sizes and margins, and other
non-data plot elements can be customized via the `layout API`_. 

.. _layout API: https://plotly.com/python/reference/layout/


Customizing the figure
----------------------

Customize figure size (same for 3D and 2D)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_3d.update_layout(width=300, height=300)
            figure_2d.update_layout(width=300, height=300)

Customize margin sizes (same for 3D and 2D)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_3d.update_layout(margin=dict(l=5, r=5, b=5, t=5, pad=4))
            figure_2d.update_layout(margin=dict(l=5, r=5, b=5, t=5, pad=4))

Customizing camera position (3D only)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_3d.update_layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=2, y=2, z=2)
                    )
                )
            )


Customizing the axes
--------------------

Customize axis title text (3D vs 2D)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_2d.update_layout(
                yaxis=dict(
                    title=dict(text="custom Y")
                )
            )

            figure_3d.update_layout(
                scene=dict(
                    yaxis=dict(
                        title=dict(text="custom Y")
                    )
                )
            )

Customize axis title font (3D vs 2D)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_2d.update_layout(
                yaxis=dict(
                    title=dict(
                        font=dict(weight=10, size=32)
                    )
                )
            )

            figure_3d.update_layout(
                scene=dict(
                    yaxis=dict(
                        title=dict(
                            font=dict(weight=10, size=32)
                        )
                    )
                )
            )

Customize axis limits (3D vs 2D)

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_2d.update_layout(
                yaxis=dict(range=[-1, 1])
            )

            figure_3d.update_layout(
                scene=dict(
                    yaxis=dict(range=[-1, 1])
                )
            )

Merging and Concatenating Plots
+++++++++++++++++++++++++++++++

To place data from one figure into another figure, use Plotly's
`Figure.add_trace()`_ method. (This was demonstrated
in :doc:`measuring-fields-body`.) Only 3D data can be added to 3D plots and
only 2D data can be added to 2D plots.

.. _Figure.add_trace(): https://plotly.com/python-api-reference/generated/generated/plotly.graph_objects.Figure.add_traces.html

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::

            figure_2d_a.add_trace(trace_2d_b)

To concatenate two figures together under a single parent figure, use Plotly's
`subplots API`_.

.. note::
    Plotly's subplots system uses 1-based indexing for rows and columns.

.. _subplots API: https://plotly.com/python/subplots/

.. tab-set::
    :sync-group: plotly-version

    .. tab-item:: Plotly 6.8.0
        :sync: plotly-6.8.0

        .. code-block::
            
            import plotly.subplots

            parent_figure = plotly.subplots.make_subplots(rows=1, cols=2)
            parent_figure.add_trace(trace_3d, row=1, col=1)
            parent_figure.add_trace(trace_2d, row=1, col=2)
