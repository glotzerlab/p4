.. _publishing-with-p4:

==================
Publishing with p4
==================

How to Publish Models
+++++++++++++++++++++

Particle and interaction models can be published by saving them to JSON and
including those files in your supplementary information. If you publish models
in this way, please cite p4 and provide a link to the Github repository.


How to Publish Plots
++++++++++++++++++++

Plotly figures can be published in two ways:

* As interactive HTML files that you can include in your supplementary
  information
* As static images that you can include in your main text

To save a Plotly figure as an interactive HTML file, call its ``write_html()``
method. See examples here: https://plotly.com/python/interactive-html-export/.

.. code-block::

    figure.write_html("filename.html")

To save a Plotly figure as a static image file, you must install an additional
dependency. Follow the instructions in Plotly's official documentation:
https://plotly.com/python/static-image-export/.
