"""Helper package for the Intro to Scaled Qualitative Analysis notebook.

Four modules, one per stage of the workflow:

* ``normalize`` -- raw transcript to CMAP rows, and the schema validator
* ``llm_handoff`` -- the code vocabulary, the dictionary lane, and the
  large-language-model handoff
* ``viz`` -- the word cloud and the code co-occurrence heatmap, recycled from
  the CMAP Visualization Toolkit (BSD 3-Clause)
* ``header`` -- the plain-ASCII run header

This file is also the sentinel the notebook looks for when it works out where
it is running. If it can find this file by walking up from the working
directory, everything it needs is already on disk and it does not clone.
Submodules are imported on demand rather than here, so that the sentinel check
stays cheap and does not require the plotting stack to be installed.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
