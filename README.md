# CALMPY

A Python implementation of the CALM neural network using numpy. For an overview of CALM please see http://www.researchgate.net/publication/253278565_Overview_of_CALM_(draft) 

Network topologies are defined using JointJS via a Django website, while simulations are done using command-line Django shell or scripts.

## Quickstart

### Editing networks

1. Create a virtual environment and install packages listed under requirements.
2. Activate the virtual env and run a standard Django migration.
3. Edit DATA_DIR so that it points to an existing location at your drive.
4. Optionally load a sample network under tests/fixtures.
5. Run `python manage.py runserver` and navigate to http://127.0.0.1:8000/ which shows a stock interface. The only functional part is the section named "Available networks".
6. When in the network editor interface, use the control menu to define modules.

### Running simulations

1. Assumming you're using the sample network named 'simple', first ensure input patterns are defined. You can copy the 'simple' directory in tests/data to the DATA_DIR destination.
2. Then, in the terminal, start a shell with `python manage.py shell` and run, e.g:

```
    from simulator import load_network
    network = load_network('simple')
    network.train(50, 50)
    network.display()
    network.performance_check()
```

3. A 2D plot with categorization performance can be created with:

```
    from simulator.tools import *
    c = ConvergenceMap(network)
    c.display('in', 0, 1, 'out', 100)
```

Note: You need to install gnuplot for this. On OS X: `brew install --with-aquaterm --with-x11 --without-emacs --with-cairo --with-qt --with-pdflib-lite gnuplot`

(more documentation to follow)
