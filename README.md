# Latlib


Any condensed matter physicist will be familiar with the pain of setting up 
and using a good indexing scheme for points on a periodic lattice. It's tough to get boundary conditions right, and even tougher to consider *disorder* (i.e. deletions) in a way that doesn't feel, well, hacky.

This becomes even more complicated when one wishes to also consider higher-dimensional geometrical objects than simple points, such as links, elementary plaquettes, and elementary cells.

This project aims to provide a **single, universal** solution to this problem,
defining a flexible, templated factory function `build_supercell` which allows the user to quickly and easily define arbitrary periodic structures in 3D (or lower).


# Installing

```bash
git clone https://github.com/Spuriosity1/lil2.git && cd lil2
meson setup build 
ninja -C build install
```

If you can't touch `/usr/local`, instead do
```bash
meson setup build -Dprefix="/your/install/prefix..."
```





