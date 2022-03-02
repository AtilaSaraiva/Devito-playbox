{ pkgs ? import "/home/atila/Files/Códigos/gitRepos/nixpkgs-devito" {} }:
let
  my-python-packages = python-packages: with python-packages; [
    numpy
    devito
    matplotlib
    jupyterlab
    # other python packages you want
  ];
  python-with-my-packages = pkgs.python3.withPackages my-python-packages;
in
pkgs.mkShell {
  buildInputs = [
    python-with-my-packages
    pkgs.ffmpeg
    pkgs.openmpi
  ];

  shellHook = ''
    export CFLAGS="-fopenmp"
    export LDFLAGS="-lgomp"
    export LD_PRELOAD="${pkgs.gcc9.cc.lib}/lib/libgomp.so"
    export DEVITO_ARCH="gcc"
    export DEVITO_LANGUAGE="openmp"
  '';
}
