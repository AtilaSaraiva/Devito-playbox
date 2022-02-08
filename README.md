# Devito-playbox
A few of my adventures with [Devito](https://www.devitoproject.org/devito/index.html#).

This repository contains a few notebooks and scripts that will lead me in the road of learning this software. Everything here is completely reproducible through the use of the [nix package manager](https://nixos.org/explore.html). Nix will be able to create an distro agnostic environment with all the packages and python modules necessary to run the code written in this repo.

## Setting things up

To install nix either follow the [instalation instructions](https://nixos.org/download.html#nix-install-linux) or simply run this command if you're using Linux simply run:
```
https://nixos.org/download.html#nix-install-linux
```

To start the development environment simply run:
```
nix-shell ./env
```
while inside the cloned directory.

In a nutshell:
```
git clone https://github.com/AtilaSaraiva/Devito-playbox.git
cd Devito-playbox
nix-shell ./env
```
