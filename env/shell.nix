with (import ./inputs.nix);
mach-nix.nixpkgs.mkShell {
  buildInputs = [
    (import ./python.nix)
    #mach-nix.mach-nix
  ];

  shellHook = ''
    jupyter lab
  '';
}
