{
  description = "Python flake";

  inputs.nixpkgs.url =
    "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";

  outputs = { self, nixpkgs }:
    let system = "x86_64-linux";
    in with import nixpkgs {
      inherit system;
    };

    let python = python39;
    in rec {

      devShell.${system} = pkgs.mkShell {

        packages = with python.pkgs; [
          ipython
          python

          deap
          numpy
          tqdm
        ];
      };
    };
}
