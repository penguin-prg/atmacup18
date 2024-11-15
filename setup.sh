cd src
rm -rf Penguin-ML-Library
git clone https://github.com/penguin-prg/Penguin-ML-Library
cd Penguin-ML-Library
find . -maxdepth 1 ! -name 'penguinml' ! -name '.' -exec rm -rf {} +