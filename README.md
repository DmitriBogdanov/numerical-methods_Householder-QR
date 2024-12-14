[![Language](https://img.shields.io/badge/C++-std=17-blue.svg?style=flat&logo=cplusplus)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DmitriBogdanov/prototyping_utils/blob/master/LICENSE.md)

# Project name

This project is a part of a numerical analysis course. It implements following algorithms:

- QR-factorization using Householder reflections
- Modification of QR-factorization for Linear Least Squares (LLS) problem
- LLS solver using QR-decomposition method
- Backwards Gaussian Elimination
- Hessenberg reduction using Householder reflections
- Modification of QR-factorization for upper-hessenberg matrices
- Regular QR-algorithm
- Shifted QR-algorithm with Hessenberg decomposition

## Documentation

See `report/report.pdf` for implementation notes (ru). Most functions include proper in-code documentation (en).

## Requirements

- Requires C++17 support

## Dependencies

### Firstparty

- [prototyping_utils](https://github.com/DmitriBogdanov/prototyping_utils)

### Thirdparty

- [Eigen](https://eigen.tuxfamily.org/index.php)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/DmitriBogdanov/prototyping_utils/blob/master/LICENSE.md) for details
