# Project2_smiqbal

The `Project2_smiqbal` package is designed to simulate particles' movement and interactions under the influence of an electric field. This package allows users to visualize particle positions over time and assess their charge and mass distributions before and after interactions.

## Directory
packaging/
|-- LICENSE
|-- pyproject.toml
|-- README.md
|-- src/
|   |-- Project_2_smiqbal/
|       |-- __init__.py
|       |-- main.py
|
|-- tests/

## Testing Installation

To test if the package is installed correctly:

```python
from Project2_smiqbal import main
```

Run some basic functions to ensure everything works as expected:

```python
# Example: Testing the PDF and CDF functions:
result_pdf = main.pdf(1.0, 2.0)
print(f"PDF Result: {result_pdf}")

result_cdf = main.cdf(1.0, 2.0)
print(f"CDF Result: {result_cdf}")
```

## License

This project uses the MIT License. Check the `LICENSE` file in the project directory for more details.
```

---
