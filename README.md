# Intelligent VM allocation



## Resource Optimization Simulation

This project simulates resource optimization for a cloud environment using a custom simulation framework. Follow the instructions below to create a Conda virtual environment, install dependencies, and run the simulation.

## Prerequisites
   - **Conda** (Anaconda or Miniconda must be installed)
   - The environment will use **Python 3.11.6**

## Setup Instructions\n\nRun the following commands in your terminal from the project's root directory:
1. **Create the Conda Virtual Environment**
    ```bash
   conda create --prefix ./venv python=3.11.6
    ```
3. **Activate the Virtual Environment**
    ```bash
    conda activate ./venv
    ```
5. **Install Dependencies**
    Ensure that the `requirements.txt` file is in your project directory, then install the required packages:
        ```bash
    pip install -r requirements.txt
   ```
7. **Run the Simulation**
    Execute the main simulation script by running:
      ```bash
    python main.py
      ```
9. **Deactivate the Environment (Optional)**
    When you are finished, deactivate the virtual environment:
      ```bash
    conda deactivate
      ```


## Project Structure
   - `main.py` 
   - The main simulation script.
   - `requirements.txt` - Contains a list of required packages not included in the default Python installation.
- `README.md` - This file.
## Requirements File
The `requirements.txt` file should include the following packages:
```numpy pandas matplotlib torch```
By following the above steps, you can set up your environment, install all necessary dependencies, and run the simulation to optimize resource utilization.
