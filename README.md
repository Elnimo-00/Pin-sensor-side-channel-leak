PIN Side‑Channel ML Web Trainer

This repository contains a web‑based tool for training and using machine‑learning models to analyse side‑channel data (e.g. motion‑sensor readings) for PIN inference.  
It is based on the work from (https://github.com/matteonerini/pin-side-channel-attacks)

Disclaimer / Research Context

This project is intended for research / educational purposes. Real‑world applicability of side‑channel PIN inference depends heavily on data quality, sensor characteristics, environment, and more. Use and evaluate with caution.

---

Getting Started — Installation & Usage

These steps will get the project running locally on your machine.
1. Create a project folder and navigate into it
   
mkdir pin-ml-trainer
cd pin-ml-trainer

2. Clone this repository

git clone https://github.com/Elnimo-00/Pin-sensor-side-channel-leak

cd Pin-sensor-side-channel-leak

3. (Optional but recommended) Create a Python virtual environment
python -m venv venv

4. Activate the virtual environment
On Windows (PowerShell):
venv\Scripts\Activate.ps1
On Linux / macOS:
source venv/bin/activate

5. Install required dependencies
The requirements file is in the backend directory

pip install -r backend/requirements.txt

6. Start the server with auto‑reload for development from within the backend directory
   
uvicorn app:app --reload --port 9000
