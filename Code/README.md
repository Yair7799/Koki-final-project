*Project Setup Guide*
This guide provides step-by-step instructions on how to set up and run the project locally. Please follow these steps to ensure everything is configured correctly.

Prerequisites
Before you begin, make sure you have the following installed:

Anaconda (Recommended) or Linux Terminal
Python 3.x (Ensure that Python is installed and added to your system's PATH)
Setup Instructions
1. Open Terminal
Open an Anaconda Prompt or a Linux Terminal. Using Anaconda is recommended as it simplifies package management and virtual environment handling.

2. Navigate to the Project Directory
Use the cd command to navigate to the Code directory within the project. Replace the following example path with the path to your project directory:

bash
Copy code
cd C:\your-own-path-to-the-folder\Koki-final-project\Code
3. Install Required Packages
Once inside the Code directory, install all the necessary Python libraries by running the following command:

bash
Copy code
pip install -r requirements.txt
Note: If you wish to add any additional libraries, you can simply add the library's name to the requirements.txt file.

4. Run the Application
After the required packages have been installed, run the application with the following command:

bash
Copy code
python app.py
This command will start the application, and you will be provided with a local address where the app is running. Open this address in your web browser to interact with the application.