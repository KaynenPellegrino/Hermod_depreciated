# Installation Guide

This guide provides detailed instructions on how to set up the Hermod project on your local machine for development and testing purposes.

## Prerequisites

- **Operating System:** Linux, macOS, or Windows
- **Python 3.8 or higher**
- **Git**
- **Virtual Environment (optional but recommended)**

## Steps

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/yourusername/Hermod.git
cd Hermod
```
### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

``` bash
python3 -m venv venv
```
Activate the virtual environment:

On Linux/macOS:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```
### 3. Install Dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```
### 4. Set Up Environment Variables
Create a .env file in the project root directory and add necessary environment variables. For example:
```env
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
```
### 5. Initialize the Database (if applicable)
If using a database, run migrations:
```bash
python src/manage.py migrate
```
### 6. Run the Application
Start the development server:
```bash
python src/manage.py runserver
```
The application should now be running at http://localhost:8000/.

### 7. Verify Installation
Open your web browser and navigate to http://localhost:8000/ to see the Hermod homepage.