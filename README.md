# Agri Chatbot

Agri Chatbot is an interactive application designed to assist users with agricultural queries and to enable easy access to the latest research for everyone. This application is built using Streamlit and was developed during the EcoHack Hackathon hosted by Zentrum für interdisziplinäre Forschung (ZiF) in Bielefeld.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ECOHack_chatbot.git
    cd ECOHack_chatbot
    ```
2. Create and activate virtual environment:
    ```bash
    python -m venv agri_chat_env
    source agri_chat_env/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Export groq API key

The Agri Chatbot application needs an API key for groq LLM services (https://console.groq.com/keys) to run the LLM models.
The API key should be made available as environment variable 'groq_api_key' as follows:
```bash
groq_api_key=YOUR_GROQ_API_KEY
export groq_api_key
```

### Running the Application

To start the Agri Chatbot application, run the following command:
```bash
streamlit run app.py
```

This will launch the application in your default web browser.

### Usage

Once the application is running, you can interact with the chatbot by typing your queries into the input box and pressing Enter.

### Contributors
Vamsi Krishna Kommineni (Friedrich Schiller University Jena), Anne Peter (Johann Heinrich von Thünen Institut), Caren Daniel (Independent Data Scientist) and Alexander Espig (Leibniz University Hannover)

### License

This project is licensed under the MIT License.
