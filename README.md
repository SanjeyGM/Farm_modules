# Farm_modules

# 🐟 Aquafarming Advisor

A Streamlit-based application that helps aquafarmers monitor their systems and get expert advice based on sensor data and farming best practices from their operations manual.

## Features

- **Operations Manual Integration**: Upload your PDF operations manual to enable AI-powered advice specific to your system
- **Sensor Data Visualization**: Monitor pH and dissolved oxygen levels with interactive time series charts
- **Smart Alerts**: Automatic detection of parameters falling below ideal thresholds
- **AI-Powered Advice**: Get specific recommendations based on your sensor data and operations manual
- **Chat Interface**: Ask questions about aquafarming and get answers backed by your operations manual
- **Vector Search**: Efficient search through your manual for relevant information

## Screenshot

![Aquafarming Advisor Screenshot](https://via.placeholder.com/800x450)

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (optional, the application works in demo mode without it)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aquafarming-advisor.git
   cd aquafarming-advisor
   ```

2. Create a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Gemini API key (optional):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

   If you don't have a Gemini API key, the application will run in demo mode with simulated AI responses.

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Application Setup:
   - Upload your operations manual PDF from the sidebar
   - Load sample sensor data or upload custom JSON data
   - Adjust ideal thresholds for pH and dissolved oxygen if needed

4. Using the Dashboard:
   - View current system status with key metrics
   - Monitor sensor data trends with interactive charts
   - See identified issues and AI recommendations

5. Chat with the Advisor:
   - Ask questions about your aquafarming system
   - Get AI-powered answers backed by relevant sections from your manual

6. Raw Data View:
   - View and download your sensor data
   - Add new sensor readings manually

## Data Format

If you want to upload your own sensor data, use JSON format with this structure:

```json
[
  {
    "datetime": "2023-01-01 08:59:31",
    "ph_fish_tank": 6.0,
    "ph_biofilter": 6.3,
    "do_fish_tank": 5.0,
    "do_biofilter": 5.3
  },
  {
    "datetime": "2023-01-01 20:59:31",
    "ph_fish_tank": 6.2,
    "ph_biofilter": 6.2,
    "do_fish_tank": 3.6,
    "do_biofilter": 3.9
  }
]
```

## Getting a Gemini API Key

To enable full AI functionality:

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Sign up or log in with your Google account
3. Navigate to the API keys section
4. Create a new API key
5. Add the key to your `.env` file

## Folder Structure

The application automatically creates the following folders:

- `cache/`: Stores extracted text from uploaded PDFs
- `vector_cache/`: Stores vectorized chunks for efficient searching

## Demo Mode

If no API key is provided, the application runs in demo mode with:

- Simulated AI responses based on keywords in queries
- Full functionality for data visualization and threshold monitoring
- Sample demo data option

## Troubleshooting

- **PDF Extraction Issues**: If text extraction is poor, try using a PDF with searchable text rather than scanned images
- **API Key Not Working**: Verify that your Gemini API key is correct and has been properly set in the `.env` file
- **Performance Issues**: For large PDFs, the initial indexing might take longer

## License

[MIT License](LICENSE)

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Uses [Gemini AI](https://deepmind.google/technologies/gemini/) for intelligent recommendations
- PDF processing with [PyMuPDF](https://pymupdf.readthedocs.io/)
- Data visualization with [Plotly](https://plotly.com/)
