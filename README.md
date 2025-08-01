# PPG-BPNet

Advanced Blood Pressure Monitoring using Deep Learning and Photoplethysmography (PPG)

## Project Overview
PPG-BPNet is a modern web application for non-invasive, real-time blood pressure estimation using advanced deep learning and PPG signals. It features:
- File-based and live monitoring interfaces
- Real-time data visualization
- Flask backend with Socket.IO for live updates
- Responsive, animated frontend (HTML/CSS/JS)

## Features
- Upload PPG signal files for analysis
- Start/stop live monitoring (simulated data)
- Real-time BP trend charts and status
- Modern UI with Plotly.js and GSAP animations

## Local Development

### Prerequisites
- Python 3.8+
- Node.js (for frontend development, optional)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Install Node dependencies for frontend
```bash
npm install
```

### 3. Run the Flask app locally
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Deployment on Render

1. **Push your code to GitHub**
2. **Create a new Web Service on [Render](https://render.com/)**
3. **Set the build and start commands:**
   - No build command needed for Flask
   - Render will use your `Procfile` automatically:
     ```
     web: gunicorn --worker-class eventlet -w 1 app:app
     ```
4. **Set environment variables** (if needed) in the Render dashboard
5. **Deploy!**

## File Structure
- `app.py` — Flask backend with Socket.IO
- `static/` — CSS, JS, and assets
- `templates/` — HTML templates
- `requirements.txt` — Python dependencies
- `Procfile` — Render deployment process
- `package.json` — (Optional) Node.js dependencies for frontend

## Notes
- The current version uses mock data for demonstration. Replace the mock data sections in `app.py` with your actual PPG-BP model and device integration for production use.
- For questions or contributions, open an issue or pull request.
