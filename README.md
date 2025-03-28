# MPCDC - Machine Learning Clustering Model Web Application

This web application provides a user interface for interacting with a Databricks LLM model that interprets insights from a machine learning clustering model trained on datasets related to changes, incidents, and services within an organization.

## Features

- Main page with project description
- Interactive chatbot in the right sidebar
- Integration with Databricks LLM API
- Responsive design
- Demo mode with predefined responses when Databricks token is not available

## Setup Instructions

### Option 1: Local Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure the Databricks API token:
   - Open the `.env` file
   - Replace `your_databricks_token_here` with your actual Databricks token
   - If no valid token is provided, the application will run in demo mode with predefined responses

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

### Option 2: Docker Setup

1. Configure the Databricks API token:
   - Open the `.env` file
   - Replace `your_databricks_token_here` with your actual Databricks token
   - If no valid token is provided, the application will run in demo mode with predefined responses

2. Build and start the Docker containers:
   ```
   docker-compose up -d --build
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000/
   ```

4. To stop the containers:
   ```
   docker-compose down
   ```

### Option 3: Kubernetes Setup

1. Build and push the Docker image to your registry:
   ```
   # On Linux/Mac
   ./build-and-push.sh
   
   # On Windows
   build-and-push.bat
   ```
   Note: Edit the script first to set your registry details.

2. Update the Kubernetes manifests as needed:
   - Edit `kubernetes/deployment.yaml` to use your image repository
   - Edit `kubernetes/ingress.yaml` to use your domain
   - Edit `kubernetes/secret.yaml` to use your base64-encoded Databricks token

3. Apply the Kubernetes manifests:
   ```
   kubectl apply -k kubernetes/
   ```

4. Access the application through your ingress domain or port-forward for testing:
   ```
   kubectl port-forward svc/mpcdc-service 8080:80
   ```
   Then visit http://localhost:8080 in your browser.

5. To delete the deployment:
   ```
   kubectl delete -k kubernetes/
   ```

## Demo Mode

When no valid Databricks token is provided, the application will run in demo mode with predefined responses. In this mode:

- A notification banner will indicate that the app is running in demo mode
- The chatbot will respond with predefined messages based on keywords in your input
- Try asking about "infrastructure", "deployment", or "security" changes to see different responses

To switch to full functionality with the Databricks LLM:
1. Obtain a valid Databricks API token
2. Add it to the `.env` file
3. Restart the application

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: HTML template for the web application
- `static/css/style.css`: CSS styles
- `static/js/chatbot.js`: JavaScript for chatbot functionality
- `.env`: Environment variables (Databricks token)
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker container configuration
- `docker-compose.yml`: Docker Compose configuration
- `.dockerignore`: Files to exclude from Docker build
- `kubernetes/`: Kubernetes deployment files
  - `deployment.yaml`: Kubernetes deployment configuration
  - `service.yaml`: Kubernetes service configuration
  - `configmap.yaml`: ConfigMap for non-sensitive configuration
  - `secret.yaml`: Secret for sensitive data
  - `ingress.yaml`: Ingress for external access
  - `pvc.yaml`: Persistent Volume Claim for storage
  - `kustomization.yaml`: Kustomize configuration
- `build-and-push.sh`: Script to build and push Docker image (Linux/Mac)
- `build-and-push.bat`: Script to build and push Docker image (Windows)

## Notes

- The chatbot connects to a Databricks LLM endpoint for inference when a valid token is provided
- The system prompt is configured to provide insights about changes and incidents based on clustering models
- The application requires a valid Databricks API token for full functionality, but can run in demo mode without it
