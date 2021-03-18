# Simple Web Server
This is a simple web server that provides REST API access.

## Steps to debug model server locally
   
### 1. Clone the repository
```
git clone https://github.com/sophie128/simple-webserver
```
### 2. Customize your model function in model.py under the model folder and resolve dependencies
You only need to substitute the foo function in model.py using your own model logic.

### 3. Run the REST API server locally
```
python app.py
```
### 4. Test model function using REST API
If your input is a string list, you are expected to encode it into a single string and then decode it in your customized function. The predefined model interface only accepts a single input string. 
```
curl -i -X POST -H 'Content-Type: application/json' -d '{"question": <input your question here>}' http://localhost:80/answer
```
### 5. Update requirement.txt
If the REST API server works properly on your local machine, you can move forward with the following steps.
```
pip freeze > requirements.txt
```
### 6. Build the Docker Image
Make sure you have Docker desktop installed and sign-up on DockerHub.Then run the following command in the terminal inside the directory where the Dockerfile is located.
```
docker build -t <Your DockerHub user ID>/restapi:latest .
```
### 7. Push Docker Image into DockerHub
```
docker push <Your DockerHub user ID>/restapi:latest
```

## Steps to deploy model server on GCP
### 1. Login to VM instance (Ubuntu18.4 LTS preferred) on GCP 
```
git clone https://github.com/sophie128/simple-webserver
```
### 2. Install Docker
```
sudo apt install docker
```
### 3. Run model server as a docker container
```
sudo docker run --name webserver -td -p 3389:80 <Your DockerHub user ID>/restapi:latest
```
### 4. Use command to test model server
```
curl -i -X POST -H 'Content-Type: application/json' -d '{"question": <input your question here>}' http://<VM instance IP>:3389/answer
```
