
# === CONFIG ===
APP_NAME=vision-api
PORT=8000

# === Build Docker Image ===
build:
	docker build -t $(APP_NAME) .

# === Run Docker Container (CPU) ===
run:
	docker run -p $(PORT):8000 $(APP_NAME)

# === Run Docker Container (GPU) ===
run-gpu:
	docker run --gpus all -p $(PORT):8000 $(APP_NAME)

# === Push to DockerHub (optional) ===
push:
	docker tag $(APP_NAME) your-dockerhub-username/$(APP_NAME)
	docker push your-dockerhub-username/$(APP_NAME)

# === Stop all containers (optional) ===
stop:
	docker stop $$(docker ps -q)

# === Clean build (optional) ===
clean:
	docker rmi $(APP_NAME)
