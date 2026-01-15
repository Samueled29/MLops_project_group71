# Building Container Images with Cloud Build

This project includes a **Cloud Build configuration** (`cloudbuild.yaml`) that allows you to build different container images using your Dockerfiles. You can manually build images for training or API usage by specifying the Dockerfile and other parameters.

### Training Container

```bash
gcloud builds submit \
--config cloudbuild.yaml \
--substitutions=_REGISTRY=my-registry,_IMAGE=fruit-train,_REGION=europe-west1,_DOCKERFILE=dockerfiles/train.dockerfile
```

### API Container

```bash
gcloud builds submit \
--config cloudbuild.yaml \
--substitutions=_REGISTRY=my-registry,_IMAGE=fruit-train,_REGION=europe-west1,_DOCKERFILE=dockerfiles/api.dockerfile
```