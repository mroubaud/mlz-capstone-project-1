# FROM emacski/tensorflow-serving:latest-linux_arm64
FROM bitnami/tensorflow-serving:latest
COPY horses_vs_humans /models/horses_vs_humans/1
ENV MODEL_NAME="horses_vs_humans"