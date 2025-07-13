# Dependency provider
FROM ubuntu:latest AS build-deps-provider
COPY requirements.txt /tmp/requirements.txt

# -----------------------------------------------------------------------------
# FastCOMICS
FROM ubuntu:latest AS fastcomics

RUN apt update && apt upgrade -y \
	&& apt install -y \
		vim \
		tree \
		git \
		zip \
		unzip \
		python3 \
		python3-pip \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,from=build-deps-provider,source=/tmp/requirements.txt,target=./requirements.txt \
	pip3 install -r ./requirements.txt
