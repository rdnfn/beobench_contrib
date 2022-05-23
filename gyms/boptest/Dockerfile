# Beobench main docker file for experiments
# To build this use
# > docker build --no-cache -t beobench_boptest:latest \
#   https://github.com/rdnfn/beobench_contrib.git#:gyms/boptest

FROM python:3.9

# install docker CLI
RUN wget -O tmp/docker-ce-cli.deb https://download.docker.com/linux/debian/dists/bullseye/pool/stable/amd64/docker-ce-cli_20.10.11~3-0~debian-bullseye_amd64.deb \
    && dpkg -i tmp/docker-ce-cli.deb \
    && rm tmp/docker-ce-cli.deb

# use installer script to install BOPTEST which requires click
RUN pip install --no-cache-dir click
COPY installer.py tmp/
RUN python tmp/installer.py \
    && rm tmp/installer.py
