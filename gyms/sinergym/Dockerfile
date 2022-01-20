# Beobench version of sinergym dockerfile
# The first 60 lines are copied from
# https://github.com/jajimer/sinergym/blob/24a37965f4e749faf6caaa3d4ece95330a478904/Dockerfile
# Licensed under MIT License, copy of original license is in the folder of this dockerfile.

# Base on nrel/energyplus from Nicholas Long but using
# Ubuntu, Python 3.6 and BCVTB
ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION}

# Configuring tzdata in order to don't ask for geographic area
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Arguments for EnergyPlus version (default values of version 8.6.0 if is not specified)
ARG ENERGYPLUS_VERSION=9.5.0
ARG ENERGYPLUS_INSTALL_VERSION=9-5-0
ARG ENERGYPLUS_SHA=de239b2e5f

# Argument for Sinergym extras libraries
ARG SINERGYM_EXTRAS=[extras]

ENV ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu18.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu18.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME

# Collapse the update of packages, download and installation into one command
# to make the container smaller & remove a bunch of the auxiliary apps/files
# that are not needed in the container
RUN apt-get update \
    && apt-get install -y ca-certificates curl libx11-6 libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf DataSets Documentation ExampleFiles WeatherData MacroDataSets PostProcess/convertESOMTRpgm \
    PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

# Remove the broken symlinks
RUN cd /usr/local/bin find -L . -type l -delete

# Install Python
RUN apt-get update && echo "Y\r" | apt-get install python3.6 && echo "Y\r" | apt-get install python3-pip

# Install OpenJDK-8
RUN apt-get update && echo "Y\r" | apt-get install default-jre openjdk-8-jdk

# Install BCVTB
ENV BCVTB_PATH=/usr/local/bcvtb
RUN apt-get install wget \
    && wget http://github.com/lbl-srg/bcvtb/releases/download/v1.6.0/bcvtb-install-linux64-v1.6.0.jar \
    && yes "1" | java -jar bcvtb-install-linux64-v1.6.0.jar \
    && cp -R 1/ $BCVTB_PATH && rm -R 1/

# Working directory and copy files
RUN pip3 install --upgrade pip
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

# End of copy from dockerfile in sinergym repo

# Custom beobench code

# Make python 3.6 default
RUN ln -s /usr/bin/python3.6 /usr/bin/python

# Installing sinergym from GitHub
#
# Note that sinergym currently needs to be installed from cloned repo,
# likely because sinergym's setup.py does not specify package names.
# The -e option ensures that the entire package path is added to the
# PYTHONPATH to address the above problem.
RUN mkdir -p /opt/beobench/ \
    && cd /opt/beobench/ \
    && git clone https://github.com/jajimer/sinergym.git \
    && cd sinergym && python3 -m pip install -e .


# add env creator
COPY env_creator.py /opt/beobench/experiment_setup/
# add env creator to python path
ENV PYTHONPATH "${PYTHONPATH}:/opt/beobench/experiment_setup/"

# install beobench package
RUN python3 -m pip --disable-pip-version-check --no-cache-dir install beobench

# Fix for click encoding problem
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8