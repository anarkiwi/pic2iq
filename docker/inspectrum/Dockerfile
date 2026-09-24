FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    inspectrum xvfb x11-apps imagemagick xdotool fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY screenshot.sh /usr/local/bin/screenshot.sh
ENTRYPOINT ["/usr/local/bin/screenshot.sh"]
