FROM osminogin/tor-simple

ARG SOCKET_START_PORT
ARG SOCKET_END_PORT
ARG CONTROL_START_PORT
ARG CONTROL_END_PORT

# Switch to root user if not already
USER root

ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 py3-pip py3-typer && ln -sf python3 /usr/bin/python
RUN #python3 -m ensurepip
RUN #pip3 install --no-cache typer setuptools

# Create directories for Tor data
RUN mkdir -p /var/lib/tor1 /var/lib/tor2 /var/lib/tor3 /var/lib/tor4

# Add Tor configuration files
ADD main.py .

EXPOSE $SOCKET_START_PORT-$SOCKET_END_PORT
EXPOSE $CONTROL_START_PORT-$CONTROL_END_PORT

# Start multiple Tor instances
CMD ["python", "main.py"]