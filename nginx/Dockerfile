FROM nginx:1.25.3

RUN apt-get update -y \
	&& apt-get install -y \
	apache2-utils \
	&& rm -rf /var/lib/apt/lists/*

ENV LISTEN_PORT=8080 \
	HTPASSWD_FILE="/etc/nginx/conf.d/auth.htpasswd" \
	FORWARD_PROTOCOL="http" \
	FORWARD_PORT=8501

WORKDIR /opt

COPY auth.htpasswd launch.sh ./

RUN chmod 0755 ./launch.sh

CMD ["./launch.sh"]
