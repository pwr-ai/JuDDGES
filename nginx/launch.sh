#!/bin/sh

envsubst < auth.htpasswd > /etc/nginx/auth.htpasswd

htpasswd -c -b /etc/nginx/auth.htpasswd $USER $PASS

echo basic-auth-pwd
cat /etc/nginx/auth.htpasswd

nginx -g "daemon off;"
