FROM nginx:1.27-alpine

RUN sed -i 's/^user  nginx;/user  root;/' /etc/nginx/nginx.conf

COPY default.conf /etc/nginx/conf.d/default.conf
