FROM nginx:alpine

# The base image renders environment variables from this template at startup.
COPY default.conf.template /etc/nginx/templates/default.conf.template

# Expose port 80
EXPOSE 80

# Run nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
