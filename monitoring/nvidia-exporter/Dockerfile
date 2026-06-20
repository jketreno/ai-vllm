FROM python:3.12-slim
COPY exporter.sh /exporter.sh
RUN chmod +x /exporter.sh
EXPOSE 9835
CMD ["/exporter.sh"]
