FROM ghcr.io/fedlearnnet/python-tool-api/base-python-dep:0.7.15.dev

USER 0

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R nonroot:nonroot /app /opt/venv \
 && chmod -R u+rwX,g+rwX /app

USER nonroot:nonroot

CMD ["python", "main.py"]