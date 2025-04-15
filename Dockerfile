FROM python:3.9
RUN pip install pandas wikipedia beautifulsoup4 requests numpy scikit-learn jupyter xgboost lxml

WORKDIR /ivado

COPY data/ data
COPY src/ src
COPy tst/ tst
# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]