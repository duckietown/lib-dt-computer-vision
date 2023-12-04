FROM duckietown/dt-commons:ente

# working directory
WORKDIR /library

# configure environment
ENV DISABLE_CONTRACTS=1

# install requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# copy everything
COPY . .

# install dependencies
RUN python3 setup.py develop
