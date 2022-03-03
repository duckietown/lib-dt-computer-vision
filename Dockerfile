FROM python:3.7

# working directory
WORKDIR /library

# configure environment
ENV DISABLE_CONTRACTS=1

# install requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# copy everything
COPY . .

# show list of files copied
RUN find .

# install dependencies
RUN pipdeptree
RUN python setup.py develop

